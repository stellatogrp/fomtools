"""Paper and talk quality renderer for the closed-loop flights.

A port of the quadcopter renderer in the learning-to-warm-start code
(``l2ws/examples/quadcopter_render.py``): a stylised quadcopter mesh with a body,
four arms, rotor rims and spinning blades; a fading position trail; a shadow of
the body and rotors on a gridded floor; the reference as a dashed curve; and a
camera that orbits slowly, shared by every pane so the panes stay comparable.

Two things differ from the original.  Orientation comes from the Euler angles
in the linear model's state rather than from a quaternion, and the frame that
maps them to a rotation is fixed by the dynamics: state 0 drives the ``x``
acceleration (``A_DYN[3, 0] > 0``), so it is the pitch-like angle and rotates
about the body ``y`` axis; state 1 drives ``y`` (``A_DYN[4, 1] < 0``), so it is
the roll-like angle and rotates about ``x``.  And the scene is framed on the
flights themselves with the floor at the ground, because these flights take off
from the origin rather than starting on the path.

Frames are rendered one image at a time and stitched, as ``plots`` always did;
the GIF frame delay is given in milliseconds, which is what ``imageio`` expects.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

from common.quadcopter import ATT  # noqa: E402
from common.simulate import GROUND, Rollout  # noqa: E402

# Helvetica Neue where it is installed (macOS), with ordinary fallbacks.  Applied
# through an ``rc_context`` around each render rather than globally, so the
# two-dimensional plots in the same process keep their own font.
_FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK + list(matplotlib.rcParams.get("font.sans-serif", [])),
}
#: The still is set in LaTeX so its text matches a paper; the animation keeps
#: the sans-serif stack above.
_RC_LATEX = {"text.usetex": True, "font.family": "serif"}


def budget_title(budget_us: float) -> str:
    """The figure-level header naming the compute budget the panes were flown at."""
    return rf"Iterations per {budget_us:g}\,$\mu$s Time Budget"


# ----------------------------------------------------------------------------
# Visual constants -- tweak here, not in the drawing code below.
# ----------------------------------------------------------------------------
BODY_COLOR = "#1a1a1f"
ARM_COLOR = "#2c2c34"
ROTOR_HUB_COLOR = "#3a3a44"
ROTOR_RIM_COLOR = "#5a5a66"
BLADE_COLOR = "#cfcfd6"
BLADE_BLUR_COLOR = "#bdbdc6"
SHADOW_COLOR = "#000000"
FLOOR_COLOR = "#f4f4f6"
FLOOR_GRID_COLOR = "#cdd0d6"
REF_COLOR = "#0d0d10"
ERR_BAR_BG = "#e8e8ec"
TEXT_COLOR = "#1a1a1f"
CRASH_COLOR = "#c0392b"

CAMERA_ELEV = 22.0
CAMERA_AZIM = -65.0


# ----------------------------------------------------------------------------
# Orientation
# ----------------------------------------------------------------------------
def euler_to_R(attitude: np.ndarray) -> np.ndarray:
    """Body-to-world rotation from the three attitude states.

    ``attitude[0]`` is the pitch-like angle (about body ``y``, drives ``x``),
    ``attitude[1]`` the roll-like angle (about body ``x``, drives ``y``), and
    ``attitude[2]`` the yaw.  To first order ``R e_z = (theta, -phi, 1)``, which
    is the direction the linear model accelerates in: ``a_x = +g theta`` and
    ``a_y = -g phi``.  ``check_render`` verifies this against a flown trajectory.
    """
    theta, phi, psi = (float(a) for a in attitude)
    c, s = np.cos, np.sin
    rz = np.array([[c(psi), -s(psi), 0.0], [s(psi), c(psi), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[c(theta), 0.0, s(theta)], [0.0, 1.0, 0.0], [-s(theta), 0.0, c(theta)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c(phi), -s(phi)], [0.0, s(phi), c(phi)]])
    return rz @ ry @ rx


# ----------------------------------------------------------------------------
# Geometry helpers (verbatim from l2ws)
# ----------------------------------------------------------------------------
def _box_faces(center: np.ndarray, R: np.ndarray, size: tuple[float, float, float]):
    """6 face polygons of an oriented rectangular prism."""
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners_local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
    )
    corners_world = (R @ corners_local.T).T + center
    faces_idx = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # -y
        [2, 3, 7, 6],  # +y
        [1, 2, 6, 5],  # +x
        [0, 3, 7, 4],  # -x
    ]
    return [corners_world[idx] for idx in faces_idx]


def _disk(
    center_local: np.ndarray, R: np.ndarray, world_origin: np.ndarray, radius: float, n: int = 24
) -> np.ndarray:
    """Disk in body-XY plane around `center_local`, transformed to world."""
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts_local = np.stack(
        [
            center_local[0] + radius * np.cos(theta),
            center_local[1] + radius * np.sin(theta),
            np.full_like(theta, center_local[2]),
        ],
        axis=1,
    )
    return (R @ pts_local.T).T + world_origin


def draw_quadcopter(ax, position: np.ndarray, R: np.ndarray, scale: float, blade_phase: float):
    """Draw a stylized quadcopter at `position` with rotation `R`.

    Returns the list of artists added so caller can clear them next frame.
    """
    artists = []

    # --- body (compact rectangular prism, smaller than rotors) -------------
    body_size = (0.85 * scale, 0.85 * scale, 0.18 * scale)
    body_faces = _box_faces(position, R, body_size)
    body = Poly3DCollection(
        body_faces,
        facecolor=BODY_COLOR,
        edgecolor=ARM_COLOR,
        linewidths=0.4,
        alpha=1.0,
    )
    body.set_zorder(10)
    ax.add_collection3d(body)
    artists.append(body)

    # --- four arms + rotor hubs + rotor rims + spinning blades --------------
    arm_offsets_local = np.array(
        [
            [scale, scale, 0.16 * scale],
            [scale, -scale, 0.16 * scale],
            [-scale, -scale, 0.16 * scale],
            [-scale, scale, 0.16 * scale],
        ]
    )
    rotor_radius = 0.62 * scale

    for k, off_local in enumerate(arm_offsets_local):
        # arm: thin oriented box from body center to rotor hub
        arm_center_local = off_local / 2.0
        arm_dir_local = off_local / np.linalg.norm(off_local[:2])
        arm_len = float(np.linalg.norm(off_local[:2]))
        # build a rotation that aligns the arm's local x-axis with arm_dir_local
        ax_local = arm_dir_local.copy()
        ax_local[2] = 0
        ax_local /= np.linalg.norm(ax_local)
        ay_local = np.array([-ax_local[1], ax_local[0], 0.0])
        az_local = np.array([0.0, 0.0, 1.0])
        R_arm_local = np.stack([ax_local, ay_local, az_local], axis=1)
        arm_size = (arm_len, 0.13 * scale, 0.09 * scale)
        # arm corners in arm-local frame
        corners_arm = np.array(
            [
                [-arm_size[0] / 2, -arm_size[1] / 2, -arm_size[2] / 2],
                [arm_size[0] / 2, -arm_size[1] / 2, -arm_size[2] / 2],
                [arm_size[0] / 2, arm_size[1] / 2, -arm_size[2] / 2],
                [-arm_size[0] / 2, arm_size[1] / 2, -arm_size[2] / 2],
                [-arm_size[0] / 2, -arm_size[1] / 2, arm_size[2] / 2],
                [arm_size[0] / 2, -arm_size[1] / 2, arm_size[2] / 2],
                [arm_size[0] / 2, arm_size[1] / 2, arm_size[2] / 2],
                [-arm_size[0] / 2, arm_size[1] / 2, arm_size[2] / 2],
            ]
        )
        # transform arm corners: arm-local -> body-local -> world
        corners_body = (R_arm_local @ corners_arm.T).T + arm_center_local
        corners_world = (R @ corners_body.T).T + position
        faces_idx = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ]
        arm_faces = [corners_world[idx] for idx in faces_idx]
        arm_poly = Poly3DCollection(
            arm_faces,
            facecolor=ARM_COLOR,
            edgecolor=BODY_COLOR,
            linewidths=0.3,
        )
        arm_poly.set_zorder(9)
        ax.add_collection3d(arm_poly)
        artists.append(arm_poly)

        # rotor hub (small filled disk in body-XY plane at off_local)
        hub_pts = _disk(off_local, R, position, 0.16 * scale, n=20)
        hub = Poly3DCollection(
            [hub_pts],
            facecolor=ROTOR_HUB_COLOR,
            edgecolor=BODY_COLOR,
            linewidths=0.4,
        )
        hub.set_zorder(11)
        ax.add_collection3d(hub)
        artists.append(hub)

        # rotor rim (outer circle, thin)
        rim_pts = _disk(off_local, R, position, rotor_radius, n=40)
        rim_pts_closed = np.vstack([rim_pts, rim_pts[:1]])
        (rim_line,) = ax.plot(
            rim_pts_closed[:, 0],
            rim_pts_closed[:, 1],
            rim_pts_closed[:, 2],
            color=ROTOR_RIM_COLOR,
            linewidth=0.9,
            zorder=12,
        )
        artists.append(rim_line)

        # spinning blade pair: 2 thin lines at blade_phase + (k flips direction)
        spin_dir = 1.0 if k % 2 == 0 else -1.0
        phase = spin_dir * blade_phase + (k * np.pi / 4.0)
        for blade_offset in (0.0, np.pi / 2.0):
            ang = phase + blade_offset
            tip_a_local = off_local + np.array(
                [rotor_radius * np.cos(ang), rotor_radius * np.sin(ang), 0.0]
            )
            tip_b_local = off_local + np.array(
                [-rotor_radius * np.cos(ang), -rotor_radius * np.sin(ang), 0.0]
            )
            tips_local = np.stack([tip_a_local, tip_b_local], axis=0)
            tips_world = (R @ tips_local.T).T + position
            (blade_line,) = ax.plot(
                tips_world[:, 0],
                tips_world[:, 1],
                tips_world[:, 2],
                color=BLADE_COLOR,
                linewidth=1.4,
                alpha=0.9,
                zorder=13,
            )
            artists.append(blade_line)

        # blur disk: faint translucent fill suggesting rotor sweep
        blur_pts = _disk(off_local, R, position, rotor_radius * 0.92, n=28)
        blur = Poly3DCollection(
            [blur_pts],
            facecolor=BLADE_BLUR_COLOR,
            edgecolor="none",
            alpha=0.18,
        )
        blur.set_zorder(11)
        ax.add_collection3d(blur)
        artists.append(blur)

    return artists


def draw_shadow(
    ax, position: np.ndarray, R: np.ndarray, scale: float, z_floor: float, alpha: float = 0.28
):
    """Quadcopter-shaped shadow on the floor plane below the body.

    Projects the body footprint and the four rotor disks straight down (light
    from above) onto z=z_floor, preserving yaw/pitch/roll. Shadow softens
    with altitude.
    """
    altitude = max(position[2] - z_floor, 0.0)
    fade = max(0.0, 1.0 - altitude / 2.5)
    if fade <= 0.02:
        return []
    eff_alpha = alpha * fade
    artists = []

    def _project_to_floor(pts: np.ndarray) -> np.ndarray:
        flat = pts.copy()
        flat[:, 2] = z_floor + 1e-3
        return flat

    # body footprint: project the bottom face of the body box
    body_size = (0.85 * scale, 0.85 * scale, 0.18 * scale)
    hx, hy, hz = body_size[0] / 2, body_size[1] / 2, body_size[2] / 2
    body_corners_local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
        ]
    )
    body_world = (R @ body_corners_local.T).T + position
    body_floor = _project_to_floor(body_world)
    body_shadow = Poly3DCollection(
        [body_floor],
        facecolor=SHADOW_COLOR,
        edgecolor="none",
        alpha=eff_alpha,
    )
    body_shadow.set_zorder(2)
    ax.add_collection3d(body_shadow)
    artists.append(body_shadow)

    # rotor footprints: project each rotor disk
    rotor_radius = 0.62 * scale
    arm_offsets_local = np.array(
        [
            [scale, scale, 0.16 * scale],
            [scale, -scale, 0.16 * scale],
            [-scale, -scale, 0.16 * scale],
            [-scale, scale, 0.16 * scale],
        ]
    )
    for off_local in arm_offsets_local:
        rotor_world = _disk(off_local, R, position, rotor_radius, n=32)
        rotor_floor = _project_to_floor(rotor_world)
        rotor_shadow = Poly3DCollection(
            [rotor_floor],
            facecolor=SHADOW_COLOR,
            edgecolor="none",
            alpha=eff_alpha,
        )
        rotor_shadow.set_zorder(2)
        ax.add_collection3d(rotor_shadow)
        artists.append(rotor_shadow)

    return artists


def draw_trail(
    ax,
    positions: np.ndarray,
    color: str,
    n_segments: int = 40,
    max_alpha: float = 0.95,
    base_lw: float = 2.4,
    min_alpha: float = 0.0,
):
    """Recent positions as a polyline with alpha gradient (older = fainter).

    `min_alpha` floors the fade so the oldest segments stay visible; use it
    for static images where the whole trajectory should read clearly.
    """
    if len(positions) < 2:
        return []
    seg_count = min(n_segments, len(positions) - 1)
    artists = []
    for j in range(seg_count):
        idx_end = len(positions) - j
        idx_start = idx_end - 2
        seg = positions[idx_start:idx_end]
        a = min_alpha + (max_alpha - min_alpha) * (1.0 - j / seg_count) ** 1.4
        (line,) = ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color=color,
            linewidth=base_lw,
            alpha=a,
            solid_capstyle="round",
            zorder=5,
        )
        artists.append(line)
    return artists


def draw_floor(ax, x_lim, y_lim, z_floor):
    """Subtle floor plane + a sparse grid for visual depth cues."""
    # plane
    floor_pts = np.array(
        [
            [x_lim[0], y_lim[0], z_floor],
            [x_lim[1], y_lim[0], z_floor],
            [x_lim[1], y_lim[1], z_floor],
            [x_lim[0], y_lim[1], z_floor],
        ]
    )
    plane = Poly3DCollection(
        [floor_pts],
        facecolor=FLOOR_COLOR,
        edgecolor="none",
        alpha=0.55,
    )
    plane.set_zorder(0)
    ax.add_collection3d(plane)
    # gridlines
    n_grid = 8
    xs = np.linspace(x_lim[0], x_lim[1], n_grid + 1)
    ys = np.linspace(y_lim[0], y_lim[1], n_grid + 1)
    for x in xs:
        ax.plot(
            [x, x],
            [y_lim[0], y_lim[1]],
            [z_floor, z_floor],
            color=FLOOR_GRID_COLOR,
            linewidth=0.5,
            alpha=0.65,
            zorder=1,
        )
    for y in ys:
        ax.plot(
            [x_lim[0], x_lim[1]],
            [y, y],
            [z_floor, z_floor],
            color=FLOOR_GRID_COLOR,
            linewidth=0.5,
            alpha=0.65,
            zorder=1,
        )


def _setup_axes(ax, x_lim, y_lim, z_lim, title, label_color=TEXT_COLOR, title_y=0.80):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))
    if title:
        # title sits low in the axes bbox so it visually belongs to its
        # 3D plot rather than floating between adjacent panes
        ax.set_title(title, fontsize=15, color=label_color, fontweight="bold", y=title_y)


# ----------------------------------------------------------------------------
# Scene: framing, error read-outs
# ----------------------------------------------------------------------------
def scene_bounds(rollouts: list[Rollout], pad: float = 0.2) -> tuple:
    """Axis limits framing every reference and every flight, floor at the ground.

    ``x`` and ``y`` share one span, centred, so the floor grid is square; ``z``
    runs from the ground (or lower, if a flight went below it) to the highest
    point reached.  The flights take off from the origin, so framing on them
    rather than on the reference alone keeps the climb in view.
    """
    pts = np.concatenate([r.reference for r in rollouts] + [r.positions for r in rollouts])
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    half = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
    mid = 0.5 * (hi + lo)
    z_floor = min(GROUND, float(pts[:, 2].min()))
    return (
        (mid[0] - half, mid[0] + half),
        (mid[1] - half, mid[1] + half),
        (z_floor, float(hi[2])),
    )


def _default_scale(bounds) -> float:
    x_lim = bounds[0]
    return 0.05 * (x_lim[1] - x_lim[0])


def _tracking_error(rollout: Rollout, k: int) -> float:
    """Distance from the flown position to the reference at the same instant."""
    return float(np.linalg.norm(rollout.positions[k] - rollout.reference[k]))


def _ended(rollout: Rollout, frame: int) -> bool:
    return rollout.crashed and frame >= rollout.steps


def frame_hud(rollout: Rollout, frame: int, show_tilt: bool = False) -> str:
    """The read-out an animation frame carries."""
    if _ended(rollout, frame):
        return f"crashed at {rollout.steps * rollout.dt:.1f} s"
    k = min(frame, rollout.steps)
    text = f"error = {100 * _tracking_error(rollout, k):5.1f} cm"
    if show_tilt:
        text += f"   tilt = {rollout.tilt_deg[k]:4.1f}\N{DEGREE SIGN}"
    return text


def summary_hud(rollout: Rollout, show_tilt: bool = False) -> str:
    """The read-out a still of the whole flight carries: what the table reports."""
    if rollout.crashed:
        return f"crashed at {rollout.steps * rollout.dt:.1f} s"
    text = f"steady error {100 * rollout.steady_error():.1f} cm"
    if show_tilt:
        text += f", max tilt {rollout.max_tilt_deg:.0f}\N{DEGREE SIGN}"
    return text


def render_pane(
    ax,
    rollout: Rollout,
    frame: int,
    bounds,
    title: str,
    *,
    scale: float,
    trail_color: str,
    trail_segments: int,
    blade_phase: float,
    elev: float,
    azim: float,
    hud: str | None,
    trail_solid: bool = False,
    trail_thin: int = 1,
    title_y: float = 0.80,
):
    """Render one pane of the comparison at ``frame``.

    A rollout that crashed is held at its final state, marked, so the panes stay
    in step and the crash stays visible.  ``hud`` is the text in the rounded box
    at the bottom of the pane; the shared time stamp is drawn once per figure.
    ``trail_thin`` keeps one trail point per that many simulation steps, so
    the fading trail covers a fixed span of flight time whatever the step.
    """
    x_lim, y_lim, z_lim = bounds
    z_floor = z_lim[0]
    _setup_axes(ax, x_lim, y_lim, z_lim, title, title_y=title_y)
    ax.view_init(elev=elev, azim=azim)
    draw_floor(ax, x_lim, y_lim, z_floor)

    ref = rollout.reference
    ax.plot(
        ref[:, 0],
        ref[:, 1],
        ref[:, 2],
        color=REF_COLOR,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.55,
        zorder=3,
    )

    last = min(frame, rollout.steps)
    pos = rollout.positions[last]
    R = euler_to_R(rollout.states[last, ATT])
    trail = rollout.positions[: last + 1]

    if trail_solid:
        # single opaque polyline: the overlapping alpha-graded segments of
        # draw_trail bead up at the joints, which shows in static images
        ax.plot(
            trail[:, 0],
            trail[:, 1],
            trail[:, 2],
            color=trail_color,
            linewidth=2.4,
            alpha=0.95,
            solid_capstyle="round",
            zorder=5,
        )
    else:
        # Thin from the newest point backwards so the trail always ends at
        # the vehicle.
        thinned = trail[::-1][::trail_thin][::-1]
        draw_trail(ax, thinned, trail_color, trail_segments)
    draw_shadow(ax, pos, R, scale, z_floor)
    draw_quadcopter(ax, pos, R, scale, blade_phase)

    ended = _ended(rollout, frame)
    if ended:
        ax.scatter(*pos, color=CRASH_COLOR, marker="x", s=120, linewidths=2.5, zorder=20)

    if hud:
        ax.text2D(
            0.5,
            0.10,
            hud,
            transform=ax.transAxes,
            fontsize=13,
            color=CRASH_COLOR if ended else TEXT_COLOR,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.36", fc=ERR_BAR_BG, ec="none", alpha=0.9),
        )


def _layout_grid(n_panes: int) -> tuple[int, int, tuple[float, float]]:
    """Panes side by side: ``(nrows, ncols, figsize)``."""
    return 1, n_panes, (5.6 * n_panes + 0.4, 5.4)


# ----------------------------------------------------------------------------
# Cropping
# ----------------------------------------------------------------------------
def _content_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the non-white pixels, ``(x0, y0, x1, y1)``."""
    arr = np.asarray(image.convert("RGB"))
    mask = (arr < 248).any(axis=2)
    if not mask.any():
        return (0, 0, arr.shape[1], arr.shape[0])
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def _autocrop_pngs(frame_paths, pad: int = 6):
    """Crop the constant-size white margin off every frame, in place.

    Uses the union of each frame's non-white bbox so all frames stay the same
    pixel size (required for clean GIF assembly).  Returns the crop box.
    ``pad`` leaves a tiny white border so antialiased edges aren't clipped.
    """
    union = None
    for p in frame_paths:
        with Image.open(p) as img:
            bbox = _content_box(img)
        union = (
            list(bbox)
            if union is None
            else [
                min(union[0], bbox[0]),
                min(union[1], bbox[1]),
                max(union[2], bbox[2]),
                max(union[3], bbox[3]),
            ]
        )
    with Image.open(frame_paths[0]) as img0:
        W, H = img0.size
    crop_box = (
        max(union[0] - pad, 0),
        max(union[1] - pad, 0),
        min(union[2] + pad, W),
        min(union[3] + pad, H),
    )
    for p in frame_paths:
        with Image.open(p) as img:
            cropped = img.crop(crop_box)
        cropped.save(p)
    return crop_box


# ----------------------------------------------------------------------------
# The two products
# ----------------------------------------------------------------------------
def make_compare_gif(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str],
    colors: list[str],
    *,
    fps: int = 20,
    frame_stride: int = 5,
    dpi: int = 100,
    scale: float | None = None,
    camera_elev: float = CAMERA_ELEV,
    camera_azim_start: float = CAMERA_AZIM,
    orbit_deg_per_sec: float = 6.0,
    blade_rpm: float = 2400.0,
    trail_segments: int = 40,
    keep_frames: bool = False,
    show_tilt: bool = False,
) -> Path:
    """Animate the comparison, one pane per rollout, panes side by side.

    Every pane shares the reference, the axis limits, and the camera, which
    orbits at ``orbit_deg_per_sec`` of *flight* time so the sweep does not
    depend on how many frames are rendered.  At ``fps`` frames per second a
    stride of ``1 / (fps dt)`` steps per frame plays in real time.  The trail
    keeps one point per frame and fades over ``trail_segments`` of them, so it
    covers ``trail_segments / fps`` seconds of flight.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rollouts)
    assert len(labels) == n and len(colors) == n

    bounds = scene_bounds(rollouts)
    scale = _default_scale(bounds) if scale is None else scale
    dt = rollouts[0].dt
    total = max(r.steps for r in rollouts)
    frames = list(range(0, total + 1, frame_stride))
    if frames[-1] != total:
        frames.append(total)
    blade_per_frame = 2 * np.pi * blade_rpm / 60.0 / fps
    nrows, ncols, figsize = _layout_grid(n)

    workdir = Path(tempfile.mkdtemp(prefix="quadcopter_frames_"))
    written = []
    try:
        with plt.rc_context(_RC):
            for k, f in enumerate(frames):
                t = f * dt
                azim = camera_azim_start + orbit_deg_per_sec * t
                fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
                for j, roll in enumerate(rollouts):
                    ax = fig.add_subplot(
                        nrows, ncols, j + 1, projection="3d", computed_zorder=False
                    )
                    render_pane(
                        ax,
                        roll,
                        f,
                        bounds,
                        labels[j],
                        scale=scale,
                        trail_color=colors[j],
                        trail_segments=trail_segments,
                        blade_phase=k * blade_per_frame,
                        elev=camera_elev,
                        azim=azim,
                        hud=frame_hud(roll, f, show_tilt),
                        trail_thin=frame_stride,
                    )
                fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.0)
                fig.text(
                    0.99,
                    0.93,
                    f"time = {t:5.2f} s",
                    ha="right",
                    va="top",
                    fontsize=13,
                    color=TEXT_COLOR,
                    bbox=dict(boxstyle="round,pad=0.36", fc=ERR_BAR_BG, ec="none", alpha=0.85),
                )
                frame_path = workdir / f"frame_{k:05d}.png"
                fig.savefig(frame_path, facecolor="white")
                plt.close(fig)
                written.append(frame_path)

        _autocrop_pngs(written)
        with imageio.get_writer(path, mode="I", duration=1000.0 / fps, loop=0) as writer:
            for p in written:
                writer.append_data(imageio.imread(p))
    finally:
        if keep_frames:
            shutil.move(str(workdir), str(path.with_suffix("")) + "_frames")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return path


def make_compare_still(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str],
    colors: list[str],
    *,
    frame: int | None = None,
    dpi: int = 300,
    scale: float | None = None,
    camera_elev: float = CAMERA_ELEV,
    camera_azim: float = CAMERA_AZIM,
    show_tilt: bool = False,
    show_hud: bool = False,
    suptitle: str | None = None,
    title_y: float = 0.92,
    latex: bool = True,
) -> Path:
    """A single image, one pane per rollout, the whole flight drawn as a line.

    The vehicle sits at ``frame`` (default: the end of the flight, so a crash
    shows where it happened).  The per-pane caption is off by default; with
    ``show_hud`` it is the statistic the tables report.  ``suptitle`` is a
    figure-level header (see :func:`budget_title`), and the text is set in
    LaTeX unless ``latex=False``.  Written as a PDF when ``path`` ends in
    ``.pdf``, cropped to the drawn content the same way the animation frames
    are; a PNG otherwise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rollouts)
    assert len(labels) == n and len(colors) == n

    bounds = scene_bounds(rollouts)
    scale = _default_scale(bounds) if scale is None else scale
    nrows, ncols, figsize = _layout_grid(n)

    with plt.rc_context(_RC_LATEX if latex else _RC):
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
        if suptitle:
            fig.suptitle(suptitle, fontsize=19, y=0.905, color=TEXT_COLOR)
        for j, roll in enumerate(rollouts):
            f = roll.steps if frame is None else min(frame, roll.steps)
            ax = fig.add_subplot(nrows, ncols, j + 1, projection="3d", computed_zorder=False)
            render_pane(
                ax,
                roll,
                f,
                bounds,
                labels[j],
                scale=scale,
                trail_color=colors[j],
                trail_segments=roll.steps,
                blade_phase=0.0,
                elev=camera_elev,
                azim=camera_azim,
                hud=(
                    (
                        summary_hud(roll, show_tilt)
                        if frame is None
                        else frame_hud(roll, f, show_tilt)
                    )
                    if show_hud
                    else None
                ),
                trail_solid=True,
                title_y=title_y,
            )
        # Negative wspace pulls the panes together: the scene fills only the
        # centre of each subplot box, so without it a still has wide gutters.
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=-0.3)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor="white")
        buffer.seek(0)
        with Image.open(buffer) as img:
            x0, y0, x1, y1 = _content_box(img)
            W, H = img.size
        pad = 6
        x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)

        if path.suffix.lower() == ".png":
            buffer.seek(0)
            with Image.open(buffer) as img:
                img.crop((x0, y0, x1, y1)).save(path)
        else:
            # The PNG content box (pixels, origin top-left) as the figure bbox
            # (inches, origin bottom-left), so the vector output is as tight.
            height_in = fig.get_figheight()
            bbox = Bbox([[x0 / dpi, height_in - y1 / dpi], [x1 / dpi, height_in - y0 / dpi]])
            fig.savefig(path, facecolor="white", bbox_inches=bbox)
        plt.close(fig)
    return path
