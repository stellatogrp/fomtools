"""Figures and animations for the closed-loop flights.

Three products, all built from the same panel:

* a still image comparing the flights side by side, one panel per rollout;
* an animated version of the same comparison;
* a two-dimensional plot of tracking error against time.

The panels share a camera, axis limits, and reference curve, so the only thing
that differs between them is the flight.  The animation is assembled the way the
learning-to-warm-start code does it, by rendering one image per frame and
stitching them, rather than through a live animation writer: the frames are
worth keeping, and a failure part way through leaves something to look at.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from simulate import Rollout  # noqa: E402

REFERENCE_COLOR = "black"
BODY_COLOR = "#2b2b2b"
ARM_COLOR = "#4a4a4a"
ROTOR_COLOR = "#8c8c8c"
SHADOW_COLOR = "#c8c8c8"
CRASH_COLOR = "#c0392b"

#: Default trail colors, in the order rollouts are given.
TRAIL_COLORS = ["#c0392b", "#e08214", "#1f77b4", "#2ca02c"]

CAMERA_ELEV = 22.0
CAMERA_AZIM = -65.0


def _rotation(attitude: np.ndarray) -> np.ndarray:
    """Body-to-world rotation from roll, pitch, and yaw."""
    phi, theta, psi = attitude
    c, s = np.cos, np.sin
    rz = np.array([[c(psi), -s(psi), 0.0], [s(psi), c(psi), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[c(theta), 0.0, s(theta)], [0.0, 1.0, 0.0], [-s(theta), 0.0, c(theta)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c(phi), -s(phi)], [0.0, s(phi), c(phi)]])
    return rz @ ry @ rx


def _draw_quadcopter(ax, position, attitude, scale: float) -> None:
    """A four-armed stick quadcopter at ``position``, oriented by ``attitude``."""
    rot = _rotation(attitude)
    arms = (
        scale
        * np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
        / np.sqrt(2.0)
    )

    for arm in arms:
        tip = position + rot @ arm
        seg = np.vstack([position, tip])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=ARM_COLOR, linewidth=1.6, zorder=5)
        ax.scatter(*tip, color=ROTOR_COLOR, s=18, depthshade=False, zorder=6)

    ax.scatter(*position, color=BODY_COLOR, s=42, depthshade=False, zorder=7)


def _draw_shadow(ax, positions: np.ndarray, floor: float) -> None:
    """The ground track, which gives the eye a depth cue the trail alone lacks."""
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        np.full(len(positions), floor),
        color=SHADOW_COLOR,
        linewidth=0.8,
        zorder=1,
    )


def scene_bounds(reference: np.ndarray, margin: float = 0.45) -> tuple:
    """A cube framing the reference curve, shared by every panel.

    Framing on the reference rather than on the flights keeps the well-tracked
    panels legible; a diverging flight leaves the box, which is the correct
    visual outcome and is annotated rather than accommodated.
    """
    center = 0.5 * (reference.max(axis=0) + reference.min(axis=0))
    half = 0.5 * float(np.max(reference.max(axis=0) - reference.min(axis=0))) + margin
    return center, half


def _setup_axes(ax, center, half, title: str | None) -> None:
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(0.0, center[2] - half), center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 0.75))
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
    ax.set_xlabel("$x$ (m)", labelpad=-8, fontsize=8)
    ax.set_ylabel("$y$ (m)", labelpad=-8, fontsize=8)
    ax.set_zlabel("$z$ (m)", labelpad=-8, fontsize=8)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10, pad=0)


def panel(
    ax,
    rollout: Rollout,
    frame: int,
    center,
    half,
    color: str,
    title: str | None = None,
    show_vehicle: bool = True,
    note: str | None = None,
) -> None:
    """Draw one flight into a three-dimensional axes, up to ``frame``.

    ``note`` is the caption under the panel; it defaults to the tracking error
    at ``frame``, which is what an animation wants.  A still of the whole flight
    should pass the summary statistic it is reporting instead.
    """
    floor = max(0.0, center[2] - half)
    reference = rollout.reference

    last = min(frame, rollout.steps)
    trail = rollout.positions[: last + 1]
    if len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, linewidth=1.8, zorder=3)
        _draw_shadow(ax, trail, floor)

    # Drawn after the trail so the target the eye compares against is never
    # buried under a flight that wandered across it.
    ax.plot(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        color=REFERENCE_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=4,
    )

    ended = rollout.crashed and frame >= rollout.steps
    if show_vehicle and len(trail):
        _draw_quadcopter(ax, trail[-1], rollout.states[last, 0:3], scale=0.24 * half)
    if ended:
        ax.scatter(*trail[-1], color=CRASH_COLOR, marker="x", s=90, linewidths=2.5, zorder=8)

    _setup_axes(ax, center, half, title)

    if note is None:
        if rollout.crashed and frame >= rollout.steps:
            note = f"crashed at {rollout.steps * rollout.dt:.1f} s"
        elif len(trail):
            err = float(np.linalg.norm(trail[-1] - reference[last]))
            note = f"error {100 * err:.0f} cm"
        else:
            note = ""
    ax.text2D(
        0.5,
        -0.02,
        note,
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color=CRASH_COLOR if rollout.crashed else "black",
    )


def _summary_note(rollout: Rollout) -> str:
    """The caption a still image carries: the statistic the table reports."""
    if rollout.crashed:
        return f"crashed at {rollout.steps * rollout.dt:.1f} s"
    return f"tracking error {100 * rollout.steady_error():.0f} cm"


def _titles(rollouts, labels):
    if labels is not None:
        return list(labels)
    return [f"$K = {r.iterations}$" for r in rollouts]


def still_comparison(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str] | None = None,
    frame: int | None = None,
    dpi: int = 300,
) -> Path:
    """A single image with one panel per rollout, whole trajectories drawn."""
    path = Path(path)
    center, half = scene_bounds(rollouts[0].reference)
    titles = _titles(rollouts, labels)

    fig, axes = plt.subplots(
        1, len(rollouts), figsize=(3.2 * len(rollouts), 3.4), subplot_kw={"projection": "3d"}
    )
    axes = np.atleast_1d(axes)
    for ax, roll, title, color in zip(axes, rollouts, titles, TRAIL_COLORS):
        end = roll.steps if frame is None else frame
        panel(
            ax,
            roll,
            end,
            center,
            half,
            color,
            title=title,
            show_vehicle=True,
            note=_summary_note(roll) if frame is None else None,
        )

    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.94, bottom=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def make_gif(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str] | None = None,
    fps: int = 20,
    frame_stride: int | None = None,
    dpi: int = 110,
    keep_frames: bool = False,
) -> Path:
    """Animate the comparison, one panel per rollout.

    A rollout that crashed is held at its final state for the rest of the
    animation, marked, so the panels stay in step and the crash stays visible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    center, half = scene_bounds(rollouts[0].reference)
    titles = _titles(rollouts, labels)

    total = max(r.steps for r in rollouts)
    if frame_stride is None:
        # Aim for a few hundred frames however long the flight is.
        frame_stride = max(1, total // 240)
    frames = range(0, total + 1, frame_stride)

    workdir = Path(tempfile.mkdtemp(prefix="quadcopter_frames_"))
    written = []
    try:
        for i, f in enumerate(frames):
            fig, axes = plt.subplots(
                1,
                len(rollouts),
                figsize=(3.2 * len(rollouts), 3.4),
                subplot_kw={"projection": "3d"},
            )
            axes = np.atleast_1d(axes)
            for ax, roll, title, color in zip(axes, rollouts, titles, TRAIL_COLORS):
                panel(ax, roll, f, center, half, color, title=title, show_vehicle=True)
            fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.94, bottom=0.06)
            frame_path = workdir / f"frame_{i:05d}.png"
            fig.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            written.append(frame_path)

        # Frames can differ by a pixel or two after a tight bounding box; pad
        # them all to a common size so the encoder does not reject the stream.
        images = [imageio.imread(p) for p in written]
        height = max(im.shape[0] for im in images)
        width = max(im.shape[1] for im in images)
        with imageio.get_writer(path, mode="I", duration=1.0 / fps, loop=0) as writer:
            for im in images:
                writer.append_data(_pad_to(im, height, width))
    finally:
        if keep_frames:
            shutil.move(str(workdir), str(path.with_suffix("")) + "_frames")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return path


def _pad_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pad an RGB(A) frame to ``height`` by ``width`` with white."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    out = np.full((height, width, image.shape[2]), 255, dtype=image.dtype)
    out[: image.shape[0], : image.shape[1]] = image
    return out


def error_vs_time(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str] | None = None,
    dpi: int = 300,
) -> Path:
    """Tracking error against time, one curve per rollout, log scale."""
    path = Path(path)
    titles = _titles(rollouts, labels)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for roll, title, color in zip(rollouts, titles, TRAIL_COLORS):
        t = np.arange(1, roll.steps + 1) * roll.dt
        ax.semilogy(t, roll.errors, color=color, linewidth=1.5, label=title)
        if roll.crashed:
            ax.scatter(t[-1], roll.errors[-1], color=color, marker="x", s=50, linewidths=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
