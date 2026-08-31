"""Figures and animations for the closed-loop flights.

Three products:

* a still image comparing the flights side by side, one pane per rollout;
* an animated version of the same comparison;
* a two-dimensional plot of tracking error against time.

The first two are drawn by :mod:`common.quadcopter_render`, a port of the
renderer in the learning-to-warm-start code: a quadcopter mesh oriented by the
attitude states, a fading trail, a shadow on a gridded floor, the reference as
a dashed curve, and a slowly orbiting camera shared by every pane.  This module
keeps the entry points the studies call, and the colour and label conventions
that make one run one colour across every figure it appears in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import quadcopter_render as render  # noqa: E402
from common.quadcopter_render import budget_title  # noqa: E402, F401  (re-exported for callers)
from common.simulate import Rollout  # noqa: E402

#: Default trail colors, in the order rollouts are given.
TRAIL_COLORS = ["#c0392b", "#e08214", "#1f77b4", "#2ca02c"]


def _titles(rollouts, labels):
    if labels is not None:
        return list(labels)
    return [f"$K = {r.iterations}$" for r in rollouts]


def _colors(rollouts, colors):
    """One colour per rollout, defaulting to :data:`TRAIL_COLORS` by position.

    A study that draws its rollouts somewhere other than a flight panel should
    pass its own, so one run is one colour across every figure it appears in.
    """
    if colors is not None:
        return list(colors)
    return [TRAIL_COLORS[i % len(TRAIL_COLORS)] for i in range(len(rollouts))]


def real_time_stride(rollouts: list[Rollout], fps: int) -> int:
    """Simulation steps per frame at which ``fps`` plays the flight in real time."""
    return max(1, int(round(1.0 / (fps * rollouts[0].dt))))


def still_comparison(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str] | None = None,
    frame: int | None = None,
    dpi: int = 300,
    colors: list[str] | None = None,
    show_tilt: bool = False,
    show_hud: bool = False,
    suptitle: str | None = None,
) -> Path:
    """A single image with one pane per rollout, whole trajectories drawn.

    ``suptitle`` is the figure-level header; :func:`budget_title` builds the
    one naming a compute budget.
    """
    return render.make_compare_still(
        rollouts,
        path,
        _titles(rollouts, labels),
        _colors(rollouts, colors),
        frame=frame,
        dpi=dpi,
        show_tilt=show_tilt,
        show_hud=show_hud,
        suptitle=suptitle,
    )


def make_gif(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str] | None = None,
    fps: int = 20,
    frame_stride: int | None = None,
    dpi: int = 100,
    keep_frames: bool = False,
    colors: list[str] | None = None,
    show_tilt: bool = False,
) -> Path:
    """Animate the comparison, one pane per rollout.

    ``frame_stride`` defaults to real-time playback at ``fps``.  A rollout that
    crashed is held at its final state for the rest of the animation, marked,
    so the panes stay in step and the crash stays visible.
    """
    if frame_stride is None:
        frame_stride = real_time_stride(rollouts, fps)
    return render.make_compare_gif(
        rollouts,
        path,
        _titles(rollouts, labels),
        _colors(rollouts, colors),
        fps=fps,
        frame_stride=frame_stride,
        dpi=dpi,
        keep_frames=keep_frames,
        show_tilt=show_tilt,
    )


def error_vs_time(
    rollouts: list[Rollout],
    path: str | Path,
    labels: list[str] | None = None,
    dpi: int = 300,
    colors: list[str] | None = None,
) -> Path:
    """Tracking error against time, one curve per rollout, log scale."""
    path = Path(path)
    titles = _titles(rollouts, labels)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for roll, title, color in zip(rollouts, titles, _colors(rollouts, colors)):
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
