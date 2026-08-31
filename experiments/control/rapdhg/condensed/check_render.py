"""The drawn quadcopter points the way it accelerates, and the animation plays.

Two checks on :mod:`common.quadcopter_render`, neither of which needs a
generated solver.

The first is about the frame.  The linear model's attitude states are ordered
so that state 0 drives the ``x`` acceleration and state 1 drives ``y``, which
the dynamics matrix says (``A_DYN[3, 0] > 0``, ``A_DYN[4, 1] < 0``) and the
docstring's ``(phi, theta, psi)`` naming obscures.  A body whose thrust axis is
drawn from the wrong Euler convention tilts about the wrong axis, which is what
made the earlier figures look strange.  So this flies the NumPy controller and
compares the drawn thrust axis ``R e_z`` with the direction the vehicle
actually accelerates in, ``a + g e_z``; the two should agree to within the
small-angle error of the model, and the legacy convention is printed alongside
for contrast.

The second is mechanical: a short animation is written and read back, and its
frame count, frame delay, and frame sizes are what was asked for.

    python -m condensed.check_render
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from common import plots, reference
from common.quadcopter import ATT, VEL
from common.quadcopter_render import euler_to_R
from common.simulate import closed_loop
from condensed.controller import build

GRAVITY = 9.81
ITERATIONS = 26
MAX_MEAN_ANGLE_DEG = 15.0


def _legacy_rotation(attitude):
    """The convention the old ``plots._rotation`` used: state 0 through ``Rx``."""
    phi, theta, psi = attitude
    c, s = np.cos, np.sin
    rz = np.array([[c(psi), -s(psi), 0.0], [s(psi), c(psi), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[c(theta), 0.0, s(theta)], [0.0, 1.0, 0.0], [-s(theta), 0.0, c(theta)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c(phi), -s(phi)], [0.0, s(phi), c(phi)]])
    return rz @ ry @ rx


def _mean_angle(states, dt, rotation):
    acc = np.diff(states[:, VEL], axis=0) / dt
    want = acc + np.array([0.0, 0.0, GRAVITY])
    want /= np.linalg.norm(want, axis=1)[:, None]
    axis = np.array([rotation(states[k, ATT]) @ np.array([0.0, 0.0, 1.0]) for k in range(len(acc))])
    cosine = np.clip(np.sum(axis * want, axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean()), acc


def main() -> None:
    ref = reference.quasi_periodic()
    controller = build(ref, max_iterations=512)
    roll = closed_loop(controller, ref, ITERATIONS)
    failures = []

    angle, acc = _mean_angle(roll.states, roll.dt, euler_to_R)
    legacy, _ = _mean_angle(roll.states, roll.dt, _legacy_rotation)
    corr_x = np.corrcoef(roll.states[1:, 0], acc[:, 0])[0, 1]
    corr_y = np.corrcoef(roll.states[1:, 1], acc[:, 1])[0, 1]
    print(f"mean angle between drawn thrust axis and acceleration: {angle:6.2f} deg")
    print(f"  the same with the legacy (state 0 -> roll) convention: {legacy:6.2f} deg")
    print(f"  corr(state 0, a_x) = {corr_x:+.3f}   corr(state 1, a_y) = {corr_y:+.3f}")
    if angle > MAX_MEAN_ANGLE_DEG:
        failures.append(f"drawn thrust axis is {angle:.1f} deg off the acceleration")
    if not (corr_x > 0.9 and corr_y < -0.9):
        failures.append("attitude states do not drive x and y with the expected signs")

    with tempfile.TemporaryDirectory(prefix="check_render_") as tmp:
        tmp = Path(tmp)
        stride = 40
        gif = plots.make_gif(
            [roll, roll], tmp / "check.gif", ["a", "b"], fps=20, frame_stride=stride
        )
        still = plots.still_comparison([roll], tmp / "check.pdf", ["a"])
        with Image.open(gif) as im:
            frames = im.n_frames
            delay = im.info.get("duration")
            sizes = set()
            for k in range(frames):
                im.seek(k)
                sizes.add(im.size)
        expected = len(range(0, roll.steps + 1, stride)) + (1 if roll.steps % stride else 0)
        print(f"animation: {frames} frames, {delay} ms per frame, sizes {sorted(sizes)}")
        print(f"still:     {still.stat().st_size} bytes")
        if frames != expected:
            failures.append(f"expected {expected} frames, wrote {frames}")
        if delay != 50:
            failures.append(f"expected a 50 ms frame delay, wrote {delay}")
        if len(sizes) != 1:
            failures.append("frames differ in size")
        if still.stat().st_size == 0:
            failures.append("the still is empty")

    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))
    print("\nall checks passed")


if __name__ == "__main__":
    main()
