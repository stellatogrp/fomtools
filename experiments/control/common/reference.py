"""Reference trajectories for the quadcopter to track.

A reference is a callable mapping a time in seconds to a position and a velocity
in R^3.  The controller needs both: the stage cost of ``quadcopter.Q_STAGE``
penalizes linear velocity, so tracking a moving target with a zero velocity
reference makes the cost fight the trajectory instead of following it.

Two generators are provided.  The figure eight is deterministic and is what the
reported figures use; the B-spline generator mirrors the procedure of the
learning-to-warm-start literature and exists for averaging over many rollouts.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

#: A reference maps time to ``(position, velocity)``, each of length 3.
Reference = Callable[[float], tuple[np.ndarray, np.ndarray]]


class ReferenceSpec(Protocol):
    """What every generator in this module returns."""

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]: ...


def figure_eight(
    period: float = 2.25,
    amplitude: float = 1.0,
    climb: float = 0.4,
    altitude: float = 1.0,
) -> Reference:
    """A Lissajous figure eight in the horizontal plane with a slow climb.

    The curve traces ``x = a sin(w t)``, ``y = (a/2) sin(2 w t)`` at angular
    frequency ``w = 2 pi / period``, and oscillates in altitude at half that
    frequency so that successive laps are at different heights.

    A short period and a large amplitude make the trajectory demanding.  The
    defaults sit at the edge of what the vehicle can fly: the rotor thrusts are
    on their bounds for roughly two thirds of the flight, so the box constraint
    of the underlying quadratic program is active and the accuracy of the solve
    decides whether the quadcopter holds the curve.  A gentler reference leaves
    the constraint slack, and then the iteration count stops mattering at all.
    """
    w = 2.0 * np.pi / period

    def ref(t: float) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(
            [
                amplitude * np.sin(w * t),
                0.5 * amplitude * np.sin(2.0 * w * t),
                altitude + climb * np.sin(0.5 * w * t),
            ]
        )
        vel = np.array(
            [
                amplitude * w * np.cos(w * t),
                amplitude * w * np.cos(2.0 * w * t),
                0.5 * climb * w * np.cos(0.5 * w * t),
            ]
        )
        return pos, vel

    return ref


def quasi_periodic(
    period: float = 2.25,
    amplitude: float = 1.0,
    climb: float = 0.35,
    altitude: float = 1.2,
    ratio: float = 1.6180339887498949,
) -> Reference:
    """A Lissajous curve whose axes are incommensurate, so it never repeats.

    The figure eight closes after one period, which makes every lap identical and
    lets the tracking error settle into a pattern tied to that period.  Giving
    the lateral axis an irrational multiple of the longitudinal frequency -- the
    golden ratio, the hardest to approximate by a fraction -- breaks that: the
    horizontal path never retraces itself and slowly fills a box, while the
    altitude drifts on a third, slower incommensurate cycle.

    The result is visibly less regular while staying analytic, so the velocity is
    the exact derivative of the position rather than a finite difference of it.
    That matters here: the stage cost penalizes linear velocity, so a reference
    whose velocity disagrees with its position makes the cost fight the path.

    The defaults keep the speed within a few percent of the figure eight's, so
    the two are comparably demanding, and hold the altitude between 0.85 and
    1.55 m, well clear of the ground.
    """
    w = 2.0 * np.pi / period
    lateral = 0.6 * amplitude

    def ref(t: float) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(
            [
                amplitude * np.sin(w * t),
                lateral * np.sin(ratio * w * t),
                altitude + climb * np.sin(w * t / ratio),
            ]
        )
        vel = np.array(
            [
                amplitude * w * np.cos(w * t),
                lateral * ratio * w * np.cos(ratio * w * t),
                climb * (w / ratio) * np.cos(w * t / ratio),
            ]
        )
        return pos, vel

    return ref


def bspline_waypoints(
    duration: float,
    num_waypoints: int = 6,
    bound: float = 1.5,
    altitude: float = 1.0,
    seed: int = 0,
) -> Reference:
    """A smooth closed path through random waypoints, as in the l2ws experiments.

    Waypoints are drawn uniformly from a cube of half-width ``bound`` centered on
    the origin at height ``altitude``, starting from the origin, and interpolated
    by a cubic B-spline.  The velocity is the analytic derivative of the spline,
    so it is consistent with the position rather than a finite difference of it.
    """
    from scipy.interpolate import splev, splprep

    rng = np.random.default_rng(seed)
    pts = rng.uniform(
        low=[-bound, -bound, altitude - 0.5 * bound],
        high=[bound, bound, altitude + 0.5 * bound],
        size=(num_waypoints - 1, 3),
    )
    pts = np.vstack([[0.0, 0.0, altitude], pts])
    tck, _ = splprep(pts.T, s=0)

    def ref(t: float) -> tuple[np.ndarray, np.ndarray]:
        # The spline is parameterized on [0, 1]; wrap so the path repeats.
        s = (t / duration) % 1.0
        pos = np.asarray(splev(s, tck)).reshape(3)
        # d(pos)/dt = d(pos)/ds * ds/dt, and ds/dt is the constant 1 / duration.
        vel = np.asarray(splev(s, tck, der=1)).reshape(3) / duration
        return pos, vel

    return ref


def sample(ref: Reference, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a reference on a grid, returning arrays of shape ``(len(times), 3)``."""
    pos = np.empty((len(times), 3))
    vel = np.empty((len(times), 3))
    for i, t in enumerate(times):
        pos[i], vel[i] = ref(float(t))
    return pos, vel
