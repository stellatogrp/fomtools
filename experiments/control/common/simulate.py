"""Closed-loop simulation, shared by both formulations.

At every sampling instant the controller measures the state, solves its
quadratic program for a fixed number of ADMM iterations from a cold start, and
returns the rotor thrusts to apply.  The plant is the same linear model the
controller predicts with, so there is no model mismatch and the only source of
tracking error is the finite iteration budget.  That is the point of the
experiment: whatever the trajectory does, the algorithm did it.

Everything specific to a formulation sits behind :class:`Controller`, so the
condensed and uncondensed studies fly the same simulator and are compared on
trajectories produced by identical code.

A rollout ends early if the quadcopter reaches the ground or leaves the flight
volume.  Both are recorded as a crash, since neither the linear model nor the
tracking metric means anything past that point.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from common.quadcopter import N_INPUT, N_STATE, POS, VEL
from common.reference import Reference

#: A rollout is abandoned once the altitude drops below this height, in metres.
GROUND = 0.0

#: ... or once the quadcopter is this far from the origin, in metres.
FLIGHT_VOLUME = 20.0

#: Seconds discarded from the front of a flight before the steady-state
#: tracking error is measured.  The quadcopter starts on the reference but
#: level and with the solver cold, so the first moments are a transient;
#: started from the origin instead, the flight opens with a climb.
SETTLE = 2.0


@dataclass
class Command:
    """What a controller returns for one sampling instant."""

    u: np.ndarray  # the thrusts to apply, already saturated
    residual: float  # the solver's fixed-point residual at the last iteration
    saturated: bool  # whether the solver's own iterate violated the thrust box


class Controller(Protocol):
    """A model predictive controller, however its program is formulated."""

    #: Sampling period in seconds.
    dt: float
    #: Number of optimization variables, reported in the summaries.
    n: int
    #: Plant matrices at this sampling period.
    A: np.ndarray
    B: np.ndarray

    def describe(self) -> str:
        """One line naming the formulation and its dimensions."""
        ...

    def command(self, state: np.ndarray, t: float, iterations: int) -> Command:
        """Solve for the thrusts to apply at time ``t`` from ``state``."""
        ...


@dataclass
class Rollout:
    """One closed-loop flight."""

    iterations: int
    dt: float
    states: np.ndarray  # (steps + 1, N_STATE), including the initial state
    inputs: np.ndarray  # (steps, N_INPUT), the thrusts actually applied
    reference: np.ndarray  # (steps + 1, 3), the position reference at each instant
    errors: np.ndarray  # (steps,), position tracking error after each step
    residuals: np.ndarray  # (steps,), the solver's residual at each instant
    saturated: np.ndarray  # (steps,), whether the applied input hit the box
    crashed: bool
    crash_reason: str | None = None

    @property
    def positions(self) -> np.ndarray:
        """The flown position trajectory, shape ``(steps + 1, 3)``."""
        return self.states[:, POS]

    @property
    def steps(self) -> int:
        return len(self.errors)

    @property
    def mean_error(self) -> float:
        """Mean error over the whole flight, opening transient included."""
        return float(np.mean(self.errors)) if self.steps else float("nan")

    @property
    def max_error(self) -> float:
        return float(np.max(self.errors)) if self.steps else float("nan")

    def steady_error(self, settle: float = SETTLE) -> float:
        """Mean error after the opening transient.

        The first seconds of a flight are the controller settling onto the curve
        (or, from the origin, a climb whose error swamps everything after it).
        This is the number that says how well the quadcopter holds the curve.
        """
        first = int(round(settle / self.dt))
        if self.steps <= first:
            return float("nan")
        return float(np.mean(self.errors[first:]))

    @property
    def tilt_deg(self) -> np.ndarray:
        """The larger of the two tilt angles at every instant, in degrees."""
        return np.degrees(np.max(np.abs(self.states[:, 0:2]), axis=1))

    @property
    def max_tilt_deg(self) -> float:
        return float(np.max(self.tilt_deg))

    def tilt_exceedance(self, limit_deg: float, slack_deg: float = 0.5) -> float:
        """Fraction of instants at which the tilt exceeds ``limit_deg`` by more than ``slack_deg``."""
        return float(np.mean(self.tilt_deg > limit_deg + slack_deg))

    @property
    def saturation_rate(self) -> float:
        """Fraction of instants at which the applied thrust was on the box."""
        return float(np.mean(self.saturated)) if self.steps else float("nan")

    def summary(self) -> str:
        if self.crashed:
            return (
                f"K={self.iterations:<6d} CRASHED after {self.steps} steps "
                f"({self.steps * self.dt:.2f} s): {self.crash_reason}"
            )
        return (
            f"K={self.iterations:<6d} steady {self.steady_error():7.3f} m  "
            f"whole {self.mean_error:7.3f} m  max {self.max_error:7.3f} m  "
            f"saturated {100 * self.saturation_rate:5.1f}%  "
            f"max tilt {self.max_tilt_deg:4.1f} deg"
        )


def initial_state(ref: Reference, on_reference: bool = True) -> np.ndarray:
    """The state the flight starts from.

    By default the quadcopter starts on the reference, at its position and
    velocity at time zero and level, as the learning-to-warm-start flights do.
    With ``on_reference=False`` it starts at the origin, hovering, and the
    flight opens with a capture climb: a violent manoeuvre that says more about
    the takeoff than about how well the controller holds the curve.
    """
    x = np.zeros(N_STATE)
    if on_reference:
        pos, vel = ref(0.0)
        x[POS] = pos
        x[VEL] = vel
    return x


def closed_loop(
    controller: Controller,
    ref: Reference,
    iterations: int,
    duration: float = 8.0,
    on_reference: bool = True,
) -> Rollout:
    """Fly the quadcopter for ``duration`` seconds at ``iterations`` per solve."""
    dt = controller.dt
    steps = int(round(duration / dt))
    x = initial_state(ref, on_reference)

    states = np.empty((steps + 1, N_STATE))
    inputs = np.empty((steps, N_INPUT))
    ref_pos = np.empty((steps + 1, 3))
    errors = np.empty(steps)
    residuals = np.empty(steps)
    saturated = np.empty(steps, dtype=bool)

    states[0] = x
    ref_pos[0] = ref(0.0)[0]

    crashed = False
    reason = None
    taken = 0

    for k in range(steps):
        t = k * dt
        command = controller.command(x, t, iterations)
        x = controller.A @ x + controller.B @ command.u

        inputs[k] = command.u
        residuals[k] = command.residual
        saturated[k] = command.saturated
        states[k + 1] = x
        ref_pos[k + 1] = ref(t + dt)[0]
        errors[k] = float(np.linalg.norm(x[POS] - ref_pos[k + 1]))
        taken = k + 1

        reason = _crash_reason(x)
        if reason is not None:
            crashed = True
            break

    return Rollout(
        iterations=iterations,
        dt=dt,
        states=states[: taken + 1],
        inputs=inputs[:taken],
        reference=ref_pos[: taken + 1],
        errors=errors[:taken],
        residuals=residuals[:taken],
        saturated=saturated[:taken],
        crashed=crashed,
        crash_reason=reason if crashed else None,
    )


def time_per_iteration(
    controller: Controller,
    ref: Reference,
    iterations: int = 500,
    repeats: int = 5,
) -> float:
    """Median wall-clock cost of one call to ``command``, divided by its iterations.

    Includes the work every sampling instant does once — stacking the reference
    window, forming the cost vector — amortized over the iteration count.  That
    is the right number for a study that varies only the iteration count, since
    the once-per-instant cost is the same in every row.  Comparing *different
    implementations* needs :func:`us_per_iteration` instead, which excludes it.
    """
    state = initial_state(ref)
    controller.command(state, 0.0, 200)  # warm the caches and the BLAS threads
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        controller.command(state, 0.0, iterations)
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return float(statistics.median(samples))


@dataclass
class IterationCost:
    """What one iteration costs, with the dispersion of the measurement.

    Converts to ``float`` as the median, so it can be used in arithmetic
    directly, while keeping the individual slopes for reporting.  Every number
    this experiment quotes about speed comes from one of these, and the spread is
    part of the number: a two-point slope on a shared machine is not stable to
    better than a few percent, and a run that catches interference can be off by
    a third.
    """

    samples: list[float]

    @property
    def us(self) -> float:
        return float(statistics.median(self.samples))

    @property
    def spread_pct(self) -> float:
        lo, hi = min(self.samples), max(self.samples)
        return 100.0 * (hi - lo) / lo if lo > 0 else float("nan")

    def __float__(self) -> float:
        return self.us

    def __format__(self, spec: str) -> str:
        return format(self.us, spec)


def slope_us(measure, low: int, high: int, trials: int = 5) -> IterationCost:
    """Per-iteration cost as a two-point slope, repeated and taken as a median.

    ``measure(k)`` returns the wall-clock microseconds of one solve at ``k``
    iterations.  Differencing two counts removes everything that does not scale
    with the count — parameter marshalling, a cold start, the once-per-instant
    work a controller does — and leaves the loop by itself.

    Both endpoints are exercised once and discarded before anything is recorded.
    That warm-up is not optional: the first measurement in a fresh process pays
    import, page-fault and thread-pool costs that no later one does, and left in
    it lands about thirty percent high.
    """
    if high <= low:
        raise ValueError("high must exceed low")
    measure(low)
    measure(high)  # warm both endpoints; discard
    slopes = []
    for _ in range(trials):
        at_low = measure(low)
        at_high = measure(high)
        slopes.append((at_high - at_low) / (high - low))
    return IterationCost(samples=slopes)


def us_per_iteration(
    controller: Controller,
    ref: Reference,
    low: int = 20,
    high: int = 200,
    repeats: int = 7,
    trials: int = 5,
) -> IterationCost:
    """Microseconds per ADMM iteration for a Python controller.

    A controller does more per sampling instant than iterate: it stacks the
    reference window and forms the cost vector, and for the condensed program
    that costs about as much as fifteen iterations do.  Charging the loop for it
    would misreport what a compute budget buys, and would do so unequally, since
    the generated solvers are timed on their loop alone.

    This is the counterpart of :meth:`common.csolver.CSolver.us_per_iteration`,
    deliberately the same measurement so the implementations are comparable.
    """
    state = initial_state(ref)

    def measure(iterations: int) -> float:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            controller.command(state, 0.0, iterations)
            samples.append((time.perf_counter() - start) * 1e6)
        return statistics.median(samples)

    return slope_us(measure, low, high, trials)


def _crash_reason(x: np.ndarray) -> str | None:
    """Why this state ends the flight, or ``None`` if it does not."""
    position = x[POS]
    if not np.all(np.isfinite(position)):
        return "the state diverged"
    if position[2] < GROUND:
        return f"hit the ground (altitude {position[2]:.2f} m)"
    if np.linalg.norm(position) > FLIGHT_VOLUME:
        return f"left the flight volume ({np.linalg.norm(position):.1f} m from the origin)"
    return None
