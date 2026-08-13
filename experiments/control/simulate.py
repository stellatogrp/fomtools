"""Closed-loop simulation of the quadcopter under the ADMM controller.

At every sampling instant the controller measures the state, forms the linear
cost of the condensed program, runs a fixed number of ADMM iterations from a
cold start, and applies the first input move.  The plant is the same linear
model the controller predicts with, so there is no model mismatch and the only
source of tracking error is the finite iteration budget.  That is the point of
the experiment: whatever the trajectory does, the algorithm did it.

A rollout ends early if the quadcopter reaches the ground or leaves the flight
volume.  Both are recorded as a crash, since neither the linear model nor the
tracking metric means anything past that point.
"""

from __future__ import annotations

from dataclasses import dataclass

import admm
import numpy as np
from mpc import MpcProblem
from quadcopter import N_INPUT, N_STATE, POS, VEL
from reference import Reference

#: A rollout is abandoned once the altitude drops below this height, in metres.
GROUND = 0.0

#: ... or once the quadcopter is this far from the origin, in metres.
FLIGHT_VOLUME = 20.0

#: Seconds discarded from the front of a flight before the steady-state
#: tracking error is measured.  The quadcopter starts at the origin and the
#: reference starts a metre above it, so the flight opens with a climb.
SETTLE = 2.0


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
        """Mean error over the whole flight, capture transient included."""
        return float(np.mean(self.errors)) if self.steps else float("nan")

    @property
    def max_error(self) -> float:
        return float(np.max(self.errors)) if self.steps else float("nan")

    def steady_error(self, settle: float = SETTLE) -> float:
        """Mean error after the capture transient.

        The flight starts at the origin while the reference starts a metre up,
        so the first second is a climb whose error swamps everything after it.
        This is the number that says how well the quadcopter holds the curve.
        """
        first = int(round(settle / self.dt))
        if self.steps <= first:
            return float("nan")
        return float(np.mean(self.errors[first:]))

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
            f"saturated {100 * self.saturation_rate:5.1f}%"
        )


def initial_state(ref: Reference, on_reference: bool = False) -> np.ndarray:
    """The state the flight starts from.

    By default the quadcopter starts at the origin, hovering.  The reference
    begins elsewhere, so the flight opens with a capture transient, which is the
    part of the trajectory that a poorly converged solve handles worst.
    """
    x = np.zeros(N_STATE)
    if on_reference:
        pos, vel = ref(0.0)
        x[POS] = pos
        x[VEL] = vel
    return x


def closed_loop(
    problem: MpcProblem,
    ref: Reference,
    iterations: int,
    duration: float = 6.0,
    on_reference: bool = False,
    warm_start: bool = False,
) -> Rollout:
    """Fly the quadcopter for ``duration`` seconds at ``iterations`` per solve.

    Parameters
    ----------
    warm_start:
        Carry the ADMM iterate across sampling instants.  Off by default, which
        matches the generated solver: its interface resets the state on every
        parameter update, so a fair comparison cold starts everywhere.
    """
    steps = int(round(duration / problem.dt))
    x = initial_state(ref, on_reference)

    states = np.empty((steps + 1, N_STATE))
    inputs = np.empty((steps, N_INPUT))
    ref_pos = np.empty((steps + 1, 3))
    errors = np.empty(steps)
    residuals = np.empty(steps)
    saturated = np.empty(steps, dtype=bool)

    states[0] = x
    ref_pos[0] = ref(0.0)[0]

    z = v = None
    crashed = False
    reason = None
    taken = 0

    for k in range(steps):
        t = k * problem.dt
        q = problem.linear_cost(x, problem.reference_window(ref, t))
        result = admm.solve(problem, q, iterations, z0=z, v0=v)
        if warm_start:
            z, v = result.z, result.v

        u = problem.first_input(result.z)
        saturated[k] = bool(np.any(np.abs(result.z[:N_INPUT] - u) > 1e-9))
        x = problem.A @ x + problem.B @ u

        inputs[k] = u
        residuals[k] = result.residual if result.residual is not None else np.nan
        states[k + 1] = x
        ref_pos[k + 1] = ref(t + problem.dt)[0]
        errors[k] = float(np.linalg.norm(x[POS] - ref_pos[k + 1]))
        taken = k + 1

        reason = _crash_reason(x)
        if reason is not None:
            crashed = True
            break

    return Rollout(
        iterations=iterations,
        dt=problem.dt,
        states=states[: taken + 1],
        inputs=inputs[:taken],
        reference=ref_pos[: taken + 1],
        errors=errors[:taken],
        residuals=residuals[:taken],
        saturated=saturated[:taken],
        crashed=crashed,
        crash_reason=reason if crashed else None,
    )


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
