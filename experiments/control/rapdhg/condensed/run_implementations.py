"""Fly the rAPDHG controller in three implementations under one compute budget.

The quadcopter has a fixed slice of each sampling period in which to solve its
quadratic program, and it will run as many iterations as fit.  How many fit is
not a property of the algorithm: it is a property of the code.  So the
algorithm, the problem, the reference, and the cold start are held fixed, and the
only thing that varies between the flights is what runs the iteration — NumPy,
the solver as first emitted by code generation, and the same solver after the
optimization phase.

Every arm is the same map from parameters to iterates, checked to the code
generator's own equivalence tolerance, so a difference between the flights is a
difference in speed and nothing else.

What distinguishes this study from the ADMM one is where the time goes.  rAPDHG
is matrix free: its iteration is matrix-vector products and vector arithmetic,
with no linear solve, so none of it lands in the deterministic
``fom_linsolve.c`` that the optimization phase is not allowed to touch.

    python -m condensed.run_implementations

Needs ``condensed.generate`` to have emitted both solvers.  Writes
``implementations.csv``, ``implementations_budget.pdf``,
``implementations_flight.pdf``, ``implementations_error.pdf``, and
``implementations.gif`` to ``out/``.
"""

from __future__ import annotations

from common import study as pipeline
from condensed.study import MAX_ITERS, build

#: A conservative lower bound on what one iteration can cost in any arm, in
#: microseconds.  Used only to keep the budget grid inside the schedule tables;
#: it is deliberately below anything measured, because being wrong in this
#: direction drops a budget and being wrong in the other direction stops a run.
FLOOR_US_PER_ITERATION = 0.11

#: Compute budgets per sampling instant, in microseconds.  The grid spans two
#: regimes.  Tracking alone is settled within a few hundred microseconds by
#: every arm; respecting the tilt bound is not.  The bound's multipliers are
#: built by dual steps, and on the figure eight the controller needs a few
#: thousand iterations before the flight stays inside it.  At 0.13 us an
#: iteration that is a few hundred microseconds; at 10 us it is longer than
#: the 10 ms sampling period.  So the grid runs to 2.5 ms, a quarter of the
#: period, which is where the arms separate on the constraint rather than on
#: the tracking error.
#:
#: The floor is set by the slowest arm rather than the fastest.  NumPy costs
#: 10 us an iteration, so anything under about 25 us buys it a single iteration
#: and several budgets in a row report the same flight.
BUDGETS_US = (25.0, 63.0, 160.0, 400.0, 1000.0, 2500.0)


def affordable(budgets, max_iters: int) -> list[float]:
    """Drop budgets that could buy more iterations than the tables describe.

    Past the end of a schedule table the emitted accessor clamps and the
    coefficients freeze, so ``common.study`` refuses to fly there.  Rather than
    let a run die most of the way through a sweep, the grid is trimmed up front
    and what was dropped is said out loud.
    """
    ceiling = max_iters * FLOOR_US_PER_ITERATION
    keep = [b for b in budgets if b <= ceiling]
    for dropped in (b for b in budgets if b > ceiling):
        print(
            f"dropping the {dropped:.0f} us budget: at {FLOOR_US_PER_ITERATION} us an "
            f"iteration it could exceed the {max_iters} iterations the schedule tables "
            f"cover, and past that the coefficients freeze"
        )
    return keep


def main() -> None:
    parser = pipeline.parser()
    parser.description = __doc__
    parser.set_defaults(budgets=list(BUDGETS_US), figure_budget=40.0)
    args = parser.parse_args()
    args.budgets = affordable(args.budgets, MAX_ITERS)
    pipeline.run_implementations(build(), args)


if __name__ == "__main__":
    main()
