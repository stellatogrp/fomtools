"""Fly the uncondensed controller in three implementations under one compute budget.

The quadcopter has a fixed slice of each sampling period in which to solve its
quadratic program, and it will run as many ADMM iterations as fit.  How many fit
is not a property of the algorithm: it is a property of the code.  So the
algorithm, the problem, the reference, and the cold start are held fixed, and the
only thing that varies between the flights is what runs the iteration — NumPy,
the solver as first emitted by code generation, and the same solver after the
optimization phase.

Every arm is the same map from parameters to iterates, checked to the code
generator's own equivalence tolerance, so a difference between the flights is a
difference in speed and nothing else.

    python -m uncondensed.run_implementations

Needs ``uncondensed.generate`` to have emitted both solvers.  Writes to ``out/``,
under names of its own so the iteration-count sweep in ``run_budget_sweep`` keeps
its figures: ``implementations.csv``, ``implementations_budget.pdf``,
``implementations_flight.pdf``, ``implementations_error.pdf``,
``implementations.gif``.
"""

from __future__ import annotations

from common import study as pipeline
from uncondensed.study import build


def main() -> None:
    parser = pipeline.parser()
    parser.description = __doc__
    args = parser.parse_args()
    pipeline.run_implementations(build(), args)


if __name__ == "__main__":
    main()
