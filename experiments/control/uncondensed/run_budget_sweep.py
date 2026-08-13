"""Fly the quadcopter on the uncondensed formulation at a range of budgets.

    python -m uncondensed.run_budget_sweep

Writes the table, the trajectories, and the figures to ``uncondensed/out/``.
See ``common/sweep.py`` for what each output is and which flags are accepted.
"""

from __future__ import annotations

from pathlib import Path

from common import sweep
from common.reference import figure_eight
from uncondensed.controller import build

OUT = Path(__file__).parent / "out"


def main() -> None:
    args = sweep.parser(__doc__).parse_args()
    sweep.run(build(figure_eight()), OUT, args)


if __name__ == "__main__":
    main()
