"""Generate both C solvers for the condensed controller.

The translation phase produces ``out/codegen/emitted``; that directory is copied
and the optimization phase runs on the copy, producing ``out/codegen/optimized``.
Keeping both is the point: they are two of the three implementations the budget
study compares, and they descend from the same translation, so the difference
between them is what the optimization phase did rather than two unrelated agent
runs.

    python -m condensed.generate            # generate whatever is missing
    python -m condensed.generate --force    # regenerate both from scratch

Needs ``FOMTOOLS_CODEGEN_AGENT=1``.  Afterwards
``python -m condensed.run_implementations`` flies them.
"""

from __future__ import annotations

import argparse

from common import codegen_driver
from common import study as pipeline
from condensed.study import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate rather than reuse")
    parser.add_argument("--max-iters", type=int, default=codegen_driver.DEFAULT_MAX_ITERS)
    args = parser.parse_args()

    s = build(force_boxes=args.force)
    print(s.numpy_controller.describe())
    pipeline.generate(s, force=args.force, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
