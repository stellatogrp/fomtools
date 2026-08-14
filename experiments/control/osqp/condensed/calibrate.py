"""Generate the emitted C solver for the condensed formulation and time it.

This is the number the budget study turns on.  A compute budget buys a different
iteration count in each implementation, and the iteration count is what the
tracking error depends on, so until the cost of an iteration is known there is no
principled budget to sweep.

    python -m condensed.calibrate            # generate if needed, then measure
    python -m condensed.calibrate --force    # regenerate from scratch

Needs ``FOMTOOLS_CODEGEN_AGENT=1`` the first time; afterwards the verified solver
in ``out/codegen/emitted`` is reused.
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
    pipeline.calibrate(s, force=args.force, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
