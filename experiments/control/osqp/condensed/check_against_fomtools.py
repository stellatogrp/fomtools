"""Check the NumPy controller against the same algorithm recorded in fomtools.

The point of the experiment is that the analyzed algorithm and the deployed
algorithm are one object.  That only holds if the iteration written by hand in
``admm.py`` is the iteration a ``Verifier`` records, so this runs the recording
of :mod:`condensed.recording` three ways and compares them; see
:func:`common.study.check_recording` for what each execution establishes.

It also saves the box of linear costs a flight visits.  That box is the parameter
set the verification problem searches over and the one the generated code is
checked on, so it belongs with the experiment rather than being invented later.

    python -m condensed.check_against_fomtools
"""

from __future__ import annotations

import numpy as np

from common import study as pipeline
from condensed import admm
from condensed.study import OUT, build

#: Iterations run in the comparison.  Long enough that the saturated-linear step
#: has been active for a while and short enough to keep the faithful replay cheap.
HORIZON = 50


def main() -> None:
    s = build()
    q = s.boxes["p0"]
    print(
        f"linear-cost box over a flight: {q.dim} coordinates, widest side {np.max(q.ub - q.lb):.3f}"
    )

    pipeline.check_recording(s, HORIZON)

    # Compared by objective value, not by iterate.  Many coordinates of the
    # solution sit exactly on the thrust box and the free subspace is poorly
    # conditioned, so the minimizer is not numerically well determined: OSQP and
    # an interior-point solver disagree with each other by more than either
    # disagrees with the converged ADMM iterate.  The objective is what the
    # controller is actually optimizing and all three agree on it.
    exact = admm.solve_exact(s.problem, q.mid)
    converged = admm.solve(s.problem, q.mid, 20_000)
    objective = lambda x: float(0.5 * x @ s.problem.P @ x + q.mid @ x)  # noqa: E731
    gap = (objective(converged.z) - objective(exact)) / abs(objective(exact))
    print(f"  converged ADMM vs OSQP       {gap:+.3e} relative objective gap")
    print(f"  final fixed-point residual   {converged.residual:.3e}")

    print(f"\nwrote {s.boxes.save(OUT / 'parameter_box.npz')}")

    failures = [
        name
        for name, err, tol in (
            # One-sided: the ADMM iterate must not be meaningfully worse than the
            # reference solution.  Coming out slightly lower is fine.
            ("OSQP", max(gap, 0.0), 1e-8),
            ("its own fixed point", converged.residual, 1e-6),
        )
        if not err < tol
    ]
    if failures:
        raise SystemExit(f"FAILED: disagreement with {', '.join(failures)}")
    print("OK: the hand-written controller, the recording, and the extraction agree.")


if __name__ == "__main__":
    main()
