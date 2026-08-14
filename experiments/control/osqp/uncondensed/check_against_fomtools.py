"""Check the uncondensed NumPy controller against the same algorithm in fomtools.

The iteration written by hand has to be the iteration a ``Verifier`` records, so
this runs the recording of :mod:`uncondensed.recording` three ways and compares
them; see :func:`common.study.check_recording` for what each execution
establishes.  It then checks the converged iterate against OSQP.

The recording is the one code generation is run on.  The measured state is a live
parameter there, not a frozen constant: it enters this formulation through the
constraint bounds, and the identity ``sat_[l,u](y) = l + sat_[0,u-l](y-l)`` moves
it into the expression the saturated-linear step is applied to, where code
generation accepts it.  So what this checks and what gets generated are the same
object.

    python -m uncondensed.check_against_fomtools
"""

from __future__ import annotations

import numpy as np

from common import study as pipeline
from uncondensed import admm
from uncondensed.study import OUT, build

#: Iterations run in the comparison.  Long enough that the saturated-linear step
#: has been active for a while and short enough to keep the faithful replay cheap.
HORIZON = 20


def main() -> None:
    s = build()
    problem = s.problem
    print(
        f"program: {problem.n} variables, {problem.m} rows, "
        f"system matrix {problem.system_nnz} entries, factor {problem.factor_nnz}"
    )

    pipeline.check_recording(s, HORIZON)

    q_mid = s.boxes["p0"].mid
    lo, hi = problem.bounds(s.boxes.state)
    exact = admm.solve_exact(problem, q_mid, lo, hi)
    converged = admm.solve(problem, q_mid, lo, hi, 20_000)
    objective = lambda z: float(0.5 * z @ (problem.P @ z) + q_mid @ z)  # noqa: E731
    gap = (objective(converged.zeta) - objective(exact)) / abs(objective(exact))
    n_eq = problem.horizon * 12
    violation = float(np.max(np.abs((problem.M @ converged.zeta)[:n_eq] - lo[:n_eq])))
    print(f"  converged ADMM vs OSQP       {gap:+.3e} relative objective gap")
    print(f"  worst equality violation     {violation:.3e}")

    print(f"\nwrote {s.boxes.save(OUT / 'parameter_box.npz')}")

    failures = [
        name
        for name, err, tol in (
            ("OSQP", abs(gap), 1e-5),
            ("the dynamics", violation, 1e-5),
        )
        if not err < tol
    ]
    if failures:
        raise SystemExit(f"FAILED: disagreement with {', '.join(failures)}")
    print("OK: the hand-written controller, the recording, and the extraction agree.")


if __name__ == "__main__":
    main()
