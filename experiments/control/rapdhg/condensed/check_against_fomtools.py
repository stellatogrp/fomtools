"""Confirm the controller, the recording, and the extraction are one algorithm.

Four things have to line up before any of the timings mean anything.

1. The flat six-slot recurrence :func:`condensed.rapdhg.solve` runs is Algorithm
   1 of the paper.  It is not obviously so: the restart has been turned from a
   change of control flow into a coefficient that is nonzero every K-th
   iteration.  :func:`condensed.rapdhg.solve_as_written` is the nested loop as
   the paper writes it, and the two are compared across restart boundaries.
2. The recording extracts, and the extracted representation reproduces the
   recurrence.  ``common.study.check_recording`` does this.
3. The materializing saturation never binds.  It is in the recording only to
   stop the one expensive product being emitted three times, so if it ever
   clipped it would silently change the algorithm.
4. The converged iterate solves the quadratic program, checked against Clarabel,
   which takes the one-sided inequality form directly and so checks the
   reformulation of ``mpc.py`` as well as the iteration; and that solution
   respects the tilt rows, which checks the sign of the state constraint.

    python -m condensed.check_against_fomtools
"""

from __future__ import annotations

import numpy as np

from common import study as pipeline
from condensed import rapdhg
from condensed.recording import MATERIALIZE_BOUND
from condensed.study import OUT, build

#: Horizon for the recording checks.  Several restart periods, so the schedule
#: tables are exercised past their first cycle rather than only inside it.
HORIZON = 100

#: Counts at which the flat recurrence is compared with the paper's nested loop.
#: Chosen around a restart boundary, where the two forms have the most room to
#: disagree.
CROSS_COUNTS = (1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257)

#: Iterations for the reference solve.  rAPDHG converges slowly on this program;
#: this is enough for a five-digit agreement in objective value.
CONVERGED = 100_000


def main() -> None:
    study = build()
    problem = study.problem
    q = study.boxes["p0"].mid
    h = study.boxes["p1"].mid
    x_mid = study.boxes.state

    print(f"K = {problem.restart}, w = {problem.primal_weight}")
    print(f"||Q|| = {problem.norm_q:.6g}, ||G|| = {problem.norm_g:.6g}")

    # 0. the structural products the NumPy arm uses are the stored matrix.
    #    They are what keeps the NumPy arm from being charged 3200 multiply-adds
    #    for a product the generated C emits sparse and does in 80.  The tilt
    #    rows are a real product either way, so agreement is to round-off.
    rng = np.random.default_rng(0)
    probe_x = rng.normal(size=problem.n)
    probe_y = rng.normal(size=problem.m)
    err_structure = max(
        float(np.max(np.abs(problem.apply_g(probe_x) - problem.G @ probe_x))),
        float(np.max(np.abs(problem.apply_gt(probe_y) - problem.G.T @ probe_y))),
    )

    # 1. the flat recurrence is the paper's algorithm.
    tables = rapdhg.schedule_tables(problem, max(CROSS_COUNTS) + 1)
    worst_form = 0.0
    for count in CROSS_COUNTS:
        flat = rapdhg.solve(problem, q, count, tables=tables, h=h)
        x_paper, y_paper = rapdhg.solve_as_written(problem, q, count, h=h)
        worst_form = max(
            worst_form,
            float(np.max(np.abs(flat.x - x_paper))),
            float(np.max(np.abs(flat.y - y_paper))),
        )

    # 2. the recording extracts and reproduces it.
    pipeline.check_recording(study, HORIZON)

    # 3. the materializing saturation never binds.
    reach = 0.0
    for count in (1, 8, 64, 512, pipeline.CHECK_ITERATIONS[-1]):
        result = rapdhg.solve(problem, q, count, h=h)
        reach = max(reach, float(np.max(np.abs(result.x))))

    # 4. the converged iterate solves the program.
    converged = rapdhg.solve(problem, q, CONVERGED, h=h)
    x_exact = rapdhg.solve_exact(problem, q, h=h)
    upper, lower = problem.state_rhs(x_mid)
    tilt_slack = float(
        min(np.min(upper - problem.C @ x_exact), np.min(lower + problem.C @ x_exact))
    )
    obj = lambda x: 0.5 * x @ problem.P @ x + q @ x  # noqa: E731
    gap = (obj(converged.x) - obj(x_exact)) / max(abs(obj(x_exact)), 1.0)

    print(f"\n  structural G products vs G      {err_structure:.3e}")
    print(f"  flat recurrence vs Algorithm 1  {worst_form:.3e}")
    print(f"  converged rAPDHG vs Clarabel    {gap:+.3e} relative objective gap")
    print(
        f"  relative KKT error at {CONVERGED}   "
        f"{rapdhg.relkkt(problem, q, converged.x, converged.y, h=h):.3e}"
    )
    print(
        f"  tilt slack of the Clarabel solution  {tilt_slack:+.3e} rad (>= 0 means within the bound)"
    )
    print(
        f"  widest iterate seen             {reach:.4g} (saturation binds at {MATERIALIZE_BOUND:.0g})"
    )

    failures = []
    if not err_structure < 1e-12:
        failures.append("the structural G products are not the stored matrix")
    if not tilt_slack > -1e-8:
        failures.append("the Clarabel solution violates the tilt rows")
    if not worst_form < 1e-9:
        failures.append("the flat recurrence is not the algorithm the paper writes")
    if not abs(gap) < 1e-6:
        failures.append("the converged iterate does not solve the program")
    if not reach < 0.01 * MATERIALIZE_BOUND:
        failures.append("the materializing saturation is close to binding")
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))

    print(f"\nwrote {study.boxes.save(OUT / 'parameter_box.npz')}")
    print("OK: the controller, the recording, and the extraction agree.")


if __name__ == "__main__":
    main()
