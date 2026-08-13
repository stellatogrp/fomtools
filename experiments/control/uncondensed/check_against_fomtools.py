"""Check the uncondensed NumPy controller against the same algorithm in fomtools.

As in the condensed study, the iteration written by hand has to be the iteration
a ``Verifier`` records, so this compares three executions on the same data: the
NumPy controller, ``run_faithful`` on the recorded step objects, and
``ReferenceOracle`` on the extracted intermediate representation.

The measured state enters this formulation through the constraint bounds rather
than the cost.  The recording therefore fixes the bounds at one representative
state and checks the algorithm there, which is all that is needed to establish
that the three executions agree: the iteration is the same map whatever numbers
sit in the bounds, and the simulation supplies those numbers afresh every
instant rather than carrying a symbolic parameter.  A recording that left the
state symbolic would be a different question, about code generation rather than
about this controller; see the note in ``README.md``.

    python -m uncondensed.check_against_fomtools
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from common.reference import figure_eight
from common.simulate import closed_loop
from uncondensed import admm
from uncondensed.controller import UncondensedController
from uncondensed.mpc import build_mpc

TOL = 1e-9
OUT = Path(__file__).parent / "out"


def record(problem, lo: np.ndarray, hi: np.ndarray, q_lb: np.ndarray, q_ub: np.ndarray):
    """Record one ADMM iteration with the bounds frozen at one instant."""
    from fomtools.verifier import Verifier

    n, m = problem.n, problem.m
    M = problem.M.toarray()
    rho = problem.rho

    verifier = Verifier(method="simulator", obbt_solver=None)
    q = verifier.add_param(n, lb=q_lb, ub=q_ub)
    zeta = verifier.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    v = verifier.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

    w = verifier.step.satlin(v, l=lo, u=hi)
    rhs = problem.sigma * zeta - q + (M.T * rho) @ (2 * w - v)
    lhs = problem.P.toarray() + problem.sigma * np.eye(n) + (M.T * rho) @ M
    zeta_next = verifier.step.implicit_linear(rhs, output_mat=lhs)
    v_next = v + M @ zeta_next - w

    return verifier, [zeta_next, v_next], [zeta, v], q


def cost_box(problem, ref, duration: float = 8.0, iterations: int = 1000):
    """The box of cost vectors a flight visits, and one representative flight point."""
    roll = closed_loop(UncondensedController(problem, ref), ref, iterations, duration=duration)
    costs = np.vstack(
        [
            problem.linear_cost(problem.reference_window(ref, k * problem.dt))
            for k in range(roll.steps)
        ]
    )
    mid = roll.steps // 2
    return costs.min(axis=0), costs.max(axis=0), costs[mid], roll.states[mid]


def main() -> None:
    problem = build_mpc()
    ref = figure_eight()

    q_lb, q_ub, q_mid, state_mid = cost_box(problem, ref)
    lo, hi = problem.bounds(state_mid)
    print(
        f"program: {problem.n} variables, {problem.m} rows, "
        f"system matrix {problem.system_nnz} entries, factor {problem.factor_nnz}"
    )

    verifier, next_state, state, q_leaf = record(problem, lo, hi, q_lb, q_ub)

    from fomtools.codegen import CodegenConfig, ReferenceOracle, TerminationSpec, extract_ir
    from fomtools.codegen.oracle import run_faithful

    horizon = 20
    ours = admm.solve(problem, q_mid, lo, hi, horizon)
    ours_state = np.concatenate([ours.zeta, ours.v])

    faithful = run_faithful(
        verifier, next_state, {q_leaf: q_mid}, n_iters=horizon, state=state, params=[q_leaf]
    )
    err_faithful = float(np.max(np.abs(faithful[-1] - ours_state)))

    config = CodegenConfig(termination=TerminationSpec(max_iters=horizon))
    ir = extract_ir(
        verifier, next_state, state=state, params=[q_leaf], config=config, name="mpc_uncondensed"
    )
    oracle = ReferenceOracle(ir).solve(param_values={"p0": q_mid})
    err_oracle = float(np.max(np.abs(oracle.solution - ours_state)))

    print(f"extracted iteration: {len(ir.body)} steps, state dimension {ir.state_dim}")
    print(f"  NumPy vs recorded steps      {err_faithful:.3e}")
    print(f"  NumPy vs extracted iteration {err_oracle:.3e}")

    exact = admm.solve_exact(problem, q_mid, lo, hi)
    converged = admm.solve(problem, q_mid, lo, hi, 20_000)
    objective = lambda z: float(0.5 * z @ (problem.P @ z) + q_mid @ z)  # noqa: E731
    gap = (objective(converged.zeta) - objective(exact)) / abs(objective(exact))
    n_eq = problem.horizon * 12
    violation = float(np.max(np.abs((problem.M @ converged.zeta)[:n_eq] - lo[:n_eq])))
    print(f"  converged ADMM vs OSQP       {gap:+.3e} relative objective gap")
    print(f"  worst equality violation     {violation:.3e}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "parameter_box.npz", q_lb=q_lb, q_ub=q_ub, q_mid=q_mid, state=state_mid)
    print(f"\nwrote {OUT / 'parameter_box.npz'}")

    failures = [
        name
        for name, err, tol in (
            ("recorded steps", err_faithful, TOL),
            ("extracted iteration", err_oracle, TOL),
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
