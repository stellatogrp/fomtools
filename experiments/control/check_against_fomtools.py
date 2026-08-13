"""Check the NumPy controller against the same algorithm recorded in fomtools.

The point of the experiment is that the analyzed algorithm and the deployed
algorithm are one object.  That only holds if the iteration written by hand in
``admm.py`` is the iteration a ``Verifier`` records, so this script records it
and compares three executions on the same data:

1. ``admm.solve``, the NumPy controller the closed-loop simulation flies with;
2. ``run_faithful``, which evaluates the recorded step objects directly,
   solving the affine system rather than reusing any folded inverse;
3. ``ReferenceOracle``, which runs the extracted intermediate representation,
   the object code generation lowers to C.

Agreement between the first two says the hand-written controller is the recorded
algorithm.  Agreement with the third says extraction preserved it, which is the
precondition for generating a solver from this recording.

The script also computes the box of linear costs that a flight actually visits
and saves it.  That box is the parameter set the verification problem searches
over and the one the generated code is checked on, so it belongs with the
experiment rather than being invented later.

    python check_against_fomtools.py
"""

from __future__ import annotations

from pathlib import Path

import admm
import numpy as np
import simulate
from mpc import build_mpc
from reference import figure_eight

TOL = 1e-9
OUT = Path(__file__).parent / "out"


def record(problem, q_lb: np.ndarray, q_ub: np.ndarray):
    """Record one ADMM iteration on a verifier, as equation (16) writes it."""
    from fomtools.verifier import Verifier

    n = problem.n
    zeros = np.zeros(n)

    verifier = Verifier(method="simulator", obbt_solver=None)
    q = verifier.add_param(n, lb=q_lb, ub=q_ub)
    z = verifier.add_iterate(n, lb=zeros, ub=zeros)
    v = verifier.add_iterate(n, lb=zeros, ub=zeros)

    # The constraint matrix of the condensed program is the identity, so
    # M' (2w - v) is 2w - v and the system matrix is P + sigma I + rho I.
    w = verifier.step.satlin(v, l=problem.lo, u=problem.hi)
    rhs = problem.sigma * z - q + problem.rho * (2 * w - v)
    z_next = verifier.step.implicit_linear(
        rhs, output_mat=problem.P + (problem.sigma + problem.rho) * np.eye(n)
    )
    v_next = v + z_next - w

    return verifier, [z_next, v_next], [z, v], q


def cost_box(problem, ref, duration: float = 8.0, iterations: int = 1000):
    """The box of linear costs a flight visits, and one representative cost.

    The flight is simulated with a converged solve, so the box describes the
    states a working controller actually reaches rather than the excursions of
    a diverging one.
    """
    roll = simulate.closed_loop(problem, ref, iterations, duration=duration)
    costs = np.vstack(
        [
            problem.linear_cost(roll.states[k], problem.reference_window(ref, k * problem.dt))
            for k in range(roll.steps)
        ]
    )
    return costs.min(axis=0), costs.max(axis=0), costs[roll.steps // 2]


def main() -> None:
    problem = build_mpc()
    ref = figure_eight()

    q_lb, q_ub, q_mid = cost_box(problem, ref)
    spread = float(np.max(q_ub - q_lb))
    print(f"linear-cost box over a flight: {problem.n} coordinates, widest side {spread:.3f}")

    verifier, next_state, state, q_leaf = record(problem, q_lb, q_ub)

    from fomtools.codegen import CodegenConfig, ReferenceOracle, TerminationSpec, extract_ir
    from fomtools.codegen.oracle import run_faithful

    horizon = 50
    ours = admm.solve(problem, q_mid, horizon)
    ours_state = np.concatenate([ours.z, ours.v])

    faithful = run_faithful(
        verifier, next_state, {q_leaf: q_mid}, n_iters=horizon, state=state, params=[q_leaf]
    )
    err_faithful = float(np.max(np.abs(faithful[-1] - ours_state)))

    config = CodegenConfig(termination=TerminationSpec(max_iters=horizon))
    ir = extract_ir(verifier, next_state, state=state, params=[q_leaf], config=config, name="mpc")
    oracle = ReferenceOracle(ir).solve(param_values={"p0": q_mid})
    err_oracle = float(np.max(np.abs(oracle.solution - ours_state)))

    print(f"extracted iteration: {len(ir.body)} steps, state dimension {ir.state_dim}")
    print(f"  NumPy vs recorded steps      {err_faithful:.3e}")
    print(f"  NumPy vs extracted iteration {err_oracle:.3e}")

    # Compared by objective value, not by iterate.  Many coordinates of the
    # solution sit exactly on the thrust box and the free subspace is poorly
    # conditioned, so the minimizer is not numerically well determined: OSQP and
    # an interior-point solver disagree with each other by more than either
    # disagrees with the converged ADMM iterate.  The objective is what the
    # controller is actually optimizing and all three agree on it.
    exact = admm.solve_exact(problem, q_mid)
    converged = admm.solve(problem, q_mid, 20_000)
    objective = lambda x: float(0.5 * x @ problem.P @ x + q_mid @ x)  # noqa: E731
    gap = (objective(converged.z) - objective(exact)) / abs(objective(exact))
    print(f"  converged ADMM vs OSQP       {gap:+.3e} relative objective gap")
    print(f"  final fixed-point residual   {converged.residual:.3e}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "parameter_box.npz", q_lb=q_lb, q_ub=q_ub, q_mid=q_mid)
    print(f"\nwrote {OUT / 'parameter_box.npz'}")

    failures = [
        name
        for name, err, tol in (
            ("recorded steps", err_faithful, TOL),
            ("extracted iteration", err_oracle, TOL),
            # One-sided: the ADMM iterate must not be meaningfully worse than
            # the reference solution.  Coming out slightly lower is fine.
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
