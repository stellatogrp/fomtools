import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import os
    import statistics
    import sys
    import textwrap
    import time
    from pathlib import Path

    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "fomtools").is_dir():
            repo_root = candidate
            break
    else:
        repo_root = start

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    examples_dir = repo_root / "examples"

    from fomtools.codegen import CodegenConfig, TerminationSpec
    from fomtools.codegen.agent import make_agent_driver
    from fomtools.codegen.harness import find_compiler
    from fomtools.verifier import Verifier

    # Generating a verified solver needs a real coding-agent backend. When it
    # isn't configured (e.g. a headless docs build), the codegen and timing
    # cells below skip with a message instead of raising.
    agent_available = os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1"

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        Verifier,
        agent_available,
        examples_dir,
        find_compiler,
        make_agent_driver,
        mo,
        np,
        statistics,
        textwrap,
        time,
    )


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
# Optimal Control: Python vs Codegen

In this tutorial we benchmark a small finite-horizon control problem for a linear dynamical system.
The solver uses the alternating direction method of multipliers: each iteration solves a small linear system, projects onto box constraints, and updates the auxiliary variables.

We compare a direct NumPy implementation with the verified generated C solver.
The generated C is timed after code generation, then timed again after the C-editing optimization pass.
""")
    )
    return


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
We fix the initial state, input bounds, prediction horizon, and quadratic costs.
Both implementations run the same update rule for the same number of iterations.
""")
    )
    return


@app.cell
def _(examples_dir, np):
    A_dyn = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0992, 0.0, 0.0],
            [0.0, -0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, -0.0016, 0.0, 0.0, 0.0992, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0992],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0488, 0.0, 0.0, 0.9846, 0.0, 0.0],
            [0.0, -0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0488, 0.0, 0.0, 0.9846, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9846],
        ]
    )
    B_dyn = np.array(
        [
            [0.0, -0.0726, 0.0, 0.0726],
            [-0.0726, 0.0, 0.0726, 0.0],
            [-0.0152, 0.0152, -0.0152, 0.0152],
            [-0.0, -0.0006, -0.0, 0.0006],
            [0.0006, 0.0, -0.0006, 0.0],
            [0.0106, 0.0106, 0.0106, 0.0106],
            [0.0, -1.4512, 0.0, 1.4512],
            [-1.4512, 0.0, 1.4512, 0.0],
            [-0.3049, 0.3049, -0.3049, 0.3049],
            [-0.0, -0.0236, 0.0, 0.0236],
            [0.0236, 0.0, -0.0236, 0.0],
            [0.2107, 0.2107, 0.2107, 0.2107],
        ]
    )

    n_state, n_input = B_dyn.shape
    horizon = 2
    K = 200

    Q = np.diag([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    R = 0.1 * np.eye(n_input)
    x_initial = 2.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_min = -0.99 * np.ones(n_input)
    u_max = 2.41 * np.ones(n_input)

    n_control = (horizon - 1) * n_input
    n_total = n_state + n_control
    Sx = np.vstack([np.linalg.matrix_power(A_dyn, i) for i in range(horizon)])
    Sz_mid = np.zeros(((horizon - 1) * n_state, n_control))
    for i in range(horizon - 1):
        Sz_mid[i * n_state : (i + 1) * n_state, :] = np.linalg.matrix_power(A_dyn, i) @ B_dyn
    Sz = np.vstack([np.zeros((n_state, n_control)), Sz_mid])

    Q_block = np.kron(np.eye(horizon), Q)
    R_block = np.kron(np.eye(horizon - 1), R)
    P = np.block(
        [
            [Sx.T @ Q_block @ Sx, Sx.T @ Q_block @ Sz],
            [Sz.T @ Q_block @ Sx, Sz.T @ Q_block @ Sz + R_block],
        ]
    )

    M = np.eye(n_total)
    lower = np.hstack([x_initial, np.kron(np.ones(horizon - 1), u_min)])
    upper = np.hstack([x_initial, np.kron(np.ones(horizon - 1), u_max)])
    sigma = 1e-4
    rho = 1.0
    lhs = P + sigma * np.eye(n_total) + rho * (M.T @ M)
    q = np.zeros(n_total)

    z0 = np.zeros(n_total)
    v0 = 0.5 * np.ones(n_total)
    out_dir = examples_dir / "_codegen_speedups" / "control_codegen_speedup"
    return K, M, lower, lhs, n_total, out_dir, q, rho, sigma, upper, v0, z0


@app.cell
def _(CodegenConfig, K, M, TerminationSpec, Verifier, lower, lhs, n_total, q, rho, sigma, upper, v0, z0):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    z = verifier.add_iterate(n_total, lb=z0, ub=z0)
    v = verifier.add_iterate(n_total, lb=v0, ub=v0)
    w = verifier.step.satlin(v, l=lower, u=upper)
    rhs = sigma * z - q + rho * M.T @ (2 * w - v)
    z_next = verifier.step.implicit_linear(rhs, output_mat=lhs)
    v_next = v + M @ z_next - w

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    next_state = [z_next, v_next]
    s_state = [z, v]
    return config, next_state, s_state, verifier


@app.cell
def _(K, M, lhs, lower, np, q, rho, sigma, statistics, time, upper, v0, z0):
    def control_solve():
        z_py = z0.copy()
        v_py = v0.copy()
        for _ in range(K):
            w_py = np.minimum(np.maximum(v_py, lower), upper)
            rhs_py = sigma * z_py - q + rho * M.T @ (2 * w_py - v_py)
            z_py = np.linalg.solve(lhs, rhs_py)
            v_py = v_py + M @ z_py - w_py
        return np.concatenate([z_py, v_py])

    for _ in range(3):
        control_solve()

    python_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        control_solve()
        python_times.append((time.perf_counter() - t0) * 1.0e6)

    python_us = statistics.median(python_times)
    return (python_us,)


@app.cell
def _(
    agent_available,
    config,
    find_compiler,
    make_agent_driver,
    next_state,
    out_dir,
    python_us,
    s_state,
    verifier,
):
    if not agent_available:
        codegen_us = iteration_us = None
        codegen_speedup = iteration_speedup = total_speedup = None
        optimization_status = solver_path = None
        print(
            "Skipping code generation: set FOMTOOLS_CODEGEN_AGENT=1 (and configure a "
            "coding-agent backend) to generate, verify and time the embedded solver."
        )
    elif find_compiler() is None:
        raise RuntimeError("No C compiler found. Install cc/gcc/clang to run codegen timing.")
    else:
        result = verifier.codegen(
            out_dir,
            next_state=next_state,
            config=config,
            driver=make_agent_driver(config=config),
            state=s_state,
            name="control_codegen_speedup",
            run_phase_b=True,
            profile=True,
        )
        if not result.translate.success:
            raise RuntimeError(result.translate.log[-4000:])

        if (
            result.timing is None
            or result.timing.c_initial_us is None
            or result.timing.c_final_us is None
        ):
            raise RuntimeError("Code generation finished without the requested timing report.")

        codegen_us = result.timing.c_initial_us
        iteration_us = result.timing.c_final_us
        codegen_speedup = python_us / codegen_us
        iteration_speedup = codegen_us / iteration_us
        total_speedup = python_us / iteration_us
        optimization_status = "kept" if result.optimization_kept else "not kept"
        solver_path = out_dir / result.target.solver_source_path()
    return (
        codegen_speedup,
        codegen_us,
        iteration_speedup,
        iteration_us,
        optimization_status,
        solver_path,
        total_speedup,
    )


@app.cell(hide_code=True)
def _(
    K,
    codegen_speedup,
    codegen_us,
    iteration_speedup,
    iteration_us,
    mo,
    optimization_status,
    out_dir,
    python_us,
    solver_path,
    total_speedup,
):
    if codegen_speedup is None:
        summary = mo.md(f"""
## Timing summary

One complete solve runs `{K}` iterations, taking `{python_us:.3f} us` in direct NumPy Python.

Code generation was skipped, so the generated C timings are unavailable in this build. Set
`FOMTOOLS_CODEGEN_AGENT=1` (with a coding-agent backend configured) and re-run to generate,
verify and time the embedded solver. Previously recorded numbers for every example are
collected in the [code generation speedup results](codegen_speedup_results.md).
"""
        )
    else:
        summary = mo.md(f"""
## Timing summary

One complete solve runs `{K}` iterations.

| implementation | median / mean per solve |
|---|---:|
| direct NumPy Python | `{python_us:.3f} us` |
| generated C after code generation | `{codegen_us:.3f} us` |
| generated C after optimization iteration | `{iteration_us:.3f} us` |

The speedup from code generation is `{codegen_speedup:.1f}x`, while the speedup from the optimization iteration is `{iteration_speedup:.2f}x`, totaling a speedup of `{total_speedup:.1f}x`.

The optimization iteration uses a live coding agent to edit `src/fom_solver.c`; the edited solver was `{optimization_status}` after verification and timing.

Generated solver directory: `{out_dir}`

Generated solver source: `{solver_path}`
"""
        )
    summary
    return


if __name__ == "__main__":
    app.run()
