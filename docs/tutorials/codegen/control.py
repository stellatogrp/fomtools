import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import statistics
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np

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

    from docs.tutorials.helper_functions.control import admm_step, solve_admm_numpy
    from docs.tutorials.helper_functions.codegen_benchmarks import run_codegen_benchmark
    from fomtools.codegen import CodegenConfig, TerminationSpec
    from fomtools.verifier import Verifier

    timing_file = start / "_recorded_timings.json"

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        Verifier,
        admm_step,
        examples_dir,
        mo,
        np,
        run_codegen_benchmark,
        solve_admm_numpy,
        statistics,
        timing_file,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Control: Python vs Codegen

    [Back to the code generation tutorials](./)

    In this tutorial we benchmark a small finite-horizon control problem for a linear dynamical system.
    The solver uses the alternating direction method of multipliers: each iteration solves a small linear system, projects onto box constraints, and updates the auxiliary variables.

    We compare a direct NumPy implementation with the verified generated C solver.
    The ADMM step and NumPy reference solver are shared with the [control verification tutorial](../verification/control.html).
    The generated C is timed after code generation, then timed again after the C-editing optimization pass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We fix the initial state, input bounds, prediction horizon, and quadratic costs.
    Both implementations run the same update rule for the same number of iterations.
    """)
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
    return K, M, lhs, lower, n_total, out_dir, q, rho, sigma, upper, v0, z0


@app.cell(hide_code=True)
def _(
    CodegenConfig,
    K,
    M,
    TerminationSpec,
    Verifier,
    admm_step,
    lhs,
    lower,
    n_total,
    q,
    rho,
    sigma,
    upper,
    v0,
    z0,
):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    z = verifier.add_iterate(n_total, lb=z0, ub=z0)
    v = verifier.add_iterate(n_total, lb=v0, ub=v0)
    def project(VP, v_prev):
        return VP.step.satlin(v_prev, l=lower, u=upper)

    z_next, v_next, _ = admm_step(
        verifier, M, q, rho, sigma, z, v, project, lhs
    )

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    next_state = [z_next, v_next]
    s_state = [z, v]
    return config, next_state, s_state, verifier


@app.cell
def _(
    K,
    M,
    lhs,
    lower,
    q,
    rho,
    sigma,
    solve_admm_numpy,
    statistics,
    time,
    upper,
    v0,
    z0,
):
    def control_solve():
        return solve_admm_numpy(
            M, q, rho, sigma, lhs, lower, upper, z0, v0, max_iters=K
        )

    for _ in range(3):
        control_solve()

    python_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        control_solve()
        python_times.append((time.perf_counter() - t0) * 1.0e6)

    python_us = statistics.median(python_times)
    return (python_us,)


@app.cell(hide_code=True)
def _(
    config,
    next_state,
    out_dir,
    python_us,
    run_codegen_benchmark,
    s_state,
    timing_file,
    verifier,
):
    timings = run_codegen_benchmark(
        config=config,
        example="control",
        name="control_codegen_speedup",
        next_state=next_state,
        out_dir=out_dir,
        params=None,
        python_us=python_us,
        state=s_state,
        timing_file=timing_file,
        verifier=verifier,
    )
    return (timings,)


@app.cell(hide_code=True)
def _(
    K,
    mo,
    out_dir,
    timings,
):
    if timings["live"]:
        _footer = f"""
    Generated solver directory: `{out_dir}`

    Generated solver source: `{timings["solver_path"]}`
    """
    else:
        _footer = ""
    summary = mo.md(f"""
    ## Timing summary

    One complete solve runs `{K}` iterations.

    | implementation | median / mean per solve |
    |---|---:|
    | direct NumPy Python | `{timings["python_us"]:.3f} µs` |
    | generated C after code generation | `{timings["codegen_us"]:.3f} µs` |
    | generated C after optimization iteration | `{timings["iteration_us"]:.3f} µs` |

    The speedup from code generation is `{timings["codegen_speedup"]:.1f}x`, while the speedup from the optimization iteration is `{timings["iteration_speedup"]:.2f}x`, totaling a speedup of `{timings["total_speedup"]:.1f}x`.
    {_footer}"""
    )
    summary
    return


if __name__ == "__main__":
    app.run()
