import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import statistics
    import sys
    import textwrap
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
    from fomtools.codegen import CodegenConfig, TerminationSpec
    from fomtools.codegen.agent import make_agent_driver
    from fomtools.codegen.harness import find_compiler
    from fomtools.verifier import Verifier

    # Generating a verified solver needs a real coding-agent backend. When it
    # isn't configured (e.g. a headless docs build), the timing cells below fall
    # back to the timings recorded on the benchmark machine instead of raising.
    agent_available = os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1"
    recorded_timings = json.loads((start / "_recorded_timings.json").read_text())

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        Verifier,
        admm_step,
        agent_available,
        examples_dir,
        find_compiler,
        make_agent_driver,
        mo,
        np,
        recorded_timings,
        statistics,
        solve_admm_numpy,
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
The ADMM step and NumPy reference solver are shared with the [control verification tutorial](control_verification.html).
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
def _(CodegenConfig, K, M, TerminationSpec, Verifier, admm_step, lower, lhs, n_total, q, rho, sigma, upper, v0, z0):
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
def _(K, M, lhs, lower, q, rho, sigma, solve_admm_numpy, statistics, time, upper, v0, z0):
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


@app.cell
def _(
    agent_available,
    config,
    find_compiler,
    make_agent_driver,
    next_state,
    out_dir,
    python_us,
    recorded_timings,
    s_state,
    verifier,
):
    if not agent_available:
        # Show the recorded row as published. The live python_us measured just
        # above is deliberately NOT reused here: pairing this machine's Python
        # time with another machine's C time would invent a cross-machine ratio.
        _rec = recorded_timings["examples"]["control"]
        table_python_us = _rec["python_us"]
        codegen_us = _rec["c_initial_us"]
        iteration_us = _rec["c_final_us"]
        codegen_speedup = _rec["codegen_speedup"]
        iteration_speedup = _rec["iteration_speedup"]
        total_speedup = _rec["total_speedup"]
        optimization_status = _rec["optimization_status"]
        solver_path = None
        timings_live = False
        print(
            "Skipping code generation: showing recorded timings. Set "
            "FOMTOOLS_CODEGEN_AGENT=1 (and configure a coding-agent backend) to "
            "generate, verify and re-time the embedded solver."
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

        table_python_us = python_us
        codegen_us = result.timing.c_initial_us
        iteration_us = result.timing.c_final_us
        codegen_speedup = python_us / codegen_us
        iteration_speedup = codegen_us / iteration_us
        total_speedup = python_us / iteration_us
        optimization_status = "kept" if result.optimization_kept else "not kept"
        solver_path = out_dir / result.target.solver_source_path()
        timings_live = True
    return (
        codegen_speedup,
        codegen_us,
        iteration_speedup,
        iteration_us,
        optimization_status,
        solver_path,
        table_python_us,
        timings_live,
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
    recorded_timings,
    solver_path,
    table_python_us,
    timings_live,
    total_speedup,
):
    if timings_live:
        _footer = f"""
Generated solver directory: `{out_dir}`

Generated solver source: `{solver_path}`
"""
    else:
        _prov = recorded_timings["provenance"]
        _footer = f"""
> **Recorded measurement.** These timings were recorded by {_prov["recorded_by"]} on
> {_prov["recorded_date"]} (commit `{_prov["recorded_commit"]}`) and are **not** recomputed in
> this build, since a docs build has no coding-agent backend and its timings would in any case
> come from a shared, noisy runner. Set `FOMTOOLS_CODEGEN_AGENT=1` with a coding-agent backend
> to generate, verify and re-time the solver on your own machine. Every example's recorded
> numbers are collected in the [code generation speedup results](codegen_speedup_results.md).
"""
    summary = mo.md(f"""
## Timing summary

One complete solve runs `{K}` iterations.

| implementation | median / mean per solve |
|---|---:|
| direct NumPy Python | `{table_python_us:.3f} µs` |
| generated C after code generation | `{codegen_us:.3f} µs` |
| generated C after optimization iteration | `{iteration_us:.3f} µs` |

The speedup from code generation is `{codegen_speedup:.1f}x`, while the speedup from the optimization iteration is `{iteration_speedup:.2f}x`, totaling a speedup of `{total_speedup:.1f}x`.

The optimization iteration uses a live coding agent to edit `src/fom_solver.c`; the edited solver was `{optimization_status}` after verification and timing.
{_footer}"""
    )
    summary
    return


if __name__ == "__main__":
    app.run()
