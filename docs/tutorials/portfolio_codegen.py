import marimo

__generated_with = "0.23.9"
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

    from docs.tutorials.helper_functions.portfolio import (
        admm_step,
        make_portfolio_data,
        solve_admm_numpy,
    )
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
        make_portfolio_data,
        mo,
        np,
        recorded_timings,
        solve_admm_numpy,
        statistics,
        textwrap,
        time,
    )


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
    # Portfolio: Python vs Codegen

    In this tutorial we benchmark a small mean-variance portfolio problem.
    The portfolio weights must satisfy lower bounds, upper bounds, and a budget equality.

    The solver uses the alternating direction method of multipliers: each iteration clips an auxiliary variable to the constraint box, solves a small linear system, and updates the auxiliary variable.
    We compare a direct NumPy implementation with the verified generated C solver.
    The problem construction, ADMM step, and NumPy reference solver are shared with the [portfolio verification tutorial](portfolio_verification.html).
    """)
    )
    return


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
    We generate a positive definite covariance matrix and a vector of expected returns.
    Both implementations start from the same fixed vectors and run the same update rule for the same number of iterations.
    """)
    )
    return


@app.cell
def _(examples_dir, make_portfolio_data, np):
    n = 4
    covariance, expected_return, A, lower, upper = make_portfolio_data(
        n=n, seed=0, position_limit=0.6
    )

    sigma = 1e-4
    rho = 1.0
    lhs = covariance + sigma * np.eye(n) + rho * A.T @ A
    x0 = np.zeros(n)
    v0 = 0.5 * np.ones(n + 1)
    K = 200
    out_dir = examples_dir / "_codegen_speedups" / "portfolio_codegen_speedup"
    return (
        A,
        K,
        expected_return,
        lhs,
        lower,
        n,
        out_dir,
        rho,
        sigma,
        upper,
        v0,
        x0,
    )


@app.cell
def _(
    A,
    CodegenConfig,
    K,
    TerminationSpec,
    Verifier,
    admm_step,
    expected_return,
    lhs,
    lower,
    n,
    rho,
    sigma,
    upper,
    v0,
    x0,
):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    x = verifier.add_iterate(n, lb=x0, ub=x0)
    v = verifier.add_iterate(n + 1, lb=v0, ub=v0)
    x_next, v_next, _ = admm_step(
        verifier, A, lower, upper, expected_return, rho, sigma, x, v, lhs
    )

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    next_state = [x_next, v_next]
    s_state = [x, v]
    return config, next_state, s_state, verifier


@app.cell
def _(
    A,
    K,
    expected_return,
    lhs,
    lower,
    rho,
    sigma,
    solve_admm_numpy,
    statistics,
    time,
    upper,
    v0,
    x0,
):
    def portfolio_solve():
        return solve_admm_numpy(
            A,
            lower,
            upper,
            expected_return,
            rho,
            sigma,
            lhs,
            x0,
            v0,
            max_iters=K,
        )

    for _ in range(3):
        portfolio_solve()

    python_times = []
    for _ in range(40):
        t0 = time.perf_counter()
        portfolio_solve()
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
        _rec = recorded_timings["examples"]["portfolio"]
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
            name="portfolio_codegen_speedup",
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
