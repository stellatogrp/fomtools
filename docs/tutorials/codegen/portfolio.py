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

    from docs.tutorials.helper_functions.portfolio import (
        admm_step,
        make_portfolio_data,
        solve_admm_numpy,
    )
    from docs.tutorials.helper_functions.codegen_benchmarks import run_codegen_benchmark
    from fomtools.codegen import CodegenConfig, TerminationSpec
    import fomtools as ft

    timing_file = start / "_recorded_timings.json"

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        ft,
        admm_step,
        examples_dir,
        make_portfolio_data,
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
    # Portfolio: Python vs Codegen

    [Back to the code generation tutorials](./)

    In this tutorial we benchmark a small mean-variance portfolio problem.
    The portfolio weights must satisfy lower bounds, upper bounds, and a budget equality.

    The solver uses the alternating direction method of multipliers: each iteration clips an auxiliary variable to the constraint box, solves a small linear system, and updates the auxiliary variable.
    We compare a direct NumPy implementation with the verified generated C solver.
    The problem construction, ADMM step, and NumPy reference solver are shared with the [portfolio verification tutorial](../verification/portfolio.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate a positive definite covariance matrix and a vector of expected returns.
    Both implementations start from the same fixed vectors and run the same update rule for the same number of iterations.
    """)
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


@app.cell(hide_code=True)
def _(
    A,
    CodegenConfig,
    K,
    TerminationSpec,
    admm_step,
    expected_return,
    ft,
    lhs,
    lower,
    n,
    rho,
    sigma,
    upper,
    v0,
    x0,
):
    verifier = ft.Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
    )

    x = verifier.iterate(n, lb=x0, ub=x0)
    v = verifier.iterate(n + 1, lb=v0, ub=v0)
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
        example="portfolio",
        name="portfolio_codegen_speedup",
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
