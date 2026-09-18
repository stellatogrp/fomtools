import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
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

    from docs.tutorials.helper_functions.lasso import (
        ista_step,
        make_lasso_data,
        solve_ista_numpy,
    )
    from docs.tutorials.helper_functions.codegen_benchmarks import (
        run_lasso_codegen_benchmark,
    )
    from fomtools.codegen import CodegenConfig, TerminationSpec
    from fomtools.verifier import Verifier

    timing_file = start / "_recorded_timings.json"

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        Verifier,
        examples_dir,
        ista_step,
        make_lasso_data,
        mo,
        np,
        run_lasso_codegen_benchmark,
        solve_ista_numpy,
        statistics,
        textwrap,
        timing_file,
        time,
    )


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
    # Lasso: Python vs Codegen

    In this tutorial we benchmark iterative soft-thresholding for the Lasso problem:

    \[
    \begin{array}{ll}
    \text{minimize} & (1/2)\lVert Ds - x \rVert_2^2 + \lambda \lVert s \rVert_1,
    \end{array}
    \]

    where \(D \in \mathbb{R}^{p \times n}\), \(s \in \mathbb{R}^n\), and \(x\) is the parameter.
    The method is

    \[
    s^{k+1} =
    \mathbf{soft}_{\lambda\eta}\left(s^k - \eta D^\top(Ds^k - x)\right).
    \]

    We compare a direct NumPy implementation with the verified generated C solver.
    The problem construction, ISTA step, and NumPy reference solver are shared with the [Lasso verification tutorial](../verification/lasso.html).
    The generated C is timed after code generation, then timed again after the C-editing optimization pass.
    """)
    )
    return


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
    We generate a small sparse dictionary \(D\), sample sparse signals, and use those samples to define the range of possible inputs \(x\).
    Both implementations run the same update rule for the same number of iterations.
    """)
    )
    return


@app.cell
def _(examples_dir, make_lasso_data, np):
    p, n, lambd = 3, 4, 1e-1
    num_star = 200
    D, x_lb, x_ub, _, eta, lambd = make_lasso_data(
        p=p, n=n, lambd=lambd, num_signals=num_star, seed=0
    )
    x_mid = 0.5 * (x_lb + x_ub)

    K = 200
    s0 = np.zeros(n)

    out_dir = examples_dir / "_codegen_speedups" / "lasso_codegen_speedup"
    return D, K, eta, lambd, n, out_dir, p, s0, x_lb, x_mid, x_ub


@app.cell
def _(
    CodegenConfig,
    D,
    K,
    TerminationSpec,
    Verifier,
    eta,
    ista_step,
    lambd,
    n,
    p,
    s0,
    x_lb,
    x_ub,
):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    x_param = verifier.add_param(p, lb=x_lb, ub=x_ub)
    s = verifier.add_iterate(n, lb=s0, ub=s0)
    next_s, _, _, _ = ista_step(
        verifier, D, eta, lambd, s, s, x_param, 1.0, k=1, momentum=False
    )

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    return config, next_s, s, verifier, x_param


@app.cell(hide_code=True)
def _(D, K, eta, lambd, s0, solve_ista_numpy, statistics, time, x_mid):
    def python_lasso_solve():
        return solve_ista_numpy(D, eta, lambd, s0, x_mid, max_iters=K)

    for _ in range(3):
        python_lasso_solve()

    python_times = []
    for _ in range(40):
        t0 = time.perf_counter()
        python_lasso_solve()
        python_times.append((time.perf_counter() - t0) * 1.0e6)

    python_us = statistics.median(python_times)
    return (python_us,)


@app.cell(hide_code=True)
def _(
    config,
    next_s,
    out_dir,
    python_us,
    run_lasso_codegen_benchmark,
    s,
    timing_file,
    verifier,
    x_param,
):
    timings = run_lasso_codegen_benchmark(
        config=config,
        next_state=next_s,
        out_dir=out_dir,
        python_us=python_us,
        state=s,
        timing_file=timing_file,
        verifier=verifier,
        x_param=x_param,
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
