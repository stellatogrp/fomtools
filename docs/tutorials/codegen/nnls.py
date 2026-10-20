import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import statistics
    import sys
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
    from fomtools.verifier import Verifier
    from docs.tutorials.helper_functions.codegen_benchmarks import run_codegen_benchmark
    from docs.tutorials.helper_functions.nnls import (
        make_nnls_matrix,
        pgd_step,
        solve_pgd_numpy,
    )

    timing_file = start / "_recorded_timings.json"

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        Verifier,
        examples_dir,
        make_nnls_matrix,
        mo,
        np,
        pgd_step,
        run_codegen_benchmark,
        solve_pgd_numpy,
        statistics,
        timing_file,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NNLS: Verified Code Generation

    [Back to the code generation tutorials](./)

    This notebook benchmarks direct NumPy projected gradient descent against
    its verified generated C solver. The problem construction and iteration
    map are shared with the [NNLS verification tutorial](../verification/nnls.html).
    """)
    return


@app.cell
def _(examples_dir, make_nnls_matrix, np):
    m, n = 5, 3
    A = make_nnls_matrix(m=m, n=n)
    x_lb = np.zeros(m)
    x_ub = 0.25 * np.ones(m)
    x_mid = 0.5 * (x_lb + x_ub)

    z0 = 5.05 * np.ones(n)
    L = float(np.linalg.eigvalsh(A.T @ A)[-1])
    eta = 1.0 / L
    K = 200
    out_dir = examples_dir / "_codegen_speedups" / "nnls_codegen_speedup"
    return A, K, eta, m, n, out_dir, x_lb, x_mid, x_ub, z0


@app.cell
def _(
    A,
    CodegenConfig,
    K,
    TerminationSpec,
    Verifier,
    eta,
    m,
    n,
    pgd_step,
    x_lb,
    x_ub,
    z0,
):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    x_param = verifier.add_param(m, lb=x_lb, ub=x_ub)
    z = verifier.add_iterate(n, lb=z0, ub=z0)
    z_next, _, _ = pgd_step(
        verifier, A, eta, z, z, x_param, k=1, momentum=False,
    )

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    next_state = z_next
    s_state = [z]
    params = [x_param]
    return config, next_state, params, s_state, verifier


@app.cell(hide_code=True)
def _(A, K, eta, solve_pgd_numpy, statistics, time, x_mid, z0):
    def nnls_solve():
        return solve_pgd_numpy(A, eta, z0, x_mid, max_iters=K)

    for _ in range(3):
        nnls_solve()

    python_times = []
    for _ in range(40):
        t0 = time.perf_counter()
        nnls_solve()
        python_times.append((time.perf_counter() - t0) * 1.0e6)

    python_us = statistics.median(python_times)
    return (python_us,)


@app.cell(hide_code=True)
def _(
    config,
    next_state,
    out_dir,
    params,
    python_us,
    run_codegen_benchmark,
    s_state,
    timing_file,
    verifier,
):
    timings = run_codegen_benchmark(
        config=config,
        example="nnls",
        name="nnls_codegen_speedup",
        next_state=next_state,
        out_dir=out_dir,
        params=params,
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
