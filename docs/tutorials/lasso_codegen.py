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

    from docs.tutorials.helper_functions.lasso import (
        ista_step,
        make_lasso_data,
        solve_ista_numpy,
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
        agent_available,
        examples_dir,
        find_compiler,
        ista_step,
        make_agent_driver,
        make_lasso_data,
        mo,
        np,
        recorded_timings,
        statistics,
        solve_ista_numpy,
        textwrap,
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
The problem construction, ISTA step, and NumPy reference solver are shared with the [Lasso verification tutorial](lasso_verification.html).
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
def _(CodegenConfig, D, K, TerminationSpec, Verifier, eta, ista_step, lambd, n, p, s0, x_lb, x_ub):
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


@app.cell
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


@app.cell
def _(
    agent_available,
    config,
    find_compiler,
    make_agent_driver,
    next_s,
    out_dir,
    python_us,
    recorded_timings,
    s,
    verifier,
    x_param,
):
    if not agent_available:
        # Show the recorded row as published. The live python_us measured just
        # above is deliberately NOT reused here: pairing this machine's Python
        # time with another machine's C time would invent a cross-machine ratio.
        _rec = recorded_timings["examples"]["lasso"]
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
            next_state=next_s,
            config=config,
            driver=make_agent_driver(config=config),
            state=[s],
            params=[x_param],
            name="lasso_codegen_speedup",
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
