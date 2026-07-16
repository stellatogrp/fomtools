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
# Nonnegative Least Squares: Python vs Codegen

In this tutorial we benchmark projected gradient descent for the nonnegative least-squares problem:

\[
\begin{array}{ll}
    \text{minimize} & (1/2)\lVert Az - x \rVert_2^2 \\
    \text{subject to} & z \ge 0.
\end{array}
\]

The update is

\[
z^{k+1} = \max\{z^k - \eta A^\top(Az^k - x), 0\}.
\]

We compare a direct NumPy implementation with the verified generated C solver.
The generated C is timed after code generation, then timed again after the C-editing optimization pass.
""")
    )
    return


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
We generate a small dense matrix \(A\), put the input vector \(x\) in a box, and start from a fixed positive vector.
Both implementations run the same update rule for the same number of iterations.
""")
    )
    return


@app.cell
def _(examples_dir, np):
    rng = np.random.RandomState(0)
    m, n = 5, 3
    A = rng.normal(size=(m, n))
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
def _(A, CodegenConfig, K, TerminationSpec, Verifier, eta, m, n, x_lb, x_ub, z0):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    x_param = verifier.add_param(m, lb=x_lb, ub=x_ub)
    z = verifier.add_iterate(n, lb=z0, ub=z0)
    z_next = verifier.step.relu(z - eta * A.T @ (A @ z - x_param))

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    next_state = z_next
    s_state = [z]
    params = [x_param]
    return config, next_state, params, s_state, verifier


@app.cell
def _(A, K, eta, np, statistics, time, x_mid, z0):
    def nnls_solve():
        z_py = z0.copy()
        for _ in range(K):
            z_py = np.maximum(z_py - eta * A.T @ (A @ z_py - x_mid), 0.0)
        return z_py

    for _ in range(3):
        nnls_solve()

    python_times = []
    for _ in range(40):
        t0 = time.perf_counter()
        nnls_solve()
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
    params,
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
            params=params,
            name="nnls_codegen_speedup",
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
