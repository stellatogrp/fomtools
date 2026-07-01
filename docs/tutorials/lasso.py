import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
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

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return (
        CodegenConfig,
        TerminationSpec,
        Verifier,
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
def _(examples_dir, np):
    rng = np.random.RandomState(0)
    p, n, lambd = 3, 4, 1e-1

    D_value = rng.normal(size=(p, n))
    D_mask = rng.binomial(n=1, p=0.2, size=(p, n))
    D = D_value * D_mask

    num_star = 200
    z_star = rng.normal(size=(n, num_star)) * rng.binomial(n=1, p=0.1, size=(n, num_star))
    x_samples = D @ z_star + rng.normal(scale=1e-2, size=(p, num_star))
    x_lb, x_ub = np.min(x_samples, axis=1), np.max(x_samples, axis=1)
    x_mid = 0.5 * (x_lb + x_ub)

    L = float(np.linalg.eigvalsh(D.T @ D)[-1])
    eta = 1.0 / L
    K = 200
    s0 = np.zeros(n)

    out_dir = examples_dir / "_codegen_speedups" / "lasso_codegen_speedup"
    return D, K, eta, lambd, n, out_dir, p, s0, x_lb, x_mid, x_ub


@app.cell
def _(CodegenConfig, D, K, TerminationSpec, Verifier, eta, lambd, n, p, s0, x_lb, x_ub):
    verifier = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False},
    )

    x_param = verifier.add_param(p, lb=x_lb, ub=x_ub)
    s = verifier.add_iterate(n, lb=s0, ub=s0)
    next_s = verifier.step.soft_threshold(
        s - eta * D.T @ (D @ s - x_param),
        lambd=lambd * eta,
    )

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    return config, next_s, s, verifier, x_param


@app.cell
def _(D, K, eta, lambd, np, s0, statistics, time, x_mid):
    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    def python_lasso_solve():
        s_py = s0.copy()
        for _ in range(K):
            s_py = soft_threshold(s_py - eta * D.T @ (D @ s_py - x_mid), lambd * eta)
        return s_py

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
    config,
    find_compiler,
    make_agent_driver,
    next_s,
    out_dir,
    python_us,
    s,
    verifier,
    x_param,
):
    if find_compiler() is None:
        raise RuntimeError("No C compiler found. Install cc/gcc/clang to run codegen timing.")

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
    mo.md(f"""
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
    return


if __name__ == "__main__":
    app.run()
