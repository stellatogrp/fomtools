import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import statistics
    import sys
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Callable

    def add_repo_root_to_path():
        cwd = Path.cwd().resolve()
        here = Path(__file__).resolve().parent if "__file__" in globals() else cwd
        for candidate in (cwd, cwd.parent, cwd.parent.parent, here, here.parent, here.parent.parent):
            if (candidate / "fomtools").is_dir():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                return here
        return here

    app_dir = add_repo_root_to_path()

    import numpy as np

    from fomtools.codegen import CodegenConfig, StubAgentDriver, TerminationSpec, generate_code
    from fomtools.codegen.extract import extract_ir
    from fomtools.codegen.harness import find_compiler, run_bench
    from fomtools.codegen.ir import ActivationKind
    from fomtools.verifier import Verifier

    @dataclass
    class SpeedupProblem:
        name: str
        title: str
        max_iters: int
        verifier: object
        next_state: object
        config: CodegenConfig
        python_solve: Callable[[], object]
        state: list[object] | None = None
        params: list[object] | None = None
        notes: str = ""

    @dataclass
    class SpeedupResult:
        python_us: float
        codegen_us: float
        speedup: float
        out_dir: Path
        solver_path: Path

    def make_verifier():
        return Verifier(
            method="mip",
            solver="highs",
            obbt_solver=None,
            params={"output_flag": False},
        )

    def make_config(max_iters, num_verify_samples=4):
        return CodegenConfig(
            termination=TerminationSpec(max_iters=max_iters),
            num_verify_samples=num_verify_samples,
        )

    def soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def median_us(fn, repeats=40, warmups=3):
        sink = None
        for _ in range(warmups):
            sink = fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            sink = fn()
            times.append((time.perf_counter() - t0) * 1.0e6)
        return statistics.median(times)

    def run_speedup(problem, repeats=40):
        if find_compiler() is None:
            raise RuntimeError("No C compiler found. Install cc/gcc/clang to run codegen timing.")

        python_us = median_us(problem.python_solve, repeats=repeats)
        out_dir = app_dir / "_codegen_speedups" / problem.name
        ir = extract_ir(
            problem.verifier,
            problem.next_state,
            state=problem.state,
            params=problem.params,
            config=problem.config,
            name=problem.name,
        )
        solver_source = emit_generic_solver(ir)
        result = generate_code(
            problem.verifier,
            problem.next_state,
            out_dir,
            config=problem.config,
            driver=StubAgentDriver(solver_source),
            state=problem.state,
            params=problem.params,
            name=problem.name,
        )
        if not result.translate.success:
            raise RuntimeError(result.translate.log[-4000:])

        bench = run_bench(out_dir)
        return SpeedupResult(
            python_us=python_us,
            codegen_us=bench.mean_us,
            speedup=python_us / bench.mean_us,
            out_dir=out_dir,
            solver_path=out_dir / result.target.solver_source_path(),
        )

    def result_markdown(problem, result):
        notes = f"\n\n{problem.notes}" if problem.notes else ""
        return f"""
## {problem.title}

One complete solve runs `{problem.max_iters}` iterations.

| implementation | median / mean per solve |
|---|---:|
| direct NumPy Python | `{result.python_us:.3f} us` |
| generated C codegen | `{result.codegen_us:.3f} us` |

**Speedup:** `{result.speedup:.1f}x`

Generated solver directory: `{result.out_dir}`{notes}
"""

    def emit_generic_solver(ir):
        lines = [
            '#include "fom_solver.h"',
            '#include "fom_data.h"',
            '#include "fom_linsolve.h"',
            '#include "fom_ops.h"',
            '#include "fom_profile.h"',
            "",
            "void fom_set_default_settings(FomSettings *settings) {",
            "    settings->max_iter = FOM_MAX_ITER;",
            "    settings->tol = FOM_TOL;",
            "    settings->check_termination = FOM_CHECK_TERMINATION;",
            "}",
            "",
            "static void reset_state(FomWorkspace *work) {",
            "    fom_int i;",
            "    for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = FOM_STATE_INIT[i];",
            "}",
            "",
            "void fom_setup(FomWorkspace *work, const FomSettings *settings) {",
            "    work->settings = *settings;",
            "    reset_state(work);",
            "    work->solution.iter = 0;",
            "    work->solution.residual = 0.0;",
            "    work->solution.status = FOM_MAX_ITER_REACHED;",
            "}",
            "",
            "void fom_update_params(FomWorkspace *work, const FomParams *params) {",
            "    work->params = *params;",
            "    reset_state(work);",
            "}",
            "",
            "fom_int fom_solve(FomWorkspace *work) {",
            "    fom_int it, i;",
            "    for (it = 0; it < work->settings.max_iter; ++it) {",
        ]

        for step in ir.body:
            dim = f"FOM_{step.name.upper()}_DIM"
            lines += [
                f"        FOM_PROF_BEGIN({step.name});",
                *assign_linear_op(step.linear_op, "work->acc", dim, indent="        "),
            ]
            act = step.activation
            if step.solve is not None:
                lines.append(f"        fom_lin_solve_{step.solve}(work->acc, work->{step.name});")
            elif act.kind == ActivationKind.IDENTITY:
                lines.append(f"        for (i = 0; i < {dim}; ++i) work->{step.name}[i] = work->acc[i];")
            elif act.kind == ActivationKind.SOFT_THRESHOLD:
                lam = f"FOM_{step.name.upper()}_LAMBDA"
                lines += [
                    f"        for (i = 0; i < {dim}; ++i) {{",
                    "            fom_float v = work->acc[i];",
                    f"            if (v > {lam}) work->{step.name}[i] = v - {lam};",
                    f"            else if (v < -{lam}) work->{step.name}[i] = v + {lam};",
                    f"            else work->{step.name}[i] = 0.0;",
                    "        }",
                ]
            elif act.kind == ActivationKind.RELU:
                lines += [
                    f"        for (i = 0; i < {dim}; ++i) {{",
                    "            fom_float v = work->acc[i];",
                    f"            work->{step.name}[i] = v > 0.0 ? v : 0.0;",
                    "        }",
                ]
            elif act.kind == ActivationKind.SATURATED_LINEAR:
                lo = f"FOM_MAT_{act.lower}"
                hi = f"FOM_MAT_{act.upper}"
                lines += [
                    f"        for (i = 0; i < {dim}; ++i) {{",
                    "            fom_float v = work->acc[i];",
                    f"            if (v < {lo}[i]) v = {lo}[i];",
                    f"            if (v > {hi}[i]) v = {hi}[i];",
                    f"            work->{step.name}[i] = v;",
                    "        }",
                ]
            else:
                raise ValueError(f"unsupported activation {act.kind}")
            lines.append(f"        FOM_PROF_END({step.name});")
            lines.append("")

        lines += [
            "        FOM_PROF_BEGIN(state_update);",
            *assign_linear_op(ir.state_update, "work->new_state", "FOM_STATE_DIM", indent="        "),
            "        FOM_PROF_END(state_update);",
            "",
            "        fom_float res = 0.0;",
            "        fom_int check = work->settings.check_termination;",
            "        fom_int do_check = FOM_USE_TOL && check > 0 && ((it + 1) % check == 0);",
            "        for (i = 0; i < FOM_STATE_DIM; ++i) {",
            "            fom_float d = work->new_state[i] - work->state[i];",
            "            if (d < 0.0) d = -d;",
            "            if (d > res) res = d;",
            "            work->state[i] = work->new_state[i];",
            "        }",
            "        work->solution.iter = it + 1;",
            "        work->solution.residual = res;",
            "        if (do_check && res <= work->settings.tol) {",
            "            for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];",
            "            work->solution.status = FOM_SOLVED;",
            "            return FOM_SOLVED;",
            "        }",
            "    }",
            "    for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];",
            "    work->solution.status = FOM_MAX_ITER_REACHED;",
            "    return work->solution.status;",
            "}",
            "",
            "void fom_cleanup(FomWorkspace *work) {",
            "    (void)work;",
            "}",
            "",
        ]
        return "\n".join(lines)

    def assign_linear_op(lin, dest, dim_macro, indent):
        lines = []
        if lin.const is None:
            lines.append(f"{indent}for (i = 0; i < {dim_macro}; ++i) {dest}[i] = 0.0;")
        else:
            lines.append(f"{indent}for (i = 0; i < {dim_macro}; ++i) {dest}[i] = FOM_MAT_{lin.const}[i];")
        for term in lin.terms:
            lines.append(f"{indent}fom_apply_{term.matrix}({source_ptr(term.source)}, {dest});")
        return lines

    def source_ptr(source):
        if source == "state":
            return "work->state"
        if source.startswith("param:"):
            return f"work->params.{source.split(':', 1)[1]}"
        if source.startswith("tmp:"):
            return f"work->{source.split(':', 1)[1]}"
        raise ValueError(f"unknown source {source!r}")

    def build_lasso_problem(max_iters=200):
        rng = np.random.RandomState(0)
        p, n, lambd = 3, 4, 1e-1
        d_value = rng.normal(size=(p, n))
        d_mask = rng.binomial(n=1, p=0.2, size=(p, n))
        d_mat = d_value * d_mask
        z_star = rng.normal(size=(n, 200)) * rng.binomial(n=1, p=0.1, size=(n, 200))
        x_cloud = d_mat @ z_star + rng.normal(scale=1e-2, size=(p, 200))
        x_lb, x_ub = np.min(x_cloud, axis=1), np.max(x_cloud, axis=1)
        l_const = float(np.linalg.eigvalsh(d_mat.T @ d_mat)[-1])
        eta = 1.0 / l_const
        x_value = 0.5 * (x_lb + x_ub)
        s0 = np.zeros(n)

        verifier = make_verifier()
        x = verifier.add_param(p, lb=x_lb, ub=x_ub)
        s = verifier.add_iterate(n, lb=s0, ub=s0)
        next_s = verifier.step.soft_threshold(
            s - eta * d_mat.T @ (d_mat @ s - x), lambd=lambd * eta
        )

        def python_solve():
            s_py = s0.copy()
            for _ in range(max_iters):
                s_py = soft_threshold(s_py - eta * d_mat.T @ (d_mat @ s_py - x_value), lambd * eta)
            return s_py

        return SpeedupProblem(
            name="lasso_codegen_speedup",
            title="Lasso ISTA",
            max_iters=max_iters,
            verifier=verifier,
            next_state=next_s,
            state=[s],
            params=[x],
            config=make_config(max_iters),
            python_solve=python_solve,
            notes="Python uses the midpoint of the input box; C benchmark cycles generated verify samples. The per-iteration work is identical.",
        )

    return build_lasso_problem, result_markdown, run_speedup


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lasso ISTA: Python vs Codegen

    This benchmarks the direct NumPy ISTA loop from `comparison_lasso.py`
    against a verified generated C solver for the same iteration map.
    """)
    return


@app.cell
def _(build_lasso_problem):
    problem = build_lasso_problem(max_iters=200)
    return (problem,)


@app.cell
def _(problem, run_speedup):
    result = run_speedup(problem, repeats=40)
    return (result,)


@app.cell
def _(mo, problem, result, result_markdown):
    mo.md(result_markdown(problem, result))
    return


if __name__ == "__main__":
    app.run()
