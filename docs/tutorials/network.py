import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import os
    import statistics
    import sys
    import textwrap
    import time
    from pathlib import Path
    from scipy.optimize import linprog

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
        linprog,
        make_agent_driver,
        mo,
        np,
        recorded_timings,
        statistics,
        textwrap,
        time,
    )


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
# Network Flow: Python vs Codegen

In this tutorial we benchmark a small capacitated network-flow problem.
There are supply nodes, demand nodes, edge capacities, and linear edge costs.

The solver uses a primal-dual projected update: each iteration clips the flow to the capacity box, projects one dual variable to be nonnegative, and updates the demand-side dual variable.
We compare a direct NumPy implementation with the verified generated C solver.
""")
    )
    return


@app.cell(hide_code=True)
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""
We generate a complete bipartite graph from supply nodes to demand nodes.
The starting flow is computed once by solving the linear program at the lower-demand endpoint.
Both implementations then run the same fixed iteration map.
""")
    )
    return


@app.cell
def _(examples_dir, linprog, np):
    rng = np.random.RandomState(2)
    n_supply, n_demand = 3, 2
    edges = [(i, j) for i in range(n_supply) for j in range(n_demand)]
    n_edges = len(edges)

    A_supply = np.zeros((n_supply, n_edges))
    A_demand = np.zeros((n_demand, n_edges))
    for edge_index, (i, j) in enumerate(edges):
        A_supply[i, edge_index] = 1.0
        A_demand[j, edge_index] = -1.0

    supply_capacity = 10.0 * np.ones(n_supply)
    edge_capacity = 5.0 * np.ones(n_edges)
    edge_cost = rng.uniform(5.0, 10.0, size=n_edges)
    x_lb = -7.0 * np.ones(n_demand)
    x_ub = -6.0 * np.ones(n_demand)
    x_mid = 0.5 * (x_lb + x_ub)

    initial_flow = linprog(
        c=edge_cost,
        A_ub=A_supply,
        b_ub=supply_capacity,
        A_eq=A_demand,
        b_eq=x_lb,
        bounds=[(0.0, float(capacity)) for capacity in edge_capacity],
        method="highs",
    )
    if not initial_flow.success:
        raise RuntimeError(initial_flow.message)

    z0 = initial_flow.x
    v0 = np.zeros(n_supply)
    w0 = np.zeros(n_demand)
    L = float(np.linalg.norm(np.hstack([A_supply.T, A_demand.T]), 2))
    eta = 0.5 / L
    K = 200
    out_dir = examples_dir / "_codegen_speedups" / "network_codegen_speedup"
    return (
        A_demand,
        A_supply,
        K,
        edge_capacity,
        edge_cost,
        eta,
        n_demand,
        n_edges,
        n_supply,
        out_dir,
        supply_capacity,
        v0,
        w0,
        x_lb,
        x_mid,
        x_ub,
        z0,
    )


@app.cell
def _(
    A_demand,
    A_supply,
    CodegenConfig,
    K,
    TerminationSpec,
    Verifier,
    edge_capacity,
    edge_cost,
    eta,
    n_demand,
    n_edges,
    n_supply,
    np,
    supply_capacity,
    v0,
    w0,
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

    x_param = verifier.add_param(n_demand, lb=x_lb, ub=x_ub)
    z = verifier.add_iterate(n_edges, lb=z0, ub=z0)
    v = verifier.add_iterate(n_supply, lb=v0, ub=v0)
    w = verifier.add_iterate(n_demand, lb=w0, ub=w0)
    z_next = verifier.step.satlin(
        z - eta * (edge_cost + A_supply.T @ v - A_demand.T @ w),
        l=np.zeros(n_edges),
        u=edge_capacity,
    )
    v_next = verifier.step.relu(v + eta * (-supply_capacity + A_supply @ (2 * z_next - z)))
    w_next = w + eta * (x_param - A_demand @ (2 * z_next - z))

    config = CodegenConfig(
        termination=TerminationSpec(max_iters=K),
        num_verify_samples=4,
    )
    next_state = [z_next, v_next, w_next]
    s_state = [z, v, w]
    params = [x_param]
    return config, next_state, params, s_state, verifier


@app.cell
def _(
    A_demand,
    A_supply,
    K,
    edge_capacity,
    edge_cost,
    eta,
    np,
    statistics,
    supply_capacity,
    time,
    v0,
    w0,
    x_mid,
    z0,
):
    def network_solve():
        z_py = z0.copy()
        v_py = v0.copy()
        w_py = w0.copy()
        for _ in range(K):
            z_next_py = np.minimum(
                np.maximum(z_py - eta * (edge_cost + A_supply.T @ v_py - A_demand.T @ w_py), 0.0),
                edge_capacity,
            )
            v_next_py = np.maximum(v_py + eta * (-supply_capacity + A_supply @ (2 * z_next_py - z_py)), 0.0)
            w_next_py = w_py + eta * (x_mid - A_demand @ (2 * z_next_py - z_py))
            z_py, v_py, w_py = z_next_py, v_next_py, w_next_py
        return np.concatenate([z_py, v_py, w_py])

    for _ in range(3):
        network_solve()

    python_times = []
    for _ in range(40):
        t0 = time.perf_counter()
        network_solve()
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
    recorded_timings,
    s_state,
    verifier,
):
    if not agent_available:
        # Show the recorded row as published. The live python_us measured just
        # above is deliberately NOT reused here: pairing this machine's Python
        # time with another machine's C time would invent a cross-machine ratio.
        _rec = recorded_timings["examples"]["network"]
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
        result = verifier.codegen_rolled(
            out_dir,
            next_state=next_state,
            config=config,
            driver=make_agent_driver(config=config),
            state=s_state,
            params=params,
            name="network_codegen_speedup",
            code_optimization=True,
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
| direct NumPy Python | `{table_python_us:.3f} us` |
| generated C after code generation | `{codegen_us:.3f} us` |
| generated C after optimization iteration | `{iteration_us:.3f} us` |

The speedup from code generation is `{codegen_speedup:.1f}x`, while the speedup from the optimization iteration is `{iteration_speedup:.2f}x`, totaling a speedup of `{total_speedup:.1f}x`.

The optimization iteration uses a live coding agent to edit `src/fom_solver.c`; the edited solver was `{optimization_status}` after verification and timing.
{_footer}"""
    )
    summary
    return


if __name__ == "__main__":
    app.run()
