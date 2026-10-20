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

    from docs.tutorials.helper_functions.network import (
        make_network_data,
        pdhg_step,
        solve_pdhg_numpy,
    )
    from docs.tutorials.helper_functions.codegen_benchmarks import run_codegen_benchmark
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
        linprog,
        make_network_data,
        mo,
        np,
        pdhg_step,
        run_codegen_benchmark,
        solve_pdhg_numpy,
        statistics,
        timing_file,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Network Flow: Python vs Codegen

    [Back to the code generation tutorials](./)

    In this tutorial we benchmark a small capacitated network-flow problem.
    There are supply nodes, demand nodes, edge capacities, and linear edge costs.

    The solver uses a primal-dual projected update: each iteration clips the flow to the capacity box, projects one dual variable to be nonnegative, and updates the demand-side dual variable.
    We compare a direct NumPy implementation with the verified generated C solver.
    The problem construction, PDHG step, and NumPy reference solver are shared with the [network verification tutorial](../verification/network.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate a complete bipartite graph from supply nodes to demand nodes.
    The starting flow is computed once by solving the linear program at the lower-demand endpoint.
    Both implementations then run the same fixed iteration map.
    """)
    return


@app.cell
def _(examples_dir, linprog, make_network_data, np):
    n_supply, n_demand = 3, 2
    A_supply, A_demand, supply_capacity, edge_capacity, edge_cost = make_network_data(
        n_supply=n_supply, n_demand=n_demand, data_seed=2, complete=True
    )
    n_edges = A_supply.shape[1]
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


@app.cell(hide_code=True)
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
    pdhg_step,
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
    z_next, v_next, w_next, _ = pdhg_step(
        verifier,
        A_supply,
        A_demand,
        eta,
        z,
        v,
        w,
        x_param,
        edge_capacity,
        edge_cost,
        supply_capacity,
        k=1,
        momentum=False,
    )

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
    solve_pdhg_numpy,
    statistics,
    supply_capacity,
    time,
    v0,
    w0,
    x_mid,
    z0,
):
    def network_solve():
        return solve_pdhg_numpy(
            A_supply,
            A_demand,
            eta,
            edge_capacity,
            edge_cost,
            supply_capacity,
            z0,
            v0,
            w0,
            x_mid,
            max_iters=K,
        )

    for _ in range(3):
        network_solve()

    python_times = []
    for _ in range(40):
        t0 = time.perf_counter()
        network_solve()
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
        example="network",
        name="network_codegen_speedup",
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
