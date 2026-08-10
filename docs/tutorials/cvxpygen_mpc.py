import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os

    import cvxpy as cp
    import numpy as np

    import fomtools.cvxpygen as fcpg
    from fomtools.codegen import CodegenConfig, TerminationSpec

    # Generating a verified embedded solver needs a real coding-agent backend.
    # When it isn't configured (e.g. a headless docs build), the codegen and
    # solve cells below skip with a message instead of raising.
    agent_available = os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1"
    return CodegenConfig, TerminationSpec, agent_available, cp, fcpg, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CVXPYgen integration example: embedded MPC solver from a CVXPY model.

    Models a small MPC QP in CVXPY (parametric initial state entering the equality
    right-hand sides, parametric linear cost) and generates a single embedded C
    project: CVXPYgen canonicalization + a fomtools ADMM solver.

    Requires `pip install 'fomtools[cvxpygen]'` and a coding-agent driver
    (FOMTOOLS_CODEGEN_AGENT=1 for the Claude agent).

    Run from this directory:  python cvxpygen_mpc.py
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MPC problem data
    """)
    return


@app.cell
def _(np):
    nx, nu, N = 4, 2, 8
    rng = np.random.default_rng(1)
    Ad = rng.normal(size=(nx, nx))
    Ad = 0.9 * Ad / np.max(np.abs(np.linalg.eigvals(Ad)))
    Bd = 0.5 * rng.normal(size=(nx, nu))
    Q, QN, R = np.eye(nx), 10.0 * np.eye(nx), 0.1 * np.eye(nu)
    return Ad, Bd, N, Q, QN, R, nu, nx, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CVXPY model with the initial state as a parameter
    """)
    return


@app.cell
def _(Ad, Bd, N, Q, QN, R, cp, nu, nx, rng):
    x0 = cp.Parameter(nx, name="x0")
    x0.value = rng.uniform(-0.5, 0.5, size=nx)

    xs = cp.Variable((N + 1, nx), name="x")
    us = cp.Variable((N, nu), name="u")
    cost = cp.quad_form(xs[N], QN)
    constraints = [xs[0] == x0]
    for t in range(N):
        cost += cp.quad_form(xs[t], Q) + cp.quad_form(us[t], R)
        constraints += [
            xs[t + 1] == Ad @ xs[t] + Bd @ us[t],
            cp.abs(xs[t]) <= 5.0,
            cp.abs(us[t]) <= 1.0,
            ]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, us, x0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generate the embedded solver
    """)
    return


@app.cell
def _(CodegenConfig, TerminationSpec, agent_available, fcpg, problem, x0):
    if agent_available:
        result = fcpg.generate_solver(
            problem,
            "mpc_solver",  # relative path: cvxpygen imports <code_dir>.cpg_solver
            fom="admm",
            param_bounds={x0: (-1.0, 1.0)},
            config=CodegenConfig(termination=TerminationSpec(max_iters=2000)),
        )
        print(f"generated verified solver in {result.out_dir}")
    else:
        result = None
        print(
            "Skipping solver generation: set FOMTOOLS_CODEGEN_AGENT=1 (and configure a "
            "coding-agent backend) to generate and verify the embedded solver."
        )
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solve through the generated code
    """)
    return


@app.cell
def _(agent_available, cp, nx, problem, result, rng, us, x0):
    if agent_available and result is not None:
        x0.value = rng.uniform(-0.5, 0.5, size=nx)
        problem.solve(method="CPG")
        print("first input:", us.value[0])

        problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)
        print("OSQP check :", us.value[0])
    else:
        print("Skipping solve: no generated CPG solver (see the previous cell).")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
