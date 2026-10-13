import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import os

    import cvxpy as cp
    import numpy as np

    import fomtools.cvxpygen as fcpg
    from fomtools.codegen import CodegenConfig, TerminationSpec

    # Generating a verified embedded solver needs a real coding-agent backend.
    # When it isn't configured (e.g. a headless docs build), the codegen and
    # solve cells below skip with a message instead of raising.
    agent_available = os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1"
    return CodegenConfig, TerminationSpec, agent_available, cp, fcpg, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MPC with CVXPYgen and a generated solver

    In this tutorial we model a small model predictive control (MPC) problem in CVXPY and turn it into an embedded C solver.
    [CVXPYgen](https://github.com/cvxgrp/cvxpygen) generates the code that maps the problem parameter to the data of a quadratic program, and FOMTools generates the first-order method that solves that quadratic program.
    Both end up in one C project with a Python wrapper, so the CVXPY problem can be solved through the generated code with `problem.solve(method="CPG")`.

    Running the whole notebook needs the `cvxpygen` and `codegen` extras (`pip install 'fomtools[cvxpygen,codegen]'`) and `cmake` on your `PATH`, which CVXPYgen uses to build the Python wrapper.
    The solver is written by an LLM coding agent and then verified against a Python reference, so generating it also needs a configured coding agent.
    Set `FOMTOOLS_CODEGEN_AGENT=1` before starting marimo to use the Claude agent.
    Without it, the generation and solve cells below print a message and skip, and the rest of the notebook still runs.
    The solver is written to `mpc_solver/` relative to the working directory, so start marimo (or `python cvxpygen_mpc.py`) from this directory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem data

    The plant is a linear system $s_{t+1} = A s_t + B u_t$ with state $s_t \in \mathbf{R}^{n_s}$ and input $u_t \in \mathbf{R}^{n_u}$, where $n_s = 4$ and $n_u = 2$.
    We draw $A$ and $B$ at random and scale $A$ to spectral radius $0.9$, so the uncontrolled system is stable.
    The horizon is $T = 8$, the stage costs are $Q = I$ and $R = 0.1 I$, and the terminal cost is $Q_T = 10 I$.
    In the code, $A$, $B$, $Q_T$, $n_s$, $n_u$, and $T$ are `Ad`, `Bd`, `QN`, `nx`, `nu`, and `N`.
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
    ## The CVXPY model

    Given an initial state $x \in \mathbf{R}^{n_s}$, the MPC problem is

    \begin{array}{ll}
        \text{minimize} & s_T^T Q_T s_T + \sum_{t=0}^{T-1} \left( s_t^T Q s_t + u_t^T R u_t \right) \\
        \text{subject to} & s_{t+1} = A s_t + B u_t, \quad t = 0, \dots, T-1 \\
        & \|s_t\|_\infty \le 5, \quad \|u_t\|_\infty \le 1, \quad t = 0, \dots, T-1 \\
        & s_0 = x,
    \end{array}

    with variables $s_0, \dots, s_T$ and $u_0, \dots, u_{T-1}$ (`xs` and `us` in the code).
    The initial state $x$ is a CVXPY `Parameter` named `x0`, so CVXPYgen treats it as data that can change between solves.
    The value assigned to it here is only a placeholder; the solve cell below sets a new one.
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
    ## Generating the solver

    A single call to `generate_solver` runs both code generators.
    CVXPYgen first reduces the problem to the quadratic program

    \begin{array}{ll}
        \text{minimize} & (1/2) z^T P z + q^T z \\
        \text{subject to} & l(x) \le M z \le u(x),
    \end{array}

    where $z$ stacks the states, the inputs, and the auxiliary variables CVXPYgen introduces.
    Only the bounds $l(x)$ and $u(x)$ depend on $x$, through the constraint $s_0 = x$, so $P$, $q$, and $M$ are fixed and stored in the generated C code.

    FOMTools then records one iteration of the ADMM method used by OSQP (`fom="admm"`, with OSQP's default parameters) on this quadratic program.
    The ADMM linear system has a constant matrix, so it is factored once at generation time and each iteration only needs triangular solves.
    A coding agent writes the iteration in C, and FOMTools compiles it and compares its iterates with the Python reference for several values of $x$ drawn from the box in `param_bounds`, here $x \in [-1, 1]^{n_s}$.
    If the comparison fails, the agent sees the error and tries again.
    Once it passes, CVXPYgen builds the Python wrapper with CMake.
    The whole step takes a few minutes, most of it spent by the agent.

    The solver runs a fixed $2000$ iterations (`TerminationSpec(max_iters=2000)`).
    The iteration count is a runtime setting, so it can be changed later without regenerating the code.
    The generated project is written to `mpc_solver/`, and the agent's solver loop is in `mpc_solver/c/solver_code/src/fom_solver.c`.
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
    ## Solving through the generated code

    `generate_solver` registers the generated code with the problem as the solve method `"CPG"`.
    We draw a new initial state inside the verified box, solve through the generated C code, and print the first input $u_0$, which is the input an MPC controller applies before re-solving at the next time step.
    As a check, we then solve the same problem with OSQP at tight tolerances.
    After $2000$ iterations the two inputs agree to within $10^{-10}$.
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


if __name__ == "__main__":
    app.run()
