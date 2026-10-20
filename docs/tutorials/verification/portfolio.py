import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import contextlib
    import io
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.sparse import csr_matrix

    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "fomtools").is_dir():
            repo_root = candidate
            break
    else:
        repo_root = start

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from docs.tutorials.helper_functions.portfolio import (
        admm_step,
        make_portfolio_data,
    )

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    solver_params = {
        "OutputFlag": 0,
        "LogToConsole": 0,
    }

    # Gurobi prints a license banner through sys.stdout the first time its
    # default environment starts, before any OutputFlag can apply. Start the
    # environment here, in this hidden cell, with stdout swallowed so the
    # banner never lands in a visible cell.
    try:
        import gurobipy as _gp
    except ImportError:
        pass
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            _gp.Model()

    return (
        admm_step,
        csr_matrix,
        make_portfolio_data,
        mo,
        np,
        plt,
        solver_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem setup: Portfolio optimization

    In this tutorial, we verify an ADMM method for a mean-variance portfolio optimization problem. Given a covariance matrix $\Sigma \succeq 0$, expected return vector $\mu$, and position limit $u_{\max}$, we solve

    \begin{array}{ll}
        \text{minimize} & (1/2)z^T\Sigma z - \mu^T z \\
        \text{subject to} & 0 \le z_i \le (u_{\max})_i,\quad i=1,\ldots,n, \\
        & \mathbf{1}^T z = 1.
    \end{array}

    The decision variable $z \in \mathbf{R}^n$ is the portfolio allocation (`x` in the code). The verification problem searches over the initial dual iterate of ADMM while keeping the problem data fixed.
    """)
    return


@app.cell
def _(make_portfolio_data, np):
    n = 4
    Sigma, mu, A_mat, l_val, u_val = make_portfolio_data(
        n=n, seed=0, position_limit=0.6
    )

    x0_val = np.zeros(n)
    v0_lb = np.zeros(n + 1)
    v0_ub = np.ones(n + 1)
    return A_mat, Sigma, l_val, mu, n, u_val, v0_lb, v0_ub, x0_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: OSQP-style ADMM

    We write the constraints as $l \le Az \le u$, with
    $$
        A = \begin{bmatrix} I \\ \mathbf{1}^T \end{bmatrix},\quad
        l = \begin{bmatrix} 0 \\ 1 \end{bmatrix},\quad
        u = \begin{bmatrix} u_{\max} \\1 \end{bmatrix}.
    $$

    The ADMM recurrence is

    \begin{array}{rcl}
        w^{k+1} &=& \Pi_{[l,u]}(v^k), \\
        z^{k+1} &=& (\Sigma + \sigma I + \rho A^TA)^{-1}
            \left(\sigma z^k + \mu + \rho A^T(2w^{k+1}-v^k)\right), \\
        v^{k+1} &=& v^k + Az^{k+1} - w^{k+1},
    \end{array}

    which is a fixed-point iteration in the pair $(z^k, v^k)$.

    We measure performance by the infinity norm of the combined fixed-point residual $r^k = (z^{k+1}-z^k,\; v^{k+1}-v^k)$.
    As in [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330), the infinity norm keeps the verification problem a mixed-integer *linear* program.
    """)
    return


@app.cell
def _(A_mat, Sigma, csr_matrix, n, np):
    sigma = 1e-4
    rho = 1.0
    lhs_mat = csr_matrix(Sigma + sigma * np.eye(n) + rho * A_mat.T @ A_mat)
    return lhs_mat, rho, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every verification mode below (simulator and MIP-VP) drives this
    *same* `admm_step` recurrence and residual; only the
    `Verifier` backend and whether $v^0$ is a sampled instance or a bounded
    search variable change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    We compare simulator statistics and exact MIP verification for the portfolio ADMM recurrence, up to $K$ iterations.
    """)
    return


@app.cell
def _():
    K = 45  # Maximum iteration number
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    We first simulate ADMM from randomly sampled initial dual iterates $v^0 \in [0,1]^{n+1}$. The samples are drawn once below and reused by every simulation cell. Running `_VP.solve()` returns a residual history for each sampled initialization.
    """)
    return


@app.cell
def _(n, np, v0_lb, v0_ub):
    n_samples = 10000
    _rng = np.random.default_rng(0)
    v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n + 1))
    return (v0_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulation is run through the following function.
    """)
    return


@app.cell
def _(A_mat, admm_step, l_val, lhs_mat, mu, n, np, rho, sigma, u_val, x0_val):
    from fomtools.verifier import Verifier

    def admm_simulation(v0_samples, K, ord=np.inf):
        """Simulate ADMM on sampled dual iterates.

        Builds a simulator-mode :class:`Verifier`, registers the initial dual
        iterate ``v0`` as a sampled parameter (fixing $x^0 = 0$), rolls out the
        recurrence with ``admm_step``, and simulates via ``VP.solve()``.
        ``v0_samples`` has shape ``(n_samples, n+1)`` (a single-row array
        replays one instance), and the returned ``residuals`` has shape
        ``(n_samples, K)``. Summary statistics (``mean``, ``min``, ``max``, ...)
        are available on the returned instance ``VP`` via ``VP[...]``.
        """
        VP = Verifier(method="simulator", obbt_solver=None)
        x = x0_val
        v = VP.add_param(n + 1, samples=v0_samples)

        lhs_mat_dense = lhs_mat.toarray()
        for _ in range(K):
            x, v, residual = admm_step(
                VP, A_mat, l_val, u_val, mu, rho, sigma, x, v, lhs_mat_dense
            )
            VP.set_objective_norm(residual, ord)

        residuals = VP.solve()
        return VP, residuals

    return Verifier, admm_simulation


@app.cell
def _(K, admm_simulation, np, v0_samples):
    _VP, _residuals = admm_simulation(v0_samples, K, ord=np.inf)

    print(
        f"Summary statistics (at K={K}):\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
    )

    simulation_residuals_admm = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_admm = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return simulation_residual_quantiles_admm, simulation_residuals_admm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm verification through verification problem (VP)

    We now obtain worst-case guarantees by solving the verification problem
    with a MIP formulation, which gives the exact worst case. The following
    function rolls out the same `admm_step` recurrence over the horizon $K$
    and maximizes $\|r^k\|_\infty$ over the bounded initial dual iterate $v^0$
    at each iteration.
    """)
    return


@app.cell
def _(
    A_mat,
    Verifier,
    admm_step,
    l_val,
    lhs_mat,
    mu,
    n,
    np,
    rho,
    sigma,
    u_val,
    v0_lb,
    v0_ub,
    x0_val,
):
    def admm_verification(
        K, method, ord=np.inf, obbt_params={"output_flag": False}, **verifier_kwargs
    ):
        """Solve the portfolio ADMM verification problem for one backend.

        Builds a ``Verifier`` in ``method`` mode, registers the initial dual
        iterate box ``[v0_lb, v0_ub]`` (fixing $x^0 = 0$), rolls out
        ``admm_step`` over the horizon ``K``, and returns ``(VP, residuals)``
        with one worst-case residual per iteration. Extra ``verifier_kwargs``
        (``solver``, ``obbt_solver``, ``params``, ...) are forwarded to the
        ``Verifier`` constructor.

        ``ord`` defaults to ``inf``, which keeps the MIP a MILP. A 2-norm
        objective is squared, so its value is square-rooted on return.

        ``obbt_params`` defaults to silencing the HiGHS OBBT solver: ``Verifier``
        forwards only the OBBT backend's own verbosity key from ``params``, so
        Gurobi's ``OutputFlag`` does not reach it.
        """
        VP = Verifier(method=method, obbt_params=obbt_params, **verifier_kwargs)
        x = VP.iterate_trajectory(K + 1, collapse_expr="auto")
        v = VP.iterate_trajectory(K + 1, collapse_expr="auto")
        x[0] = VP.add_iterate(n, lb=x0_val, ub=x0_val)
        v[0] = VP.add_iterate(n + 1, lb=v0_lb, ub=v0_ub)

        residuals = []
        for k in range(1, K + 1):
            x, v, residual = admm_step(
                VP, A_mat, l_val, u_val, mu, rho, sigma, x, v, lhs_mat, k=k
            )
            VP.set_objective_norm(residual, ord)
            val = VP.solve()
            residuals.append(np.sqrt(val) if ord == 2 else val)
        return VP, residuals

    return (admm_verification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MIP-VP

    We solve the exact MIP verification problem. The MIP maximizes $\|r^k\|_\infty$ over the bounded initial dual iterate $v^0$ for each horizon.
    """)
    return


@app.cell
def _(K, admm_verification, np, solver_params):
    _, all_residuals = admm_verification(
        K, "mip", ord=np.inf, solver="gurobi", obbt_solver="highs", params=solver_params,
    )
    return (all_residuals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical Result

    Now we plot the verification results against the in-sample simulation
    statistics, including the sample worst-case instance. Every curve reports
    $\|r^k\|_\infty$, but they answer different questions:

    - **sample worst-case** — a *lower* bound on the true worst case, from $10^4$
      draws uniform over the same box $v^0 \in [0,1]^{n+1}$ the verifiers search.
    - **MIP-VP** — the *exact* worst case over that box, for this instance.

    So the expected ordering at every $K$ is
    $\text{sample worst-case} \le \text{MIP-VP}$.

    **There is no PEPit curve here.** Earlier versions of this notebook plotted
    one, but it was not a bound on this algorithm: PEPit's `admm_residuals`
    template is textbook Douglas–Rachford with unit proximal steps, in which
    $\sigma$, $\rho$ and $A$ do not appear at all, whereas the recurrence above
    is OSQP-form ADMM with $\sigma = 10^{-4}$, $\rho = 1$ and
    $A = [I;\ \mathbf{1}^T]$ — a different operator with a different contraction
    factor. Its radius was additionally calibrated so that the PEPit value
    matched the MIP value at $K=1$, which is why the curve used to fall *below*
    the exact MIP worst case at later $K$. Restoring it would require a PEP
    template for OSQP-ADMM with $A \ne I$; the `_osqp_admm_residual` template
    used by the optimal-control tutorial models $\sigma$ and $\rho$ correctly but
    assumes $A = I$.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    plt,
    simulation_residual_quantiles_admm,
    simulation_residuals_admm,
):
    plt.figure()
    _ks = range(1, K + 1)
    plt.plot(_ks, all_residuals, label="ADMM (MIP worst-case)")
    _avg_line, = plt.plot(_ks, simulation_residuals_admm, "--", label="ADMM (in-sample avg)")
    plt.fill_between(
        _ks,
        simulation_residual_quantiles_admm[2],
        simulation_residual_quantiles_admm[0],
        alpha=.18,
        label="ADMM sample 10%-90% band",
    )
    plt.plot(_ks, simulation_residual_quantiles_admm[3], ":", color=_avg_line.get_color(), label="ADMM sample worst-case")
    plt.yscale("log")
    plt.ylabel(r"Worst-case fixed-point residual $\|r^k\|_\infty$")
    plt.xlabel(r"$K$")
    plt.suptitle("portfolio optimization ADMM example")
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
