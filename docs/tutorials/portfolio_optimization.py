import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner
    from scipy.sparse import csr_matrix

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    solver_params = {
        "OutputFlag": 0,
        "LogToConsole": 0,
    }
    return PEPitRunner, Verifier, csr_matrix, np, plt, solver_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem setup: Portfolio optimization

    In this tutorial, we verify an ADMM method for a mean-variance portfolio optimization problem. Given a covariance matrix $\Sigma \succeq 0$, expected return vector $\mu$, and position limit $u_{\max}$, we solve

    \begin{array}{ll}
        \text{minimize} & (1/2)x^T\Sigma x - \mu^T x \\
        \text{subject to} & 0 \le x_i \le (u_{\max})_i,\quad i=1,\ldots,n, \\
        & \mathbf{1}^T x = 1.
    \end{array}

    The decision variable $x \in \mathbf{R}^n$ is the portfolio allocation. The verification problem searches over the initial dual iterate of ADMM while keeping the problem data fixed.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    n = 4

    _F = np.random.normal(size=(n, n))
    Sigma = (_F @ _F.T) / n + 0.1 * np.eye(n)
    mu = np.random.uniform(0.05, 0.15, size=n)
    u_max = 0.6 * np.ones(n)

    A_mat = np.vstack([np.eye(n), np.ones((1, n))])
    l_val = np.hstack([np.zeros(n), [1.0]])
    u_val = np.hstack([u_max, [1.0]])

    x0_val = np.zeros(n)
    v0_lb = np.zeros(n + 1)
    v0_ub = np.ones(n + 1)
    return A_mat, Sigma, l_val, mu, n, u_max, u_val, v0_lb, v0_ub, x0_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: OSQP-style ADMM

    We write the constraints as $l \le Ax \le u$, with
    \[
        A = \begin{bmatrix} I \\ \mathbf{1}^T \end{bmatrix},\quad
        l = (0,\ldots,0,1),\quad
        u = (u_{\max},1).
    \]

    The ADMM recurrence is

    \begin{array}{rcl}
        w^{k+1} &=& \Pi_{[l,u]}(v^k), \\
        x^{k+1} &=& (\Sigma + \sigma I + \rho A^TA)^{-1}
            \left(\sigma x^k + \mu + \rho A^T(2w^{k+1}-v^k)\right), \\
        v^{k+1} &=& v^k + Ax^{k+1} - w^{k+1}.
    \end{array}

    We measure performance by the Euclidean norm of the combined fixed-point residual $(x^{k+1}-x^k,\; v^{k+1}-v^k)$.
    """)
    return


@app.cell
def _(A_mat, Sigma, csr_matrix, n, np):
    sigma = 1e-4
    rho = 1.0
    K = 3
    lhs_mat = csr_matrix(Sigma + sigma * np.eye(n) + rho * A_mat.T @ A_mat)
    return K, lhs_mat, rho, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    We compare simulator statistics, exact MIP verification, SDP-relaxed verification, and a PEPit bound for the portfolio ADMM recurrence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    We first simulate ADMM from randomly sampled initial dual iterates $v^0 \in [0,1]^{n+1}$. Running `_VP.solve()` returns a residual history for each sampled initialization.
    """)
    return


@app.cell
def _(
    A_mat,
    K,
    Verifier,
    lhs_mat,
    l_val,
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
    n_samples = 10000
    _rng = np.random.default_rng(0)
    _v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n + 1))

    _VP = Verifier(method="simulator", obbt_solver=None)
    _x = [x0_val]
    _v = [_VP.add_param(n + 1, samples=_v0_samples)]
    _v[0].lb, _v[0].ub = v0_lb, v0_ub
    _w = [None]
    _lhs_mat_sim = lhs_mat.toarray()

    for _k in range(K):
        _w.append(_VP.step.satlin(_v[_k], l=l_val, u=u_val))
        _rhs = sigma * _x[_k] + mu + rho * A_mat.T @ (2 * _w[_k + 1] - _v[_k])
        _x.append(_VP.step.implicit_linear(_rhs, output_mat=_lhs_mat_sim))
        _v.append(_v[_k] + A_mat @ _x[_k + 1] - _w[_k + 1])
        _VP.set_objective_norm([_x[_k + 1] - _x[_k], _v[_k + 1] - _v[_k]], ord=2)

    _residuals = _VP.solve()

    print(
        f"Summary statistics:\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"std: {_VP['std']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
        f"empirical_risk: {_VP['empirical_risk']:.4f}\n"
        f"risk_bound: {_VP['risk_bound']:.4f}\n"
        f"n_samples: {_VP['n_samples']}\n"
        f"n_iteration: {_VP['n_iterations']}\n"
        f"runtime: {_VP['runtime']:.4f}\n"
        f"time_per_sample: {_VP['time_per_sample']:.4f}\n"
        f"throughput: {_VP['throughput']:.4f}\n"
        f"residuals: {_residuals}"
    )

    simulation_residuals_admm = np.mean(_residuals, axis=0)
    simulation_residual_quantiles_admm = np.quantile(_residuals, [0.90, 0.50, 0.10], axis=0)
    return n_samples, simulation_residual_quantiles_admm, simulation_residuals_admm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MIP-VP

    We now solve the exact MIP verification problem. The MIP maximizes the ADMM fixed-point residual over the bounded initial dual iterate $v^0$ for each horizon.
    """)
    return


@app.cell
def _(
    A_mat,
    K,
    Verifier,
    lhs_mat,
    l_val,
    mu,
    n,
    rho,
    sigma,
    solver_params,
    u_val,
    v0_lb,
    v0_ub,
    x0_val,
):
    _VP = Verifier(method="mip", solver="gurobi", obbt_solver="highs", params=solver_params)

    _x = [x0_val]
    _v = [_VP.add_iterate(n + 1, lb=v0_lb, ub=v0_ub)]
    _w = [None]

    all_residuals = []
    worst_case_v0_admm = []
    for _k in range(K):
        _w.append(_VP.step.satlin(_v[_k], l=l_val, u=u_val))
        _rhs = sigma * _x[_k] + mu + rho * A_mat.T @ (2 * _w[_k + 1] - _v[_k])
        _x.append(_VP.step.implicit_linear(_rhs, output_mat=lhs_mat))
        _v.append(_v[_k] + A_mat @ _x[_k + 1] - _w[_k + 1])
        _VP.set_objective_norm([_x[_k + 1] - _x[_k], _v[_k + 1] - _v[_k]], ord=2)
        all_residuals.append(_VP.solve())
        worst_case_v0_admm.append(_VP.get_solution_value(_v[0]))
    return all_residuals, worst_case_v0_admm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We replay the horizon-specific worst-case initial dual iterates in simulator mode to check the explicit residual trajectory produced by the MIP optimizer.
    """)
    return


@app.cell
def _(
    A_mat,
    K,
    Verifier,
    lhs_mat,
    l_val,
    mu,
    n,
    np,
    rho,
    sigma,
    u_val,
    worst_case_v0_admm,
    x0_val,
):
    worst_case_residuals_admm = []
    _lhs_mat_sim = lhs_mat.toarray()
    for _k in range(1, K + 1):
        _VP = Verifier(method="simulator", obbt_solver=None)
        _v0_k = np.asarray(worst_case_v0_admm[_k - 1])

        _x = [x0_val]
        _v = [_VP.add_param(ndim=n + 1, samples=[_v0_k])]
        _v[0].lb = _v[0].ub = _v0_k
        _w = [None]

        for _j in range(_k):
            _w.append(_VP.step.satlin(_v[_j], l=l_val, u=u_val))
            _rhs = sigma * _x[_j] + mu + rho * A_mat.T @ (2 * _w[_j + 1] - _v[_j])
            _x.append(_VP.step.implicit_linear(_rhs, output_mat=_lhs_mat_sim))
            _v.append(_v[_j] + A_mat @ _x[_j + 1] - _w[_j + 1])
            _VP.set_objective_norm([_x[_j + 1] - _x[_j], _v[_j + 1] - _v[_j]], ord=2)

        worst_case_residuals_admm.append(_VP.solve()[0][-1])

    worst_case_residuals_admm = np.array(worst_case_residuals_admm)
    return (worst_case_residuals_admm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    We compare the MIP result to a PEPit ADMM residual bound for the smooth strongly-convex quadratic term and the indicator function of the portfolio constraint set. As in the comparison script, the initial radius is calibrated so the PEPit value matches the MIP value at $K=1$.
    """)
    return


@app.cell
def _(K, PEPitRunner, Sigma, all_residuals, np):
    _Sigma_eigs = np.linalg.eigvalsh(Sigma)
    _mu_f = float(_Sigma_eigs.min())
    _L_f = float(_Sigma_eigs.max())
    all_residuals_PEP, pepit_radius = PEPitRunner().admm_residuals(
        K=K,
        mu=_mu_f,
        L=_L_f,
        R=None,
        calibrate_to=all_residuals[0],
    )
    return all_residuals_PEP, pepit_radius


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SDP-VP

    We also solve the SDP-relaxed verification problem, which gives a convex upper bound on the same residual metric.
    """)
    return


@app.cell
def _(A_mat, K, Verifier, lhs_mat, l_val, mu, n, rho, sigma, u_val, v0_lb, v0_ub, x0_val):
    _VP = Verifier(method="sdp", solver="clarabel", obbt_solver="highs")

    _x = [x0_val]
    _v = [_VP.add_iterate(n + 1, lb=v0_lb, ub=v0_ub)]
    _w = [None]

    all_residuals_sdp = []
    for _k in range(K):
        _w.append(_VP.step.satlin(_v[_k], l=l_val, u=u_val))
        _rhs = sigma * _x[_k] + mu + rho * A_mat.T @ (2 * _w[_k + 1] - _v[_k])
        _x.append(_VP.step.implicit_linear(_rhs, output_mat=lhs_mat))
        _v.append(_v[_k] + A_mat @ _x[_k + 1] - _w[_k + 1])
        _VP.set_objective_norm([_x[_k + 1] - _x[_k], _v[_k + 1] - _v[_k]], ord=2)
        all_residuals_sdp.append(_VP.solve())
    return (all_residuals_sdp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we plot the verification results against the PEPit bound, the simulator statistics, and the horizon-specific worst-case instances extracted from MIP.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    all_residuals_sdp,
    plt,
    simulation_residual_quantiles_admm,
    simulation_residuals_admm,
    worst_case_residuals_admm,
):
    plt.figure()
    _ks = range(1, K + 1)
    plt.plot(_ks, all_residuals_PEP, label="ADMM (PEPit)")
    plt.plot(_ks, all_residuals, label="ADMM (MIP worst-case)")
    plt.plot(_ks, all_residuals_sdp, label="ADMM (SDP bound)")
    plt.plot(_ks, simulation_residuals_admm, "--", label="ADMM (in-sample avg)")
    plt.fill_between(
        _ks,
        simulation_residual_quantiles_admm[2],
        simulation_residual_quantiles_admm[0],
        alpha=.18,
        label="ADMM simulation 10%-90% band",
    )
    plt.plot(_ks, worst_case_residuals_admm, "-.", color="k", linewidth=.7, label="worst-case ADMM instance")
    plt.yscale("log")
    plt.ylabel("Worst-case fixed-point residual")
    plt.xlabel(r"$K$")
    plt.suptitle("portfolio optimization ADMM example")
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
