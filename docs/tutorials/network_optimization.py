import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import numpy as np
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

    from fomtools.proxable_functions import IndBox, IndNonNegative
    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }

    np.random.seed(2)
    return (
        IndBox,
        IndNonNegative,
        PEPitRunner,
        Verifier,
        cp,
        np,
        plt,
        solver_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial, we illustrate the network optimization example in the Section 4.2 of the paper: [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

    Suppose that we are given a set of $n_s$ supply nodes and $n_d$ demand nodes, which constructs a bipartite graph with capacitated directed edges from supply nodes to demand nodes.
    Let $A_s$ and $A_d$ be submatrices corresponding to supply nodes and demand nodes, respectively.
    For each edge $i$, let $c_i$ be the flow capacity, $\mu_i$ be the cost of sending a unit of flow, and $f_i$ be the variable representing the amount of flow sent.
    Finally, let $b_s \ge 0$ be the vector of supply amounts available at each supply node and $x \le 0$ be the vector of requested demand amount for each demand node.

    Consider the following minimum cost network flow problem:

    \begin{array}{ll}
        \text{minimize} & \mu^T f \\
        \text{subject to}
        & A_s f \le b_s \\
        & A_d f = x \\
        & 0 \le f \le c.
    \end{array}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We solve this problem using the standard Chambolle-Pock method, or Primal-dual Hybrid Gradient (PDHG):

    \begin{array}{rcl}
        z^{k+1} &=& \mathbf{prox}_{\mathbb{I}_{[0, c]}} \big( z^k - \eta (\mu + A_s^T v^k - A_d^T w^k) \big) \\
        v^{k+1} &=& \mathbf{prox}_{\mathbb{I}_{\mathbb{R}_+}} \big( v^k + \eta \big( - b_s + A_s (2 z^{k+1} - z^k) \big) \big) \\
        w^{k+1} &=& w^k + \eta \big( x - A_d (2 z^{k+1} - z^k) \big).
    \end{array}

    Here $\mathbf{prox}_{\mathbb{I}_{[0, c]}}$ is projection onto the flow-capacity box and $\mathbf{prox}_{\mathbb{I}_{\mathbb{R}_+}}$ is projection onto the nonnegative orthant.
    We choose $\eta = 0.5 / \| \begin{bmatrix} - A_s^T & A_d^T \end{bmatrix} \|_2$.
    """)
    return


@app.cell
def _(np):
    # Generate bipartite graph with n_s supply nodes and n_d demand nodes,
    # whose edges are added with probability 0.5.
    # n_s = 15
    # n_d = 10
    n_s = 6
    n_d = 3

    # Generate bipartite graph G and its incidence matrix A
    p = 0.5

    import networkx as nx
    G = nx.bipartite.random_graph(n_s, n_d, p, seed=1, directed=False)
    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)
    A = A.todense()
    A[n_s:, :] *= -1

    with open('incidence_matrix.npy', 'wb') as write_file:
        np.save(write_file, A)

    A = np.load('incidence_matrix.npy', allow_pickle=True)

    A_s = A[:n_s, :]
    A_d = A[n_s:, :]
    n_e = A.shape[1]

    # Supply and demand amounts
    b_s = 10. * np.ones(n_s)
    c   = 5. * np.ones(n_e)

    # Cost
    mu = np.random.uniform(5, 10, size=n_e)

    s_lb, s_ub = 10., 10.  # supply
    d_lb, d_ub = -7., -6.  # demand
    c_lb, c_ub = 5., 5.    # capacity

    # Parameter set X = [x_lb, x_ub] of demands
    x_lb = d_lb * np.ones(n_d)
    x_ub = d_ub * np.ones(n_d)
    z0_lb, z0_ub = np.zeros(n_e), c.copy()
    v0_lb, v0_ub = np.zeros(n_s), np.ones(n_s)
    w0_lb, w0_ub = -np.ones(n_d), np.ones(n_d)

    # Step size choice
    L = np.linalg.norm(np.hstack([A_s.T, A_d.T]), 2)
    eta = 0.5 / L
    return (
        A_d,
        A_s,
        L,
        b_s,
        c,
        c_lb,
        c_ub,
        eta,
        mu,
        n_d,
        n_e,
        n_s,
        v0_lb,
        v0_ub,
        w0_lb,
        w0_ub,
        x_lb,
        x_ub,
        z0_lb,
        z0_ub,
    )


@app.cell
def _(A_d, A_s, b_s, c, cp, mu, n_e, x_ub):
    # Check the feasibility of the generated min cost flow problem
    f = cp.Variable(n_e)

    constraints_check = [
        A_s @ f <= b_s,
        A_d @ f == x_ub,
        0 <= f, f <= c,
    ]

    problem_check = cp.Problem(objective=cp.Minimize(mu.T@f), constraints=constraints_check)
    problem_check.solve(solver=cp.SCS)

    print(problem_check.status, problem_check.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We verify the performance of first-order method solving the network optimization example, up to given iteration budget $K>0$.
    """)
    return


@app.cell
def _():
    K = 5
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate the PDHG algorithm on given problem instances by randomly sampling the parameter $x$ (demand vector) over the parameter set $\mathcal{X}$ and the initial iterates $(z^0, v^0, w^0)$ over their prescribed boxes.
    """)
    return


@app.cell
def _(
    A_d,
    A_s,
    IndBox,
    IndNonNegative,
    K,
    Verifier,
    b_s,
    c,
    eta,
    mu,
    n_d,
    n_s,
    np,
    v0_lb,
    v0_ub,
    w0_lb,
    w0_ub,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    n_samples = 10000

    _VP = Verifier(method='simulator', obbt_solver=None)
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, n_d))
    _z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, len(z0_lb)))
    _v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n_s))
    _w0_samples = _rng.uniform(low=w0_lb, high=w0_ub, size=(n_samples, n_d))
    _x_param = _VP.add_param(ndim=n_d, samples=_x_samples)
    _x_param.lb, _x_param.ub = x_lb, x_ub

    _flow_box = IndBox(l=0.0, u=c)
    _dual_nonnegative = IndNonNegative()
    _z = [None] * (K + 1)
    _v = [None] * (K + 1)
    _w = [None] * (K + 1)
    _z[0] = _VP.add_param(len(z0_lb), samples=_z0_samples)
    _z[0].lb, _z[0].ub = z0_lb, z0_ub
    _v[0] = _VP.add_param(n_s, samples=_v0_samples)
    _v[0].lb, _v[0].ub = v0_lb, v0_ub
    _w[0] = _VP.add_param(n_d, samples=_w0_samples)
    _w[0].lb, _w[0].ub = w0_lb, w0_ub

    for _k in range(1, K + 1):
        _z[_k] = _VP.prox(_flow_box, _z[_k-1] - eta * (mu + A_s.T @ _v[_k-1] - A_d.T @ _w[_k-1]))
        _v[_k] = _VP.prox(_dual_nonnegative, _v[_k-1] + eta * (-b_s + A_s @ (2*_z[_k] - _z[_k-1])))
        _w[_k] = _w[_k-1] + eta * (_x_param - A_d @ (2*_z[_k] - _z[_k-1]))
        _VP.set_objective_norm([
            (1/eta) * (_z[_k] - _z[_k-1]) - A_s.T@(_v[_k] - _v[_k-1]) + A_d.T@(_w[_k] - _w[_k-1]),
            - A_s@(_z[_k] - _z[_k-1]) + (1/eta) * (_v[_k] - _v[_k-1]),
            A_d@(_z[_k] - _z[_k-1]) + (1/eta) * (_w[_k] - _w[_k-1]),
        ], ord=np.inf)

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

    simulation_residuals_pdhg = np.mean(_residuals, axis=0)
    simulation_residual_quantiles_pdhg = np.quantile(_residuals, [0.90, 0.50, 0.10], axis=0)

    _z_star_samples = _z0_samples.copy()
    _v_star_samples = _v0_samples.copy()
    _w_star_samples = _w0_samples.copy()
    for _ in range(2000):
        _z_rhs = _z_star_samples - eta * (
            mu[np.newaxis, :] + _v_star_samples @ A_s - _w_star_samples @ A_d
        )
        _z_next_samples = np.clip(_z_rhs, 0.0, c)
        _v_rhs = _v_star_samples + eta * (
            -b_s[np.newaxis, :] + (2*_z_next_samples - _z_star_samples) @ A_s.T
        )
        _v_next_samples = np.maximum(_v_rhs, 0.0)
        _w_next_samples = _w_star_samples + eta * (
            _x_samples - (2*_z_next_samples - _z_star_samples) @ A_d.T
        )
        _step_norm = np.max(np.linalg.norm(
            np.hstack([
                _z_next_samples - _z_star_samples,
                _v_next_samples - _v_star_samples,
                _w_next_samples - _w_star_samples,
            ]),
            axis=1,
        ))
        _z_star_samples = _z_next_samples
        _v_star_samples = _v_next_samples
        _w_star_samples = _w_next_samples
        if _step_norm <= 1e-10:
            break
    pepit_radius = np.max(np.linalg.norm(
        np.hstack([
            _z0_samples - _z_star_samples,
            _v0_samples - _v_star_samples,
            _w0_samples - _w_star_samples,
        ]),
        axis=1,
    ))
    return (
        n_samples,
        pepit_radius,
        simulation_residual_quantiles_pdhg,
        simulation_residuals_pdhg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also consider a momentum variant of PDHG. After the primal update,
    the method extrapolates the flow iterate as

    \[
        \tilde z^{k} = z^{k} + \frac{k-1}{k+2}(z^{k} - z^{k-1}),
    \]

    and uses $\tilde z^k$ in the dual updates.
    """)
    return


@app.cell
def _(
    A_d,
    A_s,
    IndBox,
    IndNonNegative,
    K,
    Verifier,
    b_s,
    c,
    eta,
    mu,
    n_d,
    n_s,
    n_samples,
    np,
    v0_lb,
    v0_ub,
    w0_lb,
    w0_ub,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    _VP = Verifier(method='simulator', obbt_solver=None)
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, n_d))
    _z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, len(z0_lb)))
    _v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n_s))
    _w0_samples = _rng.uniform(low=w0_lb, high=w0_ub, size=(n_samples, n_d))
    _x_param = _VP.add_param(ndim=n_d, samples=_x_samples)
    _x_param.lb, _x_param.ub = x_lb, x_ub

    _flow_box = IndBox(l=0.0, u=c)
    _dual_nonnegative = IndNonNegative()
    _z = [None] * (K + 1)
    _ztilde = [None] * (K + 1)
    _v = [None] * (K + 1)
    _w = [None] * (K + 1)
    _z[0] = _VP.add_param(len(z0_lb), samples=_z0_samples)
    _z[0].lb, _z[0].ub = z0_lb, z0_ub
    _v[0] = _VP.add_param(n_s, samples=_v0_samples)
    _v[0].lb, _v[0].ub = v0_lb, v0_ub
    _w[0] = _VP.add_param(n_d, samples=_w0_samples)
    _w[0].lb, _w[0].ub = w0_lb, w0_ub

    for _k in range(1, K + 1):
        _z[_k] = _VP.prox(_flow_box, _z[_k-1] - eta * (mu + A_s.T @ _v[_k-1] - A_d.T @ _w[_k-1]))
        _ztilde[_k] = _z[_k] + (_k-1)/(_k+2) * (_z[_k] - _z[_k-1])
        _v[_k] = _VP.prox(_dual_nonnegative, _v[_k-1] + eta * (-b_s + A_s @ (2*_ztilde[_k] - _z[_k-1])))
        _w[_k] = _w[_k-1] + eta * (_x_param - A_d @ (2*_ztilde[_k] - _z[_k-1]))
        _VP.set_objective_norm([
            (1/eta) * (_z[_k] - _z[_k-1]) - A_s.T@(_v[_k] - _v[_k-1]) + A_d.T@(_w[_k] - _w[_k-1]),
            - (1 + 2*(_k-1)/(_k+2)) * A_s@(_z[_k] - _z[_k-1]) + (1/eta) * (_v[_k] - _v[_k-1]),
            (1 + 2*(_k-1)/(_k+2)) * A_d@(_z[_k] - _z[_k-1]) + (1/eta) * (_w[_k] - _w[_k-1]),
        ], ord=np.inf)

    _residuals = _VP.solve()
    simulation_residuals_pdhg_mom = np.mean(_residuals, axis=0)
    simulation_residual_quantiles_pdhg_mom = np.quantile(_residuals, [0.90, 0.50, 0.10], axis=0)
    return (
        simulation_residual_quantiles_pdhg_mom,
        simulation_residuals_pdhg_mom,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now obtain worst-case guarantee by solving the corresponding verification problem using MIP solver.
    """)
    return


@app.cell
def _(
    A_d,
    A_s,
    IndBox,
    IndNonNegative,
    K,
    Verifier,
    b_s,
    c_lb,
    c_ub,
    eta,
    mu,
    n_d,
    n_e,
    n_s,
    np,
    solver_params,
    v0_lb,
    v0_ub,
    w0_lb,
    w0_ub,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(n_d, lb=x_lb, ub=x_ub)
    _c_param = _VP.add_param(n_e, lb=c_lb*np.ones(n_e), ub=c_ub*np.ones(n_e))
    _b_s_param = _VP.add_param(n_s, lb=b_s, ub=b_s)
    _mu_param = _VP.add_param(n_e, lb=mu, ub=mu)

    _z0 = _VP.add_iterate(n_e, lb=z0_lb, ub=z0_ub)
    _v0 = _VP.add_iterate(n_s, lb=v0_lb, ub=v0_ub)
    _w0 = _VP.add_iterate(n_d, lb=w0_lb, ub=w0_ub)
    _z = [None for _ in range(K+1)]
    _v = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]
    _z[0] = _z0
    _v[0] = _v0
    _w[0] = _w0

    all_residuals = []
    worst_case_z0_pdhg = []
    worst_case_v0_pdhg = []
    worst_case_w0_pdhg = []
    worst_case_x_param_pdhg = []
    _flow_box = IndBox(l=0.0, u=_c_param)
    _dual_nonnegative = IndNonNegative()
    for _k in range(1, K+1):
        _z[_k] = _VP.prox(_flow_box, _z[_k-1] - eta * (_mu_param + A_s.T @ _v[_k-1] - A_d.T @ _w[_k-1]))
        _v[_k] = _VP.prox(_dual_nonnegative, _v[_k-1] + eta * (- _b_s_param + A_s @ (2*_z[_k] - _z[_k-1])))
        _w[_k] = _w[_k-1] + eta * (_x_param - A_d @ (2*_z[_k] - _z[_k-1]))
        # _VP.set_objective_norm([_z[_k] - _z[_k-1], _v[_k] - _v[_k-1], _w[_k] - _w[_k-1]], 2)
        _VP.set_objective_norm([
            (1/eta) * (_z[_k] - _z[_k-1]) - A_s.T@(_v[_k] - _v[_k-1]) + A_d.T@(_w[_k] - _w[_k-1]),
            - A_s@(_z[_k] - _z[_k-1]) + (1/eta) * (_v[_k] - _v[_k-1]),
            A_d@(_z[_k] - _z[_k-1]) + (1/eta) * (_w[_k] - _w[_k-1])
        ], 'inf')
        _res = _VP.solve()
        all_residuals.append(_res)
        worst_case_z0_pdhg.append(_VP.get_solution_value(_z0))
        worst_case_v0_pdhg.append(_VP.get_solution_value(_v0))
        worst_case_w0_pdhg.append(_VP.get_solution_value(_w0))
        worst_case_x_param_pdhg.append(_VP.get_solution_value(_x_param))

    print(all_residuals)
    return (
        all_residuals,
        worst_case_v0_pdhg,
        worst_case_w0_pdhg,
        worst_case_x_param_pdhg,
        worst_case_z0_pdhg,
    )


@app.cell
def _(
    A_d,
    A_s,
    IndBox,
    IndNonNegative,
    K,
    b_s,
    c,
    eta,
    mu,
    np,
    worst_case_v0_pdhg,
    worst_case_w0_pdhg,
    worst_case_x_param_pdhg,
    worst_case_z0_pdhg,
):
    # Simulate each horizon with its own worst-case demand vector.
    worst_case_residuals_pdhg = []
    _flow_box = IndBox(l=0.0, u=c)
    _dual_nonnegative = IndNonNegative()
    for _k in range(1, K+1):
        _x_k = np.array(worst_case_x_param_pdhg[_k-1], dtype=float)

        _z = np.array(worst_case_z0_pdhg[_k-1], dtype=float)
        _v = np.array(worst_case_v0_pdhg[_k-1], dtype=float)
        _w = np.array(worst_case_w0_pdhg[_k-1], dtype=float)
        _last_residual = 0.0

        for _j in range(1, _k + 1):
            _z_rhs = _z - eta * (mu + A_s.T @ _v - A_d.T @ _w)
            _z_new = _flow_box.prox(_z_rhs)

            _v_rhs = _v + eta * (-b_s + A_s @ (2 * _z_new - _z))
            _v_new = _dual_nonnegative.prox(_v_rhs)

            _w_new = _w + eta * (_x_k - A_d @ (2 * _z_new - _z))

            _residual_1 = (1 / eta) * (_z_new - _z) - A_s.T @ (_v_new - _v) + A_d.T @ (_w_new - _w)
            _residual_2 = -A_s @ (_z_new - _z) + (1 / eta) * (_v_new - _v)
            _residual_3 = A_d @ (_z_new - _z) + (1 / eta) * (_w_new - _w)
            _last_residual = np.max(np.abs(np.hstack([_residual_1, _residual_2, _residual_3])))

            _z = _z_new
            _v = _v_new
            _w = _w_new

        worst_case_residuals_pdhg.append(_last_residual)

    worst_case_residuals_pdhg = np.array(worst_case_residuals_pdhg)
    return (worst_case_residuals_pdhg,)


@app.cell
def _(
    A_d,
    A_s,
    IndBox,
    IndNonNegative,
    K,
    Verifier,
    b_s,
    c_lb,
    c_ub,
    eta,
    mu,
    n_d,
    n_e,
    n_s,
    np,
    solver_params,
    v0_lb,
    v0_ub,
    w0_lb,
    w0_ub,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    # Solve verification problem for momentum PDHG
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(n_d, lb=x_lb, ub=x_ub)
    _c_param = _VP.add_param(n_e, lb=c_lb*np.ones(n_e), ub=c_ub*np.ones(n_e))
    _b_s_param = _VP.add_param(n_s, lb=b_s, ub=b_s)
    _mu_param = _VP.add_param(n_e, lb=mu, ub=mu)

    _z0 = _VP.add_iterate(n_e, lb=z0_lb, ub=z0_ub)
    _v0 = _VP.add_iterate(n_s, lb=v0_lb, ub=v0_ub)
    _w0 = _VP.add_iterate(n_d, lb=w0_lb, ub=w0_ub)
    _z = [None for _ in range(K+1)]
    _ztilde = [None for _ in range(K+1)]
    _v = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]
    _z[0] = _z0
    _v[0] = _v0
    _w[0] = _w0

    all_residuals_momentum = []
    worst_case_z0_pdhg_mom = []
    worst_case_v0_pdhg_mom = []
    worst_case_w0_pdhg_mom = []
    worst_case_x_param_pdhg_mom = []
    _flow_box = IndBox(l=0.0, u=_c_param)
    _dual_nonnegative = IndNonNegative()
    for _k in range(1, K+1):
        _z[_k] = _VP.prox(_flow_box, _z[_k-1] - eta * (_mu_param + A_s.T @ _v[_k-1] - A_d.T @ _w[_k-1]))
        _ztilde[_k] = _z[_k] + (_k-1)/(_k+2) * (_z[_k] - _z[_k-1])
        _v[_k] = _VP.prox(_dual_nonnegative, _v[_k-1] + eta * (- _b_s_param + A_s @ (2*_ztilde[_k] - _z[_k-1])))
        _w[_k] = _w[_k-1] + eta * (_x_param - A_d @ (2*_ztilde[_k] - _z[_k-1]))
        _VP.set_objective_norm([
            (1/eta) * (_z[_k] - _z[_k-1]) - A_s.T@(_v[_k] - _v[_k-1]) + A_d.T@(_w[_k] - _w[_k-1]),
            - (1 + 2*(_k-1)/(_k+2)) * A_s@(_z[_k] - _z[_k-1]) + (1/eta) * (_v[_k] - _v[_k-1]),
            (1 + 2*(_k-1)/(_k+2)) * A_d@(_z[_k] - _z[_k-1]) + (1/eta) * (_w[_k] - _w[_k-1])
        ], 'inf')
        _res = _VP.solve()
        all_residuals_momentum.append(_res)
        worst_case_z0_pdhg_mom.append(_VP.get_solution_value(_z0))
        worst_case_v0_pdhg_mom.append(_VP.get_solution_value(_v0))
        worst_case_w0_pdhg_mom.append(_VP.get_solution_value(_w0))
        worst_case_x_param_pdhg_mom.append(_VP.get_solution_value(_x_param))

    print(all_residuals_momentum)
    return (
        all_residuals_momentum,
        worst_case_v0_pdhg_mom,
        worst_case_w0_pdhg_mom,
        worst_case_x_param_pdhg_mom,
        worst_case_z0_pdhg_mom,
    )


@app.cell
def _(
    A_d,
    A_s,
    IndBox,
    IndNonNegative,
    K,
    b_s,
    c,
    eta,
    mu,
    np,
    worst_case_v0_pdhg_mom,
    worst_case_w0_pdhg_mom,
    worst_case_x_param_pdhg_mom,
    worst_case_z0_pdhg_mom,
):
    # Simulate each horizon with its own worst-case demand vector for momentum PDHG.
    worst_case_residuals_pdhg_mom = []
    _flow_box = IndBox(l=0.0, u=c)
    _dual_nonnegative = IndNonNegative()
    for _k in range(1, K + 1):
        _x_k = np.array(worst_case_x_param_pdhg_mom[_k - 1], dtype=float)

        _z = np.array(worst_case_z0_pdhg_mom[_k - 1], dtype=float)
        _v = np.array(worst_case_v0_pdhg_mom[_k - 1], dtype=float)
        _w = np.array(worst_case_w0_pdhg_mom[_k - 1], dtype=float)
        _last_residual = 0.0

        for _j in range(1, _k + 1):
            _z_rhs = _z - eta * (mu + A_s.T @ _v - A_d.T @ _w)
            _z_new = _flow_box.prox(_z_rhs)

            _z_tilde = _z_new + (_j - 1) / (_j + 2) * (_z_new - _z)
            _v_rhs = _v + eta * (-b_s + A_s @ (2 * _z_tilde - _z))
            _v_new = _dual_nonnegative.prox(_v_rhs)

            _w_new = _w + eta * (_x_k - A_d @ (2 * _z_tilde - _z))

            _gamma_j = 1 + 2 * (_j - 1) / (_j + 2)
            _residual_1 = (1 / eta) * (_z_new - _z) - A_s.T @ (_v_new - _v) + A_d.T @ (_w_new - _w)
            _residual_2 = -_gamma_j * A_s @ (_z_new - _z) + (1 / eta) * (_v_new - _v)
            _residual_3 = _gamma_j * A_d @ (_z_new - _z) + (1 / eta) * (_w_new - _w)
            _last_residual = np.max(np.abs(np.hstack([_residual_1, _residual_2, _residual_3])))

            _z = _z_new
            _v = _v_new
            _w = _w_new

        worst_case_residuals_pdhg_mom.append(_last_residual)

    worst_case_residuals_pdhg_mom = np.array(worst_case_residuals_pdhg_mom)
    return (worst_case_residuals_pdhg_mom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compare our results to the worst-case performance of the algorithm returned
    by the performance estimation problem (PEP), using the shared `PEPitRunner`.
    Specifically, we measure the performance of PDHG applied on

    \begin{array}{ll}
        \textnormal{minimize}_x & \textnormal{maximize}_u & \mathcal{L}(x, u) = f(x) + u^T (Ax-b) - g^*(u),
    \end{array}

    which can be written as

    \begin{array}{lll}
        x^{k+1} &=& \mathbf{prox}_{\eta f} (x^k - \eta A^T u^k) \\
        u^{k+1} &=& \mathbf{prox}_{\eta g^*} (u^k + \eta ( A (2x^{k+1} - x^k) - b ) ).
    \end{array}
    """)
    return


@app.cell
def _(K, L, PEPitRunner, eta, pepit_radius):
    all_residuals_PEP, _ = PEPitRunner().pdhg_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius)
    return (all_residuals_PEP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we plot the verification results against the PEPit bound and the
    horizon-specific worst-case instances extracted from MIP.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    plt,
    simulation_residual_quantiles_pdhg,
    simulation_residuals_pdhg,
    worst_case_residuals_pdhg,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='PDHG (PEPit)')
    plt.plot(_ks, all_residuals, label='PDHG (MIP worst-case)')
    plt.plot(_ks, simulation_residuals_pdhg, '--', label='PDHG (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_pdhg[2], simulation_residual_quantiles_pdhg[0], alpha=.18, label='PDHG simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals_pdhg, '-.', color='k', linewidth=.7, label='worst-case PDHG instance')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case KKT residual')
    plt.legend()
    plt.suptitle('network optimization example')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also plot the verification results for momentum PDHG.
    """)
    return


@app.cell
def _(
    K,
    all_residuals_momentum,
    plt,
    simulation_residual_quantiles_pdhg_mom,
    simulation_residuals_pdhg_mom,
    worst_case_residuals_pdhg_mom,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_momentum, label='momentum PDHG (MIP worst-case)')
    plt.plot(_ks, simulation_residuals_pdhg_mom, '--', label='momentum PDHG (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_pdhg_mom[2], simulation_residual_quantiles_pdhg_mom[0], alpha=.18, label='momentum PDHG simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals_pdhg_mom, '-.', color='k', linewidth=.7, label='worst-case momentum PDHG instance')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case KKT residual')
    plt.legend()
    plt.suptitle('network optimization momentum PDHG example')
    return


if __name__ == "__main__":
    app.run()
