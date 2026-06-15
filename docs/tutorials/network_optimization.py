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

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    np.random.seed(2)
    return Verifier, cp, np, plt


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
        z^{k+1} &=& \Pi_{[0, c]} \big( z^k - \eta (\mu + A_s^T v^k - A_d^T w^k) \big) \\
        v^{k+1} &=& \max \big\{ v^k + \eta \big( - b_s + A_s (2 z^{k+1} - z^k) \big) \big\} \\
        w^{k+1} &=& w^k + \eta \big( x - A_d (2 z^{k+1} - z^k) \big).
    \end{array}

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
    return (
        A_d,
        A_s,
        b_s,
        c,
        c_lb,
        c_ub,
        d_lb,
        mu,
        n_d,
        n_e,
        n_s,
        s_lb,
        x_lb,
        x_ub,
    )


@app.cell
def _(A_d, A_s, c, cp, d_lb, mu, n_d, n_e, n_s, np, s_lb):
    # Initial iterate s^0 as primal-dual solution to the problem instance
    # with all demand nodes having maximum demand
    import scipy.sparse as spa

    A_block = spa.bmat([
            [A_s, spa.eye(n_s), None],
            [A_d, None, None],
            [spa.eye(n_e), None, spa.eye(n_e)]
        ])
    A_block = A_block.todense()

    b_block = np.hstack([
                s_lb * np.ones(n_s),
                d_lb * np.ones(n_d),
                c,
            ])

    c_block = np.zeros(A_block.shape[1])
    c_block[:n_e] = mu

    # Solve cvxpy problem to get initial z[0]
    x_block = cp.Variable(A_block.shape[1])

    objective = c_block.T @ x_block

    constraints = [
        A_block @ x_block == b_block,
        x_block >= 0
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.SCS)

    print(problem.status, problem.value)
    z0_value = x_block.value[:n_e]
    return (z0_value,)


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


@app.cell
def _(A_d, A_s, np):
    K = 5
    L = np.linalg.norm(np.hstack([A_s.T, A_d.T]), 2)
    eta = 0.5 / L

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }

    n_samples = 10000
    return K, L, eta, n_samples, solver_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate the PDHG algorithm on given problem instances by randomly sampling the parameter $x$ (demand vector) over the parameter set $\mathcal{X}$.
    """)
    return


@app.cell
def _(
    A_d,
    A_s,
    K,
    b_s,
    c,
    eta,
    mu,
    n_d,
    n_s,
    n_samples,
    np,
    x_lb,
    x_ub,
    z0_value,
):
    # Simulate PDHG on sampled problem instances
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, n_d))

    # PDHG iteration over samples (batched)
    _z_batch = np.tile(z0_value, (n_samples, 1))
    _v_batch = np.zeros((n_samples, n_s), dtype=float)
    _w_batch = np.zeros((n_samples, n_d), dtype=float)

    _residuals_sim = np.empty((n_samples, K), dtype=float)

    for _k in range(1, K + 1):
        # satlin: z update with box constraints [0, c]
        _z_rhs = _z_batch - eta * (mu[np.newaxis, :] + _v_batch @ A_s - _w_batch @ A_d)
        _z_new_batch = np.clip(_z_rhs, 0, c[np.newaxis, :])

        # relu: v update
        _v_rhs = _v_batch + eta * (-b_s[np.newaxis, :] + (2*_z_new_batch - _z_batch) @ A_s.T)
        _v_new_batch = np.maximum(_v_rhs, 0.0)

        # linear: w update
        _w_new_batch = _w_batch + eta * (_x_samples - (2*_z_new_batch - _z_batch) @ A_d.T)

        # residual (scaled fixed-point residual)
        _residual_1 = (1/eta) * (_z_new_batch - _z_batch) - (_v_new_batch - _v_batch) @ A_s + (_w_new_batch - _w_batch) @ A_d
        _residual_2 = -(_z_new_batch - _z_batch) @ A_s.T + (1/eta) * (_v_new_batch - _v_batch)
        _residual_3 = (_z_new_batch - _z_batch) @ A_d.T + (1/eta) * (_w_new_batch - _w_batch)
        _residuals_sim[:, _k - 1] = np.max(np.abs(np.hstack([_residual_1, _residual_2, _residual_3])), axis=1)

        _z_batch = _z_new_batch
        _v_batch = _v_new_batch
        _w_batch = _w_new_batch

    simulation_residuals_pdhg = np.mean(_residuals_sim, axis=0)
    return (simulation_residuals_pdhg,)


@app.cell
def _(
    A_d,
    A_s,
    K,
    b_s,
    c,
    eta,
    mu,
    n_d,
    n_s,
    n_samples,
    np,
    x_lb,
    x_ub,
    z0_value,
):
    # Simulate momentum PDHG on sampled problem instances
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, n_d))

    # Momentum PDHG iteration over samples (batched)
    _z_batch = np.tile(z0_value, (n_samples, 1))
    _v_batch = np.zeros((n_samples, n_s), dtype=float)
    _w_batch = np.zeros((n_samples, n_d), dtype=float)

    _residuals_sim_mom = np.empty((n_samples, K), dtype=float)

    for _k in range(1, K + 1):
        # satlin: z update with box constraints [0, c]
        _z_rhs = _z_batch - eta * (mu[np.newaxis, :] + _v_batch @ A_s - _w_batch @ A_d)
        _z_new_batch = np.clip(_z_rhs, 0, c[np.newaxis, :])

        # momentum extrapolation
        _z_tilde_batch = _z_new_batch + (_k - 1) / (_k + 2) * (_z_new_batch - _z_batch)

        # relu: v update
        _v_rhs = _v_batch + eta * (-b_s[np.newaxis, :] + (2*_z_tilde_batch - _z_batch) @ A_s.T)
        _v_new_batch = np.maximum(_v_rhs, 0.0)

        # linear: w update
        _w_new_batch = _w_batch + eta * (_x_samples - (2*_z_tilde_batch - _z_batch) @ A_d.T)

        # residual (scaled fixed-point residual with momentum correction)
        _residual_1 = (1/eta) * (_z_new_batch - _z_batch) - (_v_new_batch - _v_batch) @ A_s + (_w_new_batch - _w_batch) @ A_d
        _residual_2 = -(1 + 2*(_k-1)/(_k+2)) * (_z_new_batch - _z_batch) @ A_s.T + (1/eta) * (_v_new_batch - _v_batch)
        _residual_3 = (1 + 2*(_k-1)/(_k+2)) * (_z_new_batch - _z_batch) @ A_d.T + (1/eta) * (_w_new_batch - _w_batch)
        _residuals_sim_mom[:, _k - 1] = np.max(np.abs(np.hstack([_residual_1, _residual_2, _residual_3])), axis=1)

        _z_batch = _z_new_batch
        _v_batch = _v_new_batch
        _w_batch = _w_new_batch

    simulation_residuals_pdhg_mom = np.mean(_residuals_sim_mom, axis=0)
    return (simulation_residuals_pdhg_mom,)


@app.cell
def _(
    A_d,
    A_s,
    K,
    Verifier,
    b_s,
    c,
    c_lb,
    c_ub,
    eta,
    mu,
    n_d,
    n_e,
    n_s,
    np,
    solver_params,
    x_lb,
    x_ub,
    z0_value,
):
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(n_d, lb=x_lb, ub=x_ub)
    _c_param = _VP.add_param(n_e, lb=c_lb*np.ones(n_e), ub=c_ub*np.ones(n_e))
    _b_s_param = _VP.add_param(n_s, lb=b_s, ub=b_s)
    _mu_param = _VP.add_param(n_e, lb=mu, ub=mu)

    _z0 = _VP.add_iterate(n_e, lb=z0_value, ub=z0_value)
    _v0 = _VP.add_iterate(n_s, lb=np.zeros(n_s), ub=np.zeros(n_s))
    _w0 = _VP.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))
    _z = [None for _ in range(K+1)]
    _v = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]
    _z[0] = _z0
    _v[0] = _v0
    _w[0] = _w0

    all_residuals = []
    worst_case_x_param_pdhg = []
    for _k in range(1, K+1):
        _z[_k] = _VP.step.satlin(_z[_k-1] - eta * (_mu_param + A_s.T @ _v[_k-1] - A_d.T @ _w[_k-1]), l=np.zeros(n_e), u=c)
        _v[_k] = _VP.step.relu(_v[_k-1] + eta * (- _b_s_param + A_s @ (2*_z[_k] - _z[_k-1])))
        _w[_k] = _w[_k-1] + eta * (_x_param - A_d @ (2*_z[_k] - _z[_k-1]))
        # _VP.set_objective_norm([_z[_k] - _z[_k-1], _v[_k] - _v[_k-1], _w[_k] - _w[_k-1]], 2)
        _VP.set_objective_norm([
            (1/eta) * (_z[_k] - _z[_k-1]) - A_s.T@(_v[_k] - _v[_k-1]) + A_d.T@(_w[_k] - _w[_k-1]),
            - A_s@(_z[_k] - _z[_k-1]) + (1/eta) * (_v[_k] - _v[_k-1]),
            A_d@(_z[_k] - _z[_k-1]) + (1/eta) * (_w[_k] - _w[_k-1])
        ], 'inf')
        _res = _VP.solve()
        all_residuals.append(_res)
        worst_case_x_param_pdhg.append(_VP.get_solution_value(_x_param))

    print(all_residuals)
    return all_residuals, worst_case_x_param_pdhg


@app.cell
def _(
    A_d,
    A_s,
    K,
    b_s,
    c,
    eta,
    mu,
    n_d,
    n_s,
    np,
    worst_case_x_param_pdhg,
    z0_value,
):
    # Simulate each horizon with its own worst-case demand vector.
    worst_case_residuals_pdhg = []
    for _k in range(1, K+1):
        _x_k = np.array(worst_case_x_param_pdhg[_k-1], dtype=float)

        _z = np.array(z0_value, dtype=float)
        _v = np.zeros(n_s, dtype=float)
        _w = np.zeros(n_d, dtype=float)
        _last_residual = 0.0

        for _j in range(1, _k + 1):
            _z_rhs = _z - eta * (mu + A_s.T @ _v - A_d.T @ _w)
            _z_new = np.clip(_z_rhs, 0, c)

            _v_rhs = _v + eta * (-b_s + A_s @ (2 * _z_new - _z))
            _v_new = np.maximum(_v_rhs, 0.0)

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
    K,
    Verifier,
    b_s,
    c,
    c_lb,
    c_ub,
    eta,
    mu,
    n_d,
    n_e,
    n_s,
    np,
    solver_params,
    x_lb,
    x_ub,
    z0_value,
):
    # Solve verification problem for momentum PDHG
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(n_d, lb=x_lb, ub=x_ub)
    _c_param = _VP.add_param(n_e, lb=c_lb*np.ones(n_e), ub=c_ub*np.ones(n_e))
    _b_s_param = _VP.add_param(n_s, lb=b_s, ub=b_s)
    _mu_param = _VP.add_param(n_e, lb=mu, ub=mu)

    _z0 = _VP.add_iterate(n_e, lb=z0_value, ub=z0_value)
    _v0 = _VP.add_iterate(n_s, lb=np.zeros(n_s), ub=np.zeros(n_s))
    _w0 = _VP.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))
    _z = [None for _ in range(K+1)]
    _ztilde = [None for _ in range(K+1)]
    _v = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]
    _z[0] = _z0
    _v[0] = _v0
    _w[0] = _w0

    all_residuals_momentum = []
    worst_case_x_param_pdhg_mom = []
    for _k in range(1, K+1):
        _z[_k] = _VP.step.satlin(_z[_k-1] - eta * (_mu_param + A_s.T @ _v[_k-1] - A_d.T @ _w[_k-1]), l=np.zeros(n_e), u=c)
        _ztilde[_k] = _z[_k] + (_k-1)/(_k+2) * (_z[_k] - _z[_k-1])
        _v[_k] = _VP.step.relu(_v[_k-1] + eta * (- _b_s_param + A_s @ (2*_ztilde[_k] - _z[_k-1])))
        _w[_k] = _w[_k-1] + eta * (_x_param - A_d @ (2*_ztilde[_k] - _z[_k-1]))
        _VP.set_objective_norm([
            (1/eta) * (_z[_k] - _z[_k-1]) - A_s.T@(_v[_k] - _v[_k-1]) + A_d.T@(_w[_k] - _w[_k-1]),
            - (1 + 2*(_k-1)/(_k+2)) * A_s@(_z[_k] - _z[_k-1]) + (1/eta) * (_v[_k] - _v[_k-1]),
            (1 + 2*(_k-1)/(_k+2)) * A_d@(_z[_k] - _z[_k-1]) + (1/eta) * (_w[_k] - _w[_k-1])
        ], 'inf')
        _res = _VP.solve()
        all_residuals_momentum.append(_res)
        worst_case_x_param_pdhg_mom.append(_VP.get_solution_value(_x_param))

    print(all_residuals_momentum)
    return all_residuals_momentum, worst_case_x_param_pdhg_mom


@app.cell
def _(
    A_d,
    A_s,
    K,
    b_s,
    c,
    eta,
    mu,
    n_d,
    n_s,
    np,
    worst_case_x_param_pdhg_mom,
    z0_value,
):
    # Simulate each horizon with its own worst-case demand vector for momentum PDHG.
    worst_case_residuals_pdhg_mom = []
    for _k in range(1, K + 1):
        _x_k = np.array(worst_case_x_param_pdhg_mom[_k - 1], dtype=float)

        _z = np.array(z0_value, dtype=float)
        _v = np.zeros(n_s, dtype=float)
        _w = np.zeros(n_d, dtype=float)
        _last_residual = 0.0

        for _j in range(1, _k + 1):
            _z_rhs = _z - eta * (mu + A_s.T @ _v - A_d.T @ _w)
            _z_new = np.clip(_z_rhs, 0, c)

            _z_tilde = _z_new + (_j - 1) / (_j + 2) * (_z_new - _z)
            _v_rhs = _v + eta * (-b_s + A_s @ (2 * _z_tilde - _z))
            _v_new = np.maximum(_v_rhs, 0.0)

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


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_momentum,
    plt,
    simulation_residuals_pdhg,
    simulation_residuals_pdhg_mom,
    worst_case_residuals_pdhg,
    worst_case_residuals_pdhg_mom,
):
    plt.figure()
    plt.plot(range(1, K+1), all_residuals, label='PDHG (MIP worst-case)')
    plt.plot(range(1, K+1), simulation_residuals_pdhg, '--', label='PDHG (in-sample avg)')
    plt.plot(range(1, K+1), worst_case_residuals_pdhg, '-.', color='k', linewidth=.5, label='worst-case PDHG instance')
    plt.plot(range(1, K+1), all_residuals_momentum, label='momentum PDHG (MIP worst-case)')
    plt.plot(range(1, K+1), simulation_residuals_pdhg_mom, '--', label='momentum PDHG (in-sample avg)')
    plt.plot(range(1, K+1), worst_case_residuals_pdhg_mom, ':', color='k', linewidth=.8, label='worst-case momentum PDHG instance')
    # plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case KKT residual')
    plt.legend()
    plt.suptitle('network optimization example (MIP)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compare our results to the worst-case performance of the algorithm returned by the performance estimation problem (PEP), using the package [`PEPit`](https://pepit.readthedocs.io/en/0.5.1/). Specifically, we measure the performance of PDHG applied on

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
def _(K, L, eta):
    from PEPit import PEP, Point
    from PEPit.functions import ConvexFunction
    from PEPit.operators import LinearOperator
    from PEPit.primitive_steps import proximal_step

    all_residuals_PEP = []
    for _k_PEP in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(ConvexFunction)
        _g = _problem.declare_function(ConvexFunction)
        _linear = _problem.declare_function(LinearOperator, L=L)

        # Define optimal primal-dual pair (xs, us)
        _xs = Point()
        _us = Point()
        _f.add_point((_xs, -_linear.T.gradient(_us), _f(_xs)))
        _g.add_point((_us, _linear.gradient(_xs), _g(_us)))

        # Declare initial points (x0, u0)
        _x0 = _problem.set_initial_point()
        _u0 = _problem.set_initial_point()
        _problem.set_initial_condition((_x0 - _xs)**2 + (_u0 - _us)**2 <= 1.0)

        _x = [None for _ in range(_k_PEP+1)]
        _u = [None for _ in range(_k_PEP+1)]
        _x[0] = _x0
        _u[0] = _u0

        for _k in range(1, _k_PEP+1):
            _x[_k], _, _ = proximal_step(_x[_k-1] - eta * _linear.T.gradient(_u[_k-1]), _f, eta)
            _u[_k], _, _ = proximal_step(_u[_k-1] + eta * _linear.gradient(2*_x[_k] - _x[_k-1]), _g, eta)
        _problem.set_performance_metric((_x[_k_PEP] - _x[_k_PEP-1])**2 + (_u[_k_PEP] - _u[_k_PEP-1])**2)
        _res = _problem.solve(solver='SCS')
        all_residuals_PEP.append(_res) # account for L-infinity norm of VP
    return (
        ConvexFunction,
        LinearOperator,
        PEP,
        Point,
        all_residuals_PEP,
        proximal_step,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now obtain the result for momentum PDHG:

    \begin{array}{lll}
        x^{k+1} &=& \mathbf{prox}_{\eta f} (x^k - \eta A^T u^k) \\
        \tilde{x}^{k+1} &=& x^{k+1} + \frac{k}{k+3} (x^{k+1} - x^k) \\
        u^{k+1} &=& \mathbf{prox}_{\eta g^*} (u^k + \eta ( A (2\tilde{x}^{k+1} - x^k) - b ) ).
    \end{array}
    """)
    return


@app.cell
def _(ConvexFunction, K, L, LinearOperator, PEP, Point, eta, proximal_step):
    all_residuals_momentum_PEP = []
    for _k_PEP in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(ConvexFunction)
        _g = _problem.declare_function(ConvexFunction)
        _linear = _problem.declare_function(LinearOperator, L=L)

        # Define optimal primal-dual pair (xs, us)
        _xs = Point()
        _us = Point()
        _f.add_point((_xs, -_linear.T.gradient(_us), _f(_xs)))
        _g.add_point((_us, _linear.gradient(_xs), _g(_us)))

        # Declare initial points (x0, u0)
        _x0 = _problem.set_initial_point()
        _u0 = _problem.set_initial_point()
        _problem.set_initial_condition((_x0 - _xs)**2 + (_u0 - _us)**2 <= 1.0)

        _x = [None for _ in range(_k_PEP+1)]
        _xtilde = [None for _ in range(_k_PEP+1)]
        _u = [None for _ in range(_k_PEP+1)]
        _x[0] = _x0
        _u[0] = _u0

        for _k in range(1, _k_PEP+1):
            _x[_k], _, _ = proximal_step(_x[_k-1] - eta * _linear.T.gradient(_u[_k-1]), _f, eta)
            _xtilde[_k] = _x[_k] + (_k-1)/(_k+2) * (_x[_k] - _x[_k-1])
            _u[_k], _, _ = proximal_step(_u[_k-1] + eta * _linear.gradient(2*_xtilde[_k] - _x[_k-1]), _g, eta)
        _problem.set_performance_metric((_x[_k_PEP] - _x[_k_PEP-1])**2 + (_u[_k_PEP] - _u[_k_PEP-1])**2)
        _res = _problem.solve(solver='SCS')
        all_residuals_momentum_PEP.append(_res) # account for L-infinity norm of VP
    return (all_residuals_momentum_PEP,)


@app.cell
def _(K, all_residuals_PEP, all_residuals_momentum_PEP, plt):
    plt.figure()
    plt.plot(range(1, K+1), all_residuals_PEP, label='PDHG')
    plt.plot(range(1, K+1), all_residuals_momentum_PEP, label='momentum PDHG')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('network optimization example (PEP)')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
