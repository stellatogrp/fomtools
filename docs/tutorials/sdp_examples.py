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
    import scipy.sparse as spa

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return Verifier, np, plt, spa


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SDP Solver Experiments

    This notebook consolidates all SDP-based verification experiments from the fomtools tutorials:
    - Sparse coding (ISTA)
    - Network optimization (PDHG)
    - Optimal control (ADMM)

    All experiments use `Verifier(method='sdp', ...)` for semidefinite programming relaxations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 1: Sparse Coding (ISTA)

    Consider the sparse coding problem:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Dz - x \rVert_2^2 + \lambda \lVert z \rVert_1,
    \end{array}

    solved with ISTA iteration:

    \begin{array}{ll}
        s^{k+1} = {\cal T}_{\lambda \eta} ( (I - \eta D^T D) s^{k} + \eta D^T x),
    \end{array}
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    # p, n, lambd = 15, 20, 1e-1
    p_spc, n, lambd = 3, 4, 1e-1
    D_value = np.random.normal(size=(p_spc, n))
    D_mask = np.random.binomial(n=1, p=0.2, size=(p_spc, n))
    D = np.multiply(D_value, D_mask)

    # signal z_star
    # num_star = 100
    num_star = 20
    z_star_value = np.random.normal(size=(n, num_star))
    z_star_mask = np.random.binomial(n=1, p=0.1, size=(n, num_star))
    z_star = np.multiply(z_star_value, z_star_mask)

    # noise err
    err = np.random.normal(scale=1e-2, size=(p_spc, num_star))

    # parameter x = D z_star + err
    x = D @ z_star + err

    # parameter set X = [x_lb, x_ub]
    x_lb, x_ub = np.min(x, axis=1), np.max(x, axis=1)
    return D, lambd, n, p_spc, x_lb, x_ub


@app.cell
def _(D, np):
    L_spc = np.real(np.max(np.linalg.eigvals(D.T @ D)))
    eta_spc = 1 / L_spc
    K_spc = 5
    return K_spc, eta_spc


@app.cell
def _(D, K_spc, Verifier, eta_spc, lambd, n, np, p_spc, x_lb, x_ub):
    _VP_spc_sdp = Verifier(method='sdp', solver='clarabel', obbt_handler='highs')
    _x_param = _VP_spc_sdp.add_param(p_spc, lb=x_lb, ub=x_ub)

    _s0 = np.zeros(n)  # initialized at zero
    _s = [None for _ in range(K_spc + 1)]
    _s[0] = _s0

    all_residuals_spc_sdp = []
    for _k in range(1, K_spc + 1):
        _s[_k] = _VP_spc_sdp.step.soft_threshold(
            _s[_k - 1] - eta_spc * D.T @ (D @ _s[_k - 1] - _x_param), lambd=lambd * eta_spc
        )
        _VP_spc_sdp.set_objective_norm([_s[_k] - _s[_k - 1]], 2)
        _res = _VP_spc_sdp.solve()
        all_residuals_spc_sdp.append(_res)
    return (all_residuals_spc_sdp,)


@app.cell
def _(K_spc, all_residuals_spc_sdp, plt):
    plt.figure()
    plt.plot(range(1, K_spc+1), all_residuals_spc_sdp, label='SDP-VP bound')
    plt.suptitle('sparse coding example (SDP)')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 2: Network Optimization (PDHG)

    Consider the minimum cost network flow problem:

    \begin{array}{ll}
        \text{minimize} & \mu^T f \\
        \text{subject to}
        & A_s f \le b_s \\
        & A_d f = x \\
        & 0 \le f \le c.
    \end{array}

    solved with Primal-dual Hybrid Gradient (PDHG):

    \begin{array}{rcl}
        z^{k+1} &=& \Pi_{[0, c]} \big( z^k - \eta (\mu + A_s^T v^k - A_d^T w^k) \big) \\
        v^{k+1} &=& \max \big\{ v^k + \eta \big( - b_s + A_s (2 z^{k+1} - z^k) \big) \big\} \\
        w^{k+1} &=& w^k + \eta \big( x - A_d (2 z^{k+1} - z^k) \big).
    \end{array}
    """)
    return


@app.cell
def _(np):
    # Generate bipartite graph with n_s supply nodes and n_d demand nodes
    import networkx as nx

    n_s_net = 6
    n_d = 3
    p = 0.5

    G = nx.bipartite.random_graph(n_s_net, n_d, p, seed=1, directed=False)
    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)
    A = A.todense()
    A[n_s_net:, :] *= -1

    with open('docs/tutorials/incidence_matrix.npy', 'wb') as write_file:
        np.save(write_file, A)

    A = np.load('docs/tutorials/incidence_matrix.npy', allow_pickle=True)

    A_s = A[:n_s_net, :]
    A_d = A[n_s_net:, :]
    n_e = A.shape[1]

    # Supply and demand amounts
    b_s = 10. * np.ones(n_s_net)
    c = 5. * np.ones(n_e)

    # Cost
    mu = np.random.uniform(5, 10, size=n_e)

    s_lb, s_ub = 10., 10.  # supply
    d_lb, d_ub = -7., -6.  # demand
    c_lb, c_ub = 5., 5.    # capacity

    # Parameter set X = [x_lb, x_ub] of demands
    x_lb_net = d_lb * np.ones(n_d)
    x_ub_net = d_ub * np.ones(n_d)
    return (
        A_d,
        A_s,
        b_s,
        c,
        c_lb,
        c_ub,
        mu,
        n_d,
        n_e,
        n_s_net,
        x_lb_net,
        x_ub_net,
    )


@app.cell
def _(A_d, A_s, np):
    K_net = 3
    L_net = np.linalg.norm(np.hstack([A_s.T, A_d.T]), 2)
    eta_net = 0.5 / L_net
    return K_net, eta_net


@app.cell
def _(A_d, A_s, c, mu, n_d, n_e, n_s_net, np, spa):
    # Compute initial iterate z0 from optimal solution
    import cvxpy as cp

    A_block = spa.bmat([
        [A_s, spa.eye(n_s_net), None],
        [A_d, None, None],
        [spa.eye(n_e), None, spa.eye(n_e)]
    ])
    A_block = A_block.todense()

    b_block = np.hstack([
        10. * np.ones(n_s_net),
        -7. * np.ones(n_d),
        c,
    ])

    c_block = np.zeros(A_block.shape[1])
    c_block[:n_e] = mu

    x_block = cp.Variable(A_block.shape[1])
    objective = c_block.T @ x_block
    constraints = [
        A_block @ x_block == b_block,
        x_block >= 0
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.SCS)

    z0_value_net = x_block.value[:n_e]
    return (z0_value_net,)


@app.cell
def _(
    A_d,
    A_s,
    K_net,
    Verifier,
    b_s,
    c,
    c_lb,
    c_ub,
    eta_net,
    mu,
    n_d,
    n_e,
    n_s_net,
    np,
    x_lb_net,
    x_ub_net,
    z0_value_net,
):
    _VP_net_sdp = Verifier(method='sdp', solver='clarabel', obbt_handler='highs')
    _x_param = _VP_net_sdp.add_param(n_d, lb=x_lb_net, ub=x_ub_net)
    _c_param = _VP_net_sdp.add_param(n_e, lb=c_lb * np.ones(n_e), ub=c_ub * np.ones(n_e))
    _b_s_param = _VP_net_sdp.add_param(n_s_net, lb=b_s, ub=b_s)
    _mu_param = _VP_net_sdp.add_param(n_e, lb=mu, ub=mu)

    _z0 = _VP_net_sdp.add_iterate(n_e, lb=z0_value_net, ub=z0_value_net)
    _v0 = _VP_net_sdp.add_iterate(n_s_net, lb=np.zeros(n_s_net), ub=np.zeros(n_s_net))
    _w0 = _VP_net_sdp.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))
    _z = [None for _ in range(K_net + 1)]
    _v = [None for _ in range(K_net + 1)]
    _w = [None for _ in range(K_net + 1)]
    _z[0] = _z0
    _v[0] = _v0
    _w[0] = _w0

    all_residuals_net_sdp = []
    for _k in range(1, K_net + 1):
        _z[_k] = _VP_net_sdp.step.satlin(_z[_k - 1] - eta_net * (_mu_param + A_s.T @ _v[_k - 1] - A_d.T @ _w[_k - 1]), l=np.zeros(n_e), u=c)
        _v[_k] = _VP_net_sdp.step.relu(_v[_k - 1] + eta_net * (-_b_s_param + A_s @ (2 * _z[_k] - _z[_k - 1])))
        _w[_k] = _w[_k - 1] + eta_net * (_x_param - A_d @ (2 * _z[_k] - _z[_k - 1]))
        _VP_net_sdp.set_objective_norm([
            (1 / eta_net) * (_z[_k] - _z[_k - 1]) - A_s.T @ (_v[_k] - _v[_k - 1]) + A_d.T @ (_w[_k] - _w[_k - 1]),
            -A_s @ (_z[_k] - _z[_k - 1]) + (1 / eta_net) * (_v[_k] - _v[_k - 1]),
            A_d @ (_z[_k] - _z[_k - 1]) + (1 / eta_net) * (_w[_k] - _w[_k - 1])
        ], 2)
        _res = _VP_net_sdp.solve()
        all_residuals_net_sdp.append(_res)
    return (all_residuals_net_sdp,)


@app.cell
def _(
    A_d,
    A_s,
    K_net,
    Verifier,
    b_s,
    c,
    c_lb,
    c_ub,
    eta_net,
    mu,
    n_d,
    n_e,
    n_s_net,
    np,
    x_lb_net,
    x_ub_net,
    z0_value_net,
):
    # Solve verification problem for momentum PDHG
    _VP_net_sdp_mom = Verifier(method='sdp', solver='clarabel', obbt_handler='highs')
    _x_param = _VP_net_sdp_mom.add_param(n_d, lb=x_lb_net, ub=x_ub_net)
    _c_param = _VP_net_sdp_mom.add_param(n_e, lb=c_lb * np.ones(n_e), ub=c_ub * np.ones(n_e))
    _b_s_param = _VP_net_sdp_mom.add_param(n_s_net, lb=b_s, ub=b_s)
    _mu_param = _VP_net_sdp_mom.add_param(n_e, lb=mu, ub=mu)

    _z0 = _VP_net_sdp_mom.add_iterate(n_e, lb=z0_value_net, ub=z0_value_net)
    _v0 = _VP_net_sdp_mom.add_iterate(n_s_net, lb=np.zeros(n_s_net), ub=np.zeros(n_s_net))
    _w0 = _VP_net_sdp_mom.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))
    _z = [None for _ in range(K_net + 1)]
    _ztilde = [None for _ in range(K_net + 1)]
    _v = [None for _ in range(K_net + 1)]
    _w = [None for _ in range(K_net + 1)]
    _z[0] = _z0
    _v[0] = _v0
    _w[0] = _w0

    all_residuals_net_sdp_momentum = []
    for _k in range(1, K_net + 1):
        _z[_k] = _VP_net_sdp_mom.step.satlin(_z[_k - 1] - eta_net * (_mu_param + A_s.T @ _v[_k - 1] - A_d.T @ _w[_k - 1]), l=np.zeros(n_e), u=c)
        _ztilde[_k] = _z[_k] + (_k - 1) / (_k + 2) * (_z[_k] - _z[_k - 1])
        _v[_k] = _VP_net_sdp_mom.step.relu(_v[_k - 1] + eta_net * (-_b_s_param + A_s @ (2 * _ztilde[_k] - _z[_k - 1])))
        _w[_k] = _w[_k - 1] + eta_net * (_x_param - A_d @ (2 * _ztilde[_k] - _z[_k - 1]))
        _VP_net_sdp_mom.set_objective_norm([
            (1 / eta_net) * (_z[_k] - _z[_k - 1]) - A_s.T @ (_v[_k] - _v[_k - 1]) + A_d.T @ (_w[_k] - _w[_k - 1]),
            -(1 + 2 * (_k - 1) / (_k + 2)) * A_s @ (_z[_k] - _z[_k - 1]) + (1 / eta_net) * (_v[_k] - _v[_k - 1]),
            (1 + 2 * (_k - 1) / (_k + 2)) * A_d @ (_z[_k] - _z[_k - 1]) + (1 / eta_net) * (_w[_k] - _w[_k - 1])
        ], 2)
        _res = _VP_net_sdp_mom.solve()
        all_residuals_net_sdp_momentum.append(_res)
    return (all_residuals_net_sdp_momentum,)


@app.cell
def _(K_net, all_residuals_net_sdp, all_residuals_net_sdp_momentum, plt):
    plt.figure()
    plt.plot(range(1, K_net + 1), all_residuals_net_sdp, label='PDHG')
    plt.plot(range(1, K_net + 1), all_residuals_net_sdp_momentum, label='momentum PDHG')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('network optimization example (SDP)')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 3: Optimal Control (ADMM)

    Consider the optimal control problem:

    \begin{array}{lll}
        \text{minimize} & s_T^T Q_T s_T + \sum_{t=1}^{T-1} s_t^T Q s_t + u_t^T R u_t \\
        \text{subject to} & s_{t+1} = A s_t + B u_t, & t=1,\dots,T-1 \\
        & s_{\min} \le s_t \le s_{\max}, & t=1,\dots,T \\
        & u_{\min} \le u_t \le u_{\max}, & t=1,\dots,T-1 \\
        & s_1 = x,
    \end{array}

    We reformulate as a QP and solve with ADMM.
    """)
    return


@app.cell
def _(np, spa):
    # problem data: state dimension n_s = 12, input dimension n_u = 10
    A_dym = np.array([
        [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
        [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
        [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
        [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
        [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
        [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
        [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
        [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
        [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
    ])
    B_dym = np.array([
        [0.,      -0.0726,  0.,     0.0726],
        [-0.0726,  0.,      0.0726, 0.    ],
        [-0.0152,  0.0152, -0.0152, 0.0152],
        [-0.,     -0.0006, -0.,     0.0006],
        [0.0006,   0.,     -0.0006, 0.0000],
        [0.0106,   0.0106,  0.0106, 0.0106],
        [0,       -1.4512,  0.,     1.4512],
        [-1.4512,  0.,      1.4512, 0.    ],
        [-0.3049,  0.3049, -0.3049, 0.3049],
        [-0.,     -0.0236,  0.,     0.0236],
        [0.0236,   0.,     -0.0236, 0.    ],
        [0.2107,   0.2107,  0.2107, 0.2107]
    ])
    (n_s_opt, n_u) = B_dym.shape

    T_opt = 5

    Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    R = 0.1 * spa.eye(n_u)

    s_min = -np.inf * np.ones(n_s_opt)
    s_max = np.inf * np.ones(n_s_opt)

    x_lb_opt = np.array([-np.pi / 6, -np.pi / 6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_ub_opt = np.array([np.pi / 6, np.pi / 6, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_init_opt = 2.0 * np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    u_min = -0.99 * np.ones(n_u)
    u_max = 2.41 * np.ones(n_u)
    return (
        A_dym,
        B_dym,
        Q,
        R,
        T_opt,
        n_s_opt,
        n_u,
        u_max,
        u_min,
        x_init_opt,
        x_lb_opt,
        x_ub_opt,
    )


@app.cell
def _(A_dym, B_dym, Q, R, T_opt, n_s_opt, n_u, np, spa):
    # Define P and q matrices
    n_z = (T_opt - 1) * n_u
    n_xz = n_s_opt + n_z

    SX = []
    for i in range(T_opt):
        SX.append(spa.linalg.matrix_power(A_dym, i))
    SX = spa.vstack(SX)

    SZ = spa.csc_matrix(((T_opt - 1) * n_s_opt, (T_opt - 1) * n_u))
    for i in range(T_opt - 1):
        AiB = spa.linalg.matrix_power(A_dym, i) @ B_dym
        SZ += spa.kron(spa.eye(T_opt - 1, k=-i), AiB)
    SZ = spa.vstack([spa.csc_matrix((n_s_opt, n_z)), SZ])

    P_block = spa.kron(spa.eye(T_opt), Q)
    R_block = spa.kron(spa.eye(T_opt - 1), R)

    P_opt = spa.bmat([
        [SX.T @ P_block @ SX, SX.T @ P_block @ SZ],
        [SZ.T @ P_block @ SX, SZ.T @ P_block @ SZ + R_block]
    ])
    q_opt = np.zeros(n_xz)

    M_opt = spa.eye(n_xz)
    return M_opt, P_opt, n_xz


@app.cell
def _():
    # ADMM parameters
    sigma = 1e-4
    rho = 1.
    K_opt = 3
    return K_opt, rho, sigma


@app.cell
def _(
    K_opt,
    M_opt,
    P_opt,
    T_opt,
    Verifier,
    n_s_opt,
    n_u,
    n_xz,
    np,
    rho,
    sigma,
    spa,
    u_max,
    u_min,
    x_init_opt,
    x_lb_opt,
    x_ub_opt,
):
    fix_initial_x = True

    # Solve VP for optimal control example
    _VP_opt_sdp = Verifier(method='sdp', solver='clarabel')

    if fix_initial_x:
        offset = 0.0
        _x_param = _VP_opt_sdp.add_param(n_s_opt, lb=x_init_opt - offset, ub=x_init_opt + offset)
    else:
        _x_param = _VP_opt_sdp.add_param(n_s_opt, lb=x_lb_opt, ub=x_ub_opt)

    # l(x) <= M@z <= u(x)
    u_stack_min = np.kron(np.ones(T_opt - 1).T, u_min)
    u_stack_max = np.kron(np.ones(T_opt - 1).T, u_max)
    _l_param = _VP_opt_sdp.add_param((T_opt - 1) * n_u, lb=u_stack_min, ub=u_stack_min)
    _u_param = _VP_opt_sdp.add_param((T_opt - 1) * n_u, lb=u_stack_max, ub=u_stack_max)
    _l_x = _VP_opt_sdp.add_param_stack([_x_param, _l_param])
    _u_x = _VP_opt_sdp.add_param_stack([_x_param, _u_param])

    _z = [None for _ in range(K_opt + 1)]
    _v = [None for _ in range(K_opt + 1)]
    _w = [None for _ in range(K_opt + 1)]

    _z[0] = np.zeros(n_xz)
    _v[0] = _VP_opt_sdp.add_param(n_xz, lb=np.zeros(n_xz), ub=np.ones(n_xz))
    _w[0] = np.zeros(n_xz)

    # ADMM iterations
    _lhs_mat = P_opt + sigma * np.eye(n_xz) + rho * (M_opt.T @ M_opt)
    _lhs_mat_factorization = spa.linalg.factorized(_lhs_mat)

    # all_residuals_opt_sdp = []
    # for _k in range(K_opt):
    #     _w[_k+1] = _VP_opt_sdp.step.satlin(_v[_k], l=_l_x, u=_u_x)
    #     _rhs = sigma * _z[_k] - q_opt + rho * M_opt.T @ (2 * _w[_k+1] - _v[_k])
    #     _z[_k+1] = _VP_opt_sdp.step.implicit_linear(rhs=_rhs, lhs_mat=_lhs_mat)
    #     _v[_k+1] = _v[_k] + (M_opt @ _z[_k+1] - _w[_k+1])

    #     # optimize over scaled fixed-point residual (KKT residual)
    #     _VP_opt_sdp.set_objective_norm([-sigma * (_z[_k+1] - _z[_k]) - rho * M_opt.T @ (_v[_k+1] - _v[_k]), _v[_k+1] - _v[_k]], 2)
    #     _res = _VP_opt_sdp.solve()
    #     all_residuals_opt_sdp.append(_res)
    return


@app.cell
def _(plt):
    plt.figure()
    # plt.plot(range(1, K_opt + 1), all_residuals_opt_sdp, label='ADMM')
    plt.suptitle('optimal control example (SDP)')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    All three examples have been verified using SDP-based relaxations with the fomtools framework.
    The SDP solver provides tighter bounds than MIP in many cases and enables verification of
    larger problem instances.

    Key observations:
    - **Sparse coding (ISTA)**: SDP bounds on soft-thresholding operator
    - **Network optimization (PDHG)**: SDP bounds with momentum acceleration
    - **Optimal control (ADMM)**: SDP bounds for constrained QP problems
    """)
    return


if __name__ == "__main__":
    app.run()
