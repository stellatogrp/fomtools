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
    from pepit_runner import PEPitRunner
    from scipy.sparse import csr_matrix

    from fomtools.proxable_functions import IndBox
    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
    }
    return (
        IndBox,
        PEPitRunner,
        Verifier,
        csr_matrix,
        np,
        plt,
        solver_params,
        spa,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optimal control problem:

    \begin{array}{lll}
        \text{minimize} & s_T^T Q_T s_T + \sum_{t=1}^{T-1} s_t^T Q s_t + u_t^T R u_t \\
        \text{subject to} & s_{t+1} = A s_t + B u_t, & t=1,\dots,T-1 \\
        & s_{\min} \le s_t \le s_{\max}, & t=1,\dots,T \\
        & u_{\min} \le u_t \le u_{\max}, & t=1,\dots,T-1 \\
        & s_1 = x,
    \end{array}

    with optimization variable $z = (u_1, \dots, u_{T-1}) \in \R^{(T-1) n_u}$ and parameter $x \in \R^{n_s}$.
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
            ]) # A_dym.shape = (n_s, n_s)
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
            ]) # B_dym.shape = (n_s, n_u)
    (n_s, n_u) = B_dym.shape

    T = 3        # finite horizon t=1,2,...,T

    Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.]) # state cost matrix Q_t = Q_T = Q
    R = 0.1*spa.eye(n_u) # input cost matrix

    # State constraints are unconstrained in this tutorial.

    # initial state s_init as parameter x: the set X = [x_lb, x_ub]
    x_lb = np.array([-np.pi/6, -np.pi/6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_ub = np.array([ np.pi/6,  np.pi/6, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_init = 2.0*np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # initial x when we fix it

    # input constraint
    u_min = -0.99*np.ones(n_u)
    u_max =  2.41*np.ones(n_u)

    # # input constraint
    # u_0 = 10.5916
    # u_min = 9.6*np.ones(n_u) - u_0
    # u_max = 13.*np.ones(n_u) - u_0
    return A_dym, B_dym, Q, R, T, n_s, n_u, u_max, u_min, x_init, x_lb, x_ub


@app.cell
def _():
    # ADMM parameters
    sigma = 1e-4
    rho = 1.  # 0.1, 1, 10
    K = 5
    return K, rho, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, consider the equivalent problem of the form:

    \begin{array}{ll}
        \text{minimize} & (1/2) z^T P z + q(x)^T z \\
        \text{subject to} & l(x) \le Mz \le u(x),
    \end{array}

    where $z = (u_1, \dots, u_{T-1})$. Note that for $t=0,\dots,T-1$,

    $$ s_{t+1} = A s_t + B u_t = A^t x + \sum_{m=1}^{t} A^{t-m} B u_m = A^t x + \begin{bmatrix} A^{t-1} B & \dots & B & 0 & \dots & 0 \end{bmatrix} z. $$

    In verification problem (VP) implementation, we use the optimization variable `xz_stack` defined as $(x, u_1, \dots, u_{T-1}) \in \mathbf{R}^{n_s + (T-1) n_u}$.
    """)
    return


@app.cell
def _(A_dym, B_dym, Q, R, T, n_s, n_u, np, spa):
    # Define state s = (s_1, ..., s_T) in terms of z = (u_1, ..., u_{T-1}): s = SX@x_param + SZ@z
    n_z = (T-1)*n_u # z dimension
    n_xz = n_s + n_z # xz_stack dimension

    SX = [] # linear part of x in state s
    for i in range(T):
        SX.append(spa.linalg.matrix_power(A_dym, i))
    SX = spa.vstack(SX)

    SZ = spa.csc_matrix(((T-1)*n_s, (T-1)*n_u)) # linear part of z in state s
    for i in range(T-1):
        AiB = spa.linalg.matrix_power(A_dym, i) @ B_dym
        SZ += spa.kron(spa.eye(T-1, k=-i), AiB)
    SZ = spa.vstack([spa.csc_matrix((n_s, n_z)), SZ])

    P_block = spa.kron(spa.eye(T), Q)
    R_block = spa.kron(spa.eye(T-1), R)

    # Define P and q in objective: minimize (1/2)z.T@P@z + q.T@z
    P = spa.bmat([
            [SX.T @ P_block @ SX, SX.T @ P_block @ SZ],
            [SZ.T @ P_block @ SX, SZ.T @ P_block @ SZ + R_block]
        ])
    q = np.zeros(n_xz)

    # Define M
    M = spa.eye(n_xz)
    z0_lb, z0_ub = -np.ones(n_xz), np.ones(n_xz)
    v0_lb, v0_ub = np.zeros(n_xz), np.ones(n_xz)
    return M, P, n_xz, q, v0_lb, v0_ub, z0_lb, z0_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate the ADMM algorithm on given problem instances by randomly sampling the fixed initial state parameter $x_\text{init}$ around its nominal value and the initial iterates $(z^0, v^0)$ over their prescribed boxes.
    The output will be residuals computed for each iteration $k=1,\ldots,K$.
    """)
    return


@app.cell
def _(
    IndBox,
    K,
    M,
    P,
    T,
    Verifier,
    csr_matrix,
    n_s,
    n_u,
    n_xz,
    np,
    q,
    rho,
    sigma,
    spa,
    u_stack_max,
    u_stack_min,
    v0_lb,
    v0_ub,
    x_init,
    z0_lb,
    z0_ub,
):
    n_samples = 10000

    _VP = Verifier(method='simulator', obbt_solver=None)
    _rng = np.random.default_rng(0)

    _offset_range = 0.1
    _x_samples = _rng.uniform(
        low=x_init - _offset_range,
        high=x_init + _offset_range,
        size=(n_samples, n_s),
    )
    _z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n_xz))
    _v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n_xz))

    _x_param = _VP.add_param(ndim=n_s, samples=_x_samples)
    _x_param.lb, _x_param.ub = x_init - _offset_range, x_init + _offset_range
    _l_param = _VP.add_param((T-1)*n_u, lb=u_stack_min, ub=u_stack_min)
    _u_param = _VP.add_param((T-1)*n_u, lb=u_stack_max, ub=u_stack_max)
    _l_x = _VP.add_param_stack([_x_param, _l_param])
    _u_x = _VP.add_param_stack([_x_param, _u_param])

    _z = [None] * (K + 1)
    _v = [None] * (K + 1)
    _w = [None] * (K + 1)
    _z[0] = _VP.add_param(n_xz, samples=_z0_samples)
    _z[0].lb, _z[0].ub = z0_lb, z0_ub
    _v[0] = _VP.add_param(n_xz, samples=_v0_samples)
    _v[0].lb, _v[0].ub = v0_lb, v0_ub
    _w[0] = np.zeros(n_xz)

    _indbox = IndBox(l=_l_x, u=_u_x)
    _lhs_mat_sparse = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))
    _lhs_mat_sim = _lhs_mat_sparse.toarray()
    for _k in range(K):
        _w[_k+1] = _VP.prox(function=_indbox, rhs=_v[_k])
        _rhs = sigma * _z[_k] - q + rho * M.T @ (2*_w[_k+1] - _v[_k])
        _z[_k+1] = _VP.step.implicit_linear(rhs=_rhs, lhs_mat=_lhs_mat_sim)
        _v[_k+1] = _v[_k] + (M @ _z[_k+1] - _w[_k+1])
        _VP.set_objective_norm([
            -sigma*(_z[_k+1]-_z[_k]) - rho*M.T@(_v[_k+1]-_v[_k]),
            _v[_k+1]-_v[_k],
        ], np.inf)

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

    _lhs_mat_lu = spa.linalg.splu(_lhs_mat_sparse.tocsc())
    _z_star_samples = _z0_samples.copy()
    _v_star_samples = _v0_samples.copy()
    _l_x_samples = np.hstack([
        _x_samples,
        u_stack_min[np.newaxis, :].repeat(n_samples, axis=0),
    ])
    _u_x_samples = np.hstack([
        _x_samples,
        u_stack_max[np.newaxis, :].repeat(n_samples, axis=0),
    ])
    for _ in range(2000):
        _w_star_samples = np.clip(_v_star_samples, _l_x_samples, _u_x_samples)
        _rhs_samples = sigma * _z_star_samples - q[np.newaxis, :] + rho * (
            2*_w_star_samples - _v_star_samples
        ) @ M
        _z_next_samples = _lhs_mat_lu.solve(_rhs_samples.T).T
        _v_next_samples = _v_star_samples + _z_next_samples @ M - _w_star_samples
        _step_norm = np.max(np.linalg.norm(
            np.hstack([
                _z_next_samples - _z_star_samples,
                _v_next_samples - _v_star_samples,
            ]),
            axis=1,
        ))
        _z_star_samples = _z_next_samples
        _v_star_samples = _v_next_samples
        if _step_norm <= 1e-10:
            break
    pepit_radius = np.max(np.linalg.norm(
        np.hstack([
            _z0_samples - _z_star_samples,
            _v0_samples - _v_star_samples,
        ]),
        axis=1,
    ))
    return (
        pepit_radius,
        simulation_residual_quantiles_admm,
        simulation_residuals_admm,
    )


@app.cell
def _(P, T, n_u, np, q, u_max, u_min, x_init):
    # Define and solve this problem with CVXPY
    import cvxpy as cp

    # state s = SX@x_param + SZ@z where z = u_stack = (u_1, ..., u_{T-1})
    u_stack = cp.Variable((T-1)*n_u)
    u_stack_min = np.kron(np.ones(T-1).T, u_min)
    u_stack_max = np.kron(np.ones(T-1).T, u_max)

    # parameter x is fixed as x_init
    x = x_init

    # define constraints
    constraints = [
        u_stack_min <= u_stack,
        u_stack <= u_stack_max,
    ]

    x_u_stack = cp.hstack([x, u_stack])

    # define objective and solve problem
    problem = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x_u_stack, P) + q.T@x_u_stack), constraints=constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    print(problem.status, problem.value)
    return u_stack_max, u_stack_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now obtain worst-case guarantee by solving the corresponding verification problem using MIP solver.
    """)
    return


@app.cell
def _(
    IndBox,
    K,
    M,
    P,
    T,
    Verifier,
    csr_matrix,
    n_s,
    n_u,
    n_xz,
    np,
    q,
    rho,
    sigma,
    solver_params,
    u_stack_max,
    u_stack_min,
    v0_lb,
    v0_ub,
    x_init,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    fix_initial_x = True # True if x is fixed, False if x is in set X

    # Solve VP for optimal control example
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)

    if fix_initial_x :
        offset = 0.0
        _x_param = _VP.add_param(n_s, lb=x_init-offset, ub=x_init+offset)
    else :
        _x_param = _VP.add_param(n_s, lb=x_lb, ub=x_ub)

    # l(x) <= M@z <= u(x)
    _l_param = _VP.add_param((T-1)*n_u, lb=u_stack_min, ub=u_stack_min)
    _u_param = _VP.add_param((T-1)*n_u, lb=u_stack_max, ub=u_stack_max)
    _l_x = _VP.add_param_stack([_x_param, _l_param])
    _u_x = _VP.add_param_stack([_x_param, _u_param])

    _z = [None for _ in range(K+1)]
    _v = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]

    _z[0] = _VP.add_iterate(n_xz, lb=z0_lb, ub=z0_ub)
    _v[0] = _VP.add_iterate(n_xz, lb=v0_lb, ub=v0_ub)
    _w[0] = np.zeros(n_xz)

    # ADMM (OSQP) iterations
    indbox = IndBox(l=_l_x, u=_u_x)
    _lhs_mat = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))

    all_residuals = []
    worst_case_x_param_admm = []
    worst_case_z_admm = []
    worst_case_v_admm = []
    for _k in range(K):
        # _w[_k+1] = _VP.step.satlin(_v[_k], l=_l_x, u=_u_x)
        _w[_k+1] = _VP.prox(function=indbox, rhs=_v[_k])
        _rhs = sigma * _z[_k] - q + rho * M.T @ (2*_w[_k+1] - _v[_k])
        _z[_k+1] = _VP.step.implicit_linear(rhs=_rhs, lhs_mat=_lhs_mat)
        _v[_k+1] = _v[_k] + (M @ _z[_k+1] - _w[_k+1])

        # optimize over scaled fixed-point residual (KKT residual)
        _VP.set_objective_norm([-sigma*(_z[_k+1]-_z[_k]) - rho*M.T@(_v[_k+1]-_v[_k]), _v[_k+1]-_v[_k]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)
        # Store the horizon-specific worst-case initial state for this k.
        worst_case_x_param_admm.append(_VP.get_solution_value(_x_param))
        worst_case_z_admm.append(_VP.get_solution_value(_z[0]))
        worst_case_v_admm.append(_VP.get_solution_value(_v[0]))

    print(all_residuals)
    return (
        all_residuals,
        worst_case_v_admm,
        worst_case_x_param_admm,
        worst_case_z_admm,
    )


@app.cell
def _(
    K,
    M,
    P,
    csr_matrix,
    n_xz,
    np,
    q,
    rho,
    sigma,
    spa,
    u_stack_max,
    u_stack_min,
    worst_case_v_admm,
    worst_case_x_param_admm,
    worst_case_z_admm,
):
    # Simulate each horizon with the corresponding worst-case initial state.
    _lhs_mat = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))
    _lhs_mat_lu = spa.linalg.splu(_lhs_mat.tocsc())

    worst_case_residuals_admm = []
    for _k in range(1, K + 1):
        _x_k = np.array(worst_case_x_param_admm[_k-1], dtype=float)
        _z = np.array(worst_case_z_admm[_k-1], dtype=float)
        _v = np.array(worst_case_v_admm[_k-1], dtype=float)
        _last_residual = 0.0

        for _j in range(_k):
            _l_x = np.hstack([_x_k, u_stack_min])
            _u_x = np.hstack([_x_k, u_stack_max])
            _w = np.clip(_v, _l_x, _u_x)

            _rhs = sigma * _z - q + rho * (2 * _w - _v) @ M
            _z_new = _lhs_mat_lu.solve(_rhs)
            _v_new = _v + _z_new @ M - _w

            _residual_1 = -sigma * (_z_new - _z) - rho * (_v_new - _v) @ M
            _residual_2 = _v_new - _v
            _last_residual = np.max(np.abs(np.hstack([_residual_1, _residual_2])))

            _z = _z_new
            _v = _v_new

        worst_case_residuals_admm.append(_last_residual)

    worst_case_residuals_admm = np.array(worst_case_residuals_admm)
    return (worst_case_residuals_admm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compare our results to the worst-case performance of the algorithm returned
    by the performance estimation problem (PEP), using the shared `PEPitRunner`.
    Specifically, we measure the performance of ADMM solving

    \begin{array}{ll}
        \textnormal{minimize} & f(x) + g(x),
    \end{array}

    where $f$ is a smooth strongly-convex function and $g$ is an indicator function to a convex set.
    """)
    return


@app.cell
def _(K, P, PEPitRunner, np, pepit_radius, spa):
    _P_eigs = np.real(spa.linalg.eigs(P)[0])
    _positive_eigs = _P_eigs[_P_eigs > 1e-10]
    _mu = float(np.min(_positive_eigs)) if len(_positive_eigs) else 0.0
    _L = float(np.max(_P_eigs))
    all_residuals_PEP, _ = PEPitRunner().admm_residuals(
        K=K, mu=_mu, L=_L, R=pepit_radius)
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
    simulation_residual_quantiles_admm,
    simulation_residuals_admm,
    worst_case_residuals_admm,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='ADMM (PEPit)')
    plt.plot(_ks, all_residuals, label='ADMM (MIP worst-case)')
    plt.plot(_ks, simulation_residuals_admm, '--', label='ADMM (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_admm[2], simulation_residual_quantiles_admm[0], alpha=.18, label='ADMM simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals_admm, '-.', color='k', linewidth=.7, label='worst-case ADMM instance')
    plt.suptitle('optimal control example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
