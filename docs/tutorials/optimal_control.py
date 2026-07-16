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
    from scipy.sparse import csr_matrix

    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner
    from fomtools.proxable_functions import IndBox
    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
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
    # Problem setup: Constrained optimal control

    In this tutorial, we verify an OSQP-style ADMM method for a finite-horizon constrained optimal control problem. The system has state $s_t \in \mathbf{R}^{n_s}$, input $u_t \in \mathbf{R}^{n_u}$, and initial state parameter $x$.

    \begin{array}{lll}
        \text{minimize} & s_T^T Q_T s_T + \sum_{t=1}^{T-1} s_t^T Q s_t + u_t^T R u_t \\
        \text{subject to} & s_{t+1} = A s_t + B u_t, & t=1,\dots,T-1 \\
        & s_{\min} \le s_t \le s_{\max}, & t=1,\dots,T \\
        & u_{\min} \le u_t \le u_{\max}, & t=1,\dots,T-1 \\
        & s_1 = x,
    \end{array}

    with optimization variable $z = (u_1, \dots, u_{T-1}) \in \R^{(T-1) n_u}$ and parameter $x \in \R^{n_s}$.
    The verification problem searches over a bounded set of initial ADMM iterates and, depending on the experiment, either fixes $x$ or lets it vary over the prescribed box.
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
    u_stack_min = np.tile(u_min, T - 1)
    u_stack_max = np.tile(u_max, T - 1)

    # # input constraint
    # u_0 = 10.5916
    # u_min = 9.6*np.ones(n_u) - u_0
    # u_max = 13.*np.ones(n_u) - u_0
    return (
        A_dym,
        B_dym,
        Q,
        R,
        T,
        n_s,
        n_u,
        u_stack_max,
        u_stack_min,
        x_init,
        x_lb,
        x_ub,
    )


@app.cell
def _():
    # ADMM parameters
    sigma = 1e-4
    rho = 1.  # 0.1, 1, 10
    K = 5
    fix_initial_x = True  # False lets x vary over [x_lb, x_ub].
    return K, fix_initial_x, rho, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quadratic program form

    Now, consider the equivalent problem of the form:

    \begin{array}{ll}
        \text{minimize} & (1/2) z^T P z + q(x)^T z \\
        \text{subject to} & l(x) \le Mz \le u(x),
    \end{array}

    where $z = (u_1, \dots, u_{T-1})$. Note that for $t=0,\dots,T-1$,

    $$ s_{t+1} = A s_t + B u_t = A^t x + \sum_{m=1}^{t} A^{t-m} B u_m = A^t x + \begin{bmatrix} A^{t-1} B & \dots & B & 0 & \dots & 0 \end{bmatrix} z. $$

    In verification problem (VP) implementation, we use the optimization variable `xz_stack` defined as $(x, u_1, \dots, u_{T-1}) \in \mathbf{R}^{n_s + (T-1) n_u}$.
    This makes the initial state constraints and input box constraints part of one stacked box constraint.
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
    ## Simulator class

    We first simulate the ADMM algorithm by sampling the initial iterates
    $(z^0,v^0)$ over their prescribed boxes. The parameter $x$ is either fixed
    at $x_\text{init}$ or sampled from $[x_\mathrm{lb},x_\mathrm{ub}]$,
    according to `fix_initial_x`; the MIP and radius computation use the same
    choice.

    ### First-order method: OSQP-style ADMM

    We solve the boxed quadratic program with the splitting used by OSQP:

    \begin{array}{rcl}
        w^{k+1} &=& \Pi_{[l(x), u(x)]}(v^k), \\
        z^{k+1} &=& (P + \sigma I + \rho M^T M)^{-1}
            \left(\sigma z^k - q + \rho M^T(2w^{k+1} - v^k)\right), \\
        v^{k+1} &=& v^k + Mz^{k+1} - w^{k+1}.
    \end{array}

    To compare all verification methods in the same units, we measure the
    squared Euclidean fixed-point residual of the OSQP state:

    $$
        \left\| z^{k+1} - z^k \right\|_2^2 + \left\| v^{k+1} - v^k \right\|_2^2.
    $$
    """)
    return


@app.cell
def _(
    K,
    M,
    P,
    Verifier,
    csr_matrix,
    fix_initial_x,
    n_s,
    n_xz,
    np,
    q,
    rho,
    sigma,
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
    n_samples = 10000

    _VP = Verifier(method='simulator', obbt_solver=None)
    _rng = np.random.default_rng(0)

    if fix_initial_x:
        x_samples = np.repeat(x_init[np.newaxis, :], n_samples, axis=0)
    else:
        x_samples = _rng.uniform(
            low=x_lb,
            high=x_ub,
            size=(n_samples, n_s),
        )
    z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n_xz))
    v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n_xz))

    _x_param = _VP.add_param(ndim=n_s, samples=x_samples)

    _z = [None] * (K + 1)
    _v = [None] * (K + 1)
    _w = [None] * (K + 1)
    _z[0] = _VP.add_param(n_xz, samples=z0_samples)
    _v[0] = _VP.add_param(n_xz, samples=v0_samples)
    _w[0] = np.zeros(n_xz)

    lhs_mat_sparse = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))
    _lhs_mat_sim = lhs_mat_sparse.toarray()
    for _k in range(K):
        _w_control = _VP.step.satlin(
            _v[_k][n_s:],
            l=u_stack_min,
            u=u_stack_max,
        )
        _w[_k+1] = _VP.add_param_stack([_x_param, _w_control])
        _rhs = sigma * _z[_k] - q + rho * M.T @ (2*_w[_k+1] - _v[_k])
        _z[_k+1] = _VP.step.implicit_linear(_rhs, output_mat=_lhs_mat_sim)
        _v[_k+1] = _v[_k] + (M @ _z[_k+1] - _w[_k+1])
        _VP.set_objective_norm(
            [_z[_k+1] - _z[_k], _v[_k+1] - _v[_k]],
            ord=2,
        )

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
    return (
        lhs_mat_sparse,
        n_samples,
        simulation_residual_quantiles_admm,
        simulation_residuals_admm,
        v0_samples,
        x_samples,
        z0_samples,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the radius-conditioned PEP comparison, we estimate the OSQP-state
    initial radius
    \[
        \|z^0-z^\star\|_2^2+\|v^0-v^\star\|_2^2 \le R^2
    \]
    from the same initial-state parameter set used by the MIP. Because the
    maximum below is taken over random initial iterates, it is an empirical
    radius rather than a certificate for every point in the initialization
    boxes.
    """)
    return


@app.cell
def _(
    M,
    lhs_mat_sparse,
    n_samples,
    np,
    q,
    rho,
    sigma,
    spa,
    u_stack_max,
    u_stack_min,
    v0_samples,
    x_samples,
    z0_samples,
):
    _lhs_mat_lu = spa.linalg.splu(lhs_mat_sparse.tocsc())
    _z_star_samples = z0_samples.copy()
    _v_star_samples = v0_samples.copy()
    _l_x_samples = np.hstack([
        x_samples,
        u_stack_min[np.newaxis, :].repeat(n_samples, axis=0),
    ])
    _u_x_samples = np.hstack([
        x_samples,
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

    pepit_radius_empirical = np.max(np.linalg.norm(
        np.hstack([
            z0_samples - _z_star_samples,
            v0_samples - _v_star_samples,
        ]),
        axis=1,
    ))
    return (pepit_radius_empirical,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm verification through verification problem (VP)

    ### MIP-VP
    Maximizing a squared Euclidean norm would make the verification problem a
    nonconvex MIQP. Instead, the MIP maximizes the infinity norm of the same
    OSQP-state residual
    \[
        r^k=(z^{k+1}-z^k,\;v^{k+1}-v^k).
    \]
    Since $r^k\in\mathbf{R}^{2n_{xz}}$, the norm inequality
    \[
        \|r^k\|_2^2 \le 2n_{xz}\|r^k\|_\infty^2
    \]
    converts the exact MIP value into a valid upper bound in the squared
    Euclidean units used by PEP and the simulations.
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
    fix_initial_x,
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

    all_residuals_inf = []
    worst_case_x_param_admm = []
    worst_case_z_admm = []
    worst_case_v_admm = []
    for _k in range(K):
        # _w[_k+1] = _VP.step.satlin(_v[_k], l=_l_x, u=_u_x)
        _w[_k+1] = _VP.prox(indbox, _v[_k])
        _rhs = sigma * _z[_k] - q + rho * M.T @ (2*_w[_k+1] - _v[_k])
        _z[_k+1] = _VP.step.implicit_linear(_rhs, output_mat=_lhs_mat)
        _v[_k+1] = _v[_k] + (M @ _z[_k+1] - _w[_k+1])

        # Keep the exact verification problem linear.
        _VP.set_objective_norm(
            [_z[_k+1] - _z[_k], _v[_k+1] - _v[_k]],
            ord=np.inf,
        )
        _res = _VP.solve()
        all_residuals_inf.append(_res)
        # Store the horizon-specific worst-case initial state for this k.
        worst_case_x_param_admm.append(_VP.get_solution_value(_x_param))
        worst_case_z_admm.append(_VP.get_solution_value(_z[0]))
        worst_case_v_admm.append(_VP.get_solution_value(_v[0]))

    # Convert the infinity-norm guarantee to squared Euclidean units.
    all_residuals = 2 * n_xz * np.square(all_residuals_inf)
    print(f"infinity-norm MIP bounds: {all_residuals_inf}")
    print(f"squared-Euclidean upper bounds: {all_residuals}")
    return (
        all_residuals,
        worst_case_v_admm,
        worst_case_x_param_admm,
        worst_case_z_admm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We replay the horizon-specific infinity-norm worst-case instances in
    simulator mode and evaluate their actual squared Euclidean residual. These
    points need not maximize the Euclidean residual, but they provide explicit
    feasible trajectories below the norm-converted MIP upper bound. After
    calling `solve()`, the replay's final residual is available through the
    `final_residuals` summary statistic.
    """)
    return


@app.cell
def _(
    K,
    M,
    P,
    Verifier,
    csr_matrix,
    n_s,
    n_xz,
    np,
    q,
    rho,
    sigma,
    u_stack_max,
    u_stack_min,
    worst_case_v_admm,
    worst_case_x_param_admm,
    worst_case_z_admm,
):
    # Simulate each horizon with the corresponding worst-case initial state.
    _lhs_mat = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))
    _lhs_mat_sim = _lhs_mat.toarray()

    worst_case_residuals_admm = []
    for _k in range(1, K + 1):
        _VP = Verifier(method='simulator', obbt_solver=None)
        _x_k = np.asarray(worst_case_x_param_admm[_k - 1])
        _x_param = _VP.add_param(ndim=n_s, samples=[_x_k])

        _z = [None] * (_k + 1)
        _v = [None] * (_k + 1)
        _w = [None] * (_k + 1)
        _z0_k = np.asarray(worst_case_z_admm[_k - 1])
        _v0_k = np.asarray(worst_case_v_admm[_k - 1])
        _z[0] = _VP.add_iterate(ndim=n_xz, init=_z0_k)
        _v[0] = _VP.add_iterate(ndim=n_xz, init=_v0_k)
        _w[0] = np.zeros(n_xz)

        for _j in range(_k):
            _w_control = _VP.step.satlin(
                _v[_j][n_s:],
                l=u_stack_min,
                u=u_stack_max,
            )
            _w[_j+1] = _VP.add_param_stack([_x_param, _w_control])
            _rhs = sigma * _z[_j] - q + rho * M.T @ (2*_w[_j+1] - _v[_j])
            _z[_j+1] = _VP.step.implicit_linear(_rhs, output_mat=_lhs_mat_sim)
            _v[_j+1] = _v[_j] + (M @ _z[_j+1] - _w[_j+1])
            _VP.set_objective_norm(
                [_z[_j+1] - _z[_j], _v[_j+1] - _v[_j]],
                ord=2,
            )

        _VP.solve()
        # Each replay contains one sampled instance.
        _final_residual = float(_VP['final_residuals'][0])
        worst_case_residuals_admm.append(_final_residual)

    worst_case_residuals_admm = np.array(worst_case_residuals_admm)
    return (worst_case_residuals_admm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    We compare the MIP result with a PEP that uses the same OSQP recurrence,
    including the projection-first ordering and the parameters $\sigma$ and
    $\rho$. For $M=I$, the linear solve is the proximal step
    \[
        z^{k+1}
        = \operatorname{prox}_{f/(\sigma+\rho)}
        \left(
            \frac{\sigma z^k+\rho(2w^{k+1}-v^k)}
                 {\sigma+\rho}
        \right).
    \]

    PEP still considers the larger class of all $L$-smooth,
    $\mu$-strongly-convex functions and all convex indicator sets:

    \begin{array}{ll}
        \textnormal{minimize} & f(x) + g(x),
    \end{array}

    so it does not retain the exact matrix $P$, dimension, or box geometry. We
    show both the bound conditioned on the empirical radius above and a
    rate-shape curve calibrated to match the norm-converted MIP upper bound at
    $K=1$. The calibrated curve is not an independent absolute guarantee.
    """)
    return


@app.cell
def _(
    K,
    P,
    PEPitRunner,
    all_residuals,
    np,
    pepit_radius_empirical,
    rho,
    sigma,
):
    _P_eigs = np.linalg.eigvalsh(P.toarray())
    _mu = float(_P_eigs.min())
    _L = float(_P_eigs.max())
    _runner = PEPitRunner()
    all_residuals_PEP_radius, _ = _runner.osqp_admm_residuals(
        K=K,
        mu=_mu,
        L=_L,
        sigma=sigma,
        rho=rho,
        R=pepit_radius_empirical,
    )
    all_residuals_PEP, pepit_radius_calibrated = _runner.osqp_admm_residuals(
        K=K,
        mu=_mu,
        L=_L,
        sigma=sigma,
        rho=rho,
        calibrate_to=all_residuals[0],
    )
    return all_residuals_PEP, all_residuals_PEP_radius


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical Result

    We plot every curve in squared Euclidean units. The MIP curve uses the
    dimension-dependent norm conversion above; simulation, replay, and PEP use
    the squared Euclidean residual directly.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    all_residuals_PEP_radius,
    plt,
    simulation_residual_quantiles_admm,
    simulation_residuals_admm,
    worst_case_residuals_admm,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='ADMM (PEPit, matched at K=1)')
    plt.plot(
        _ks,
        all_residuals_PEP_radius,
        ':',
        label='ADMM (PEPit, empirical-radius conditioned)',
    )
    plt.plot(
        _ks,
        all_residuals,
        label=r'ADMM (MIP upper bound via $2n_{xz}\|\cdot\|_\infty^2$)',
    )
    plt.plot(_ks, simulation_residuals_admm, '--', label='ADMM (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_admm[2], simulation_residual_quantiles_admm[0], alpha=.18, label='ADMM simulation 10%-90% band')
    plt.plot(
        _ks,
        worst_case_residuals_admm,
        '-.',
        color='k',
        linewidth=.7,
        label=r'$\ell_\infty$ worst-case instance (actual squared $\ell_2$)',
    )
    plt.suptitle('optimal control example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel(r'Worst-case squared Euclidean fixed-point residual')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
