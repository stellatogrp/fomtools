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

    from fomtools.verifier import Verifier
    from fomtools import LinearExpression
    import scipy.sparse as spa
    from scipy.sparse import csr_matrix

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return Verifier, csr_matrix, np, plt, spa


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

    T = 5        # finite horizon t=1,2,...,T

    Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.]) # state cost matrix Q_t = Q_T = Q
    R = 0.1*spa.eye(n_u) # input cost matrix

    s_min = -np.inf*np.ones(n_s)
    s_max =  np.inf*np.ones(n_s)

    # initial state s_init as parameter x: the set X = [x_lb, x_ub]
    x_lb = np.array([-np.pi/6, -np.pi/6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_ub = np.array([ np.pi/6,  np.pi/6, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_init = 2.0*np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # initial x when we fix it

    # input constraint
    u_min = -0.99*np.ones(n_u)
    u_max =  2.41*np.ones(n_u)
    # u_max = -2.41*np.ones(n_u)

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
    return M, P, n_xz, q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate the ADMM algorithm on given problem instances by randomly sampling the fixed initial state parameter $x_\text{init}$ around its nominal value.
    The output will be residuals computed for each iteration $k=1,\ldots,K$.
    """)
    return


@app.cell
def _(
    K,
    M,
    P,
    T,
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
    x_init,
):
    # Simulate ADMM on sampled problem instances
    _n_samples = 100
    _rng = np.random.default_rng(0)
    _offset_range = 0.1
    _x_samples = _rng.uniform(low=x_init - _offset_range, high=x_init + _offset_range, size=(_n_samples, n_s))

    # Convert to xz_stack format: [x; 0_{n_z}]
    _xz_stacks = np.hstack([_x_samples, np.zeros((_n_samples, (T-1)*n_u))])

    # ADMM iteration over samples (batched)
    _z_batch = np.zeros((_n_samples, n_xz), dtype=float)
    _v_batch = np.zeros((_n_samples, n_xz), dtype=float)
    _w_batch = np.zeros((_n_samples, n_xz), dtype=float)

    _lhs_mat = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))
    _lhs_mat_lu = spa.linalg.splu(_lhs_mat.tocsc())

    _residuals_sim = np.empty((_n_samples, K), dtype=float)

    for k in range(K):
        # satlin: clip w to [l_x, u_x]
        _l_x = np.hstack([_x_samples, u_stack_min[np.newaxis, :].repeat(_n_samples, axis=0)])
        _u_x = np.hstack([_x_samples, u_stack_max[np.newaxis, :].repeat(_n_samples, axis=0)])
        _w_batch = np.clip(_v_batch, _l_x, _u_x)

        # implicit_linear: solve for z
        _rhs_batch = sigma * _z_batch - q[np.newaxis, :] + rho * (2*_w_batch - _v_batch) @ M
        _z_new_batch = np.array([_lhs_mat_lu.solve(_rhs_batch[i]) for i in range(_n_samples)])

        # v update
        _v_new_batch = _v_batch + _z_new_batch @ M - _w_batch

        # residual: scaled fixed-point residual
        _residual_1 = -sigma*(_z_new_batch - _z_batch) - rho*(_v_new_batch - _v_batch) @ M
        _residual_2 = _v_new_batch - _v_batch
        _residuals_sim[:, k] = np.max(np.abs(np.hstack([_residual_1, _residual_2])), axis=1)

        _z_batch = _z_new_batch
        _v_batch = _v_new_batch

    simulation_residuals_admm = np.mean(_residuals_sim, axis=0)
    return (simulation_residuals_admm,)


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
    problem.solve(verbose=True)

    print(problem.status, problem.value)
    return u_stack_max, u_stack_min


@app.cell
def _(
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
    u_stack_max,
    u_stack_min,
    x_init,
    x_lb,
    x_ub,
):
    fix_initial_x = True # True if x is fixed, False if x is in set X

    solver_params = {
        'OutputFlag': 0,
    }

    # Solve VP for optimal control example
    _VP = Verifier(solver='gurobi', params=solver_params)

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

    _z[0] = np.zeros(n_xz)
    _v[0] = _VP.add_param(n_xz, lb=np.zeros(n_xz), ub=np.ones(n_xz))
    _w[0] = np.zeros(n_xz)

    # Box constraint
    # from fomtools.proxable_functions import IndBox
    # indbox = IndBox(l=l_x, u=u_x)

    # ADMM (OSQP) iterations
    _lhs_mat = csr_matrix(P + sigma*np.eye(n_xz) + rho*(M.T@M))

    all_residuals = []
    for _k in range(K):
        _w[_k+1] = _VP.step.satlin(_v[_k], l=_l_x, u=_u_x)
        # _w[_k+1] = _VP.prox(function=indbox, rhs=_v[_k])
        _rhs = sigma * _z[_k] - q + rho * M.T @ (2*_w[_k+1] - _v[_k])
        _z[_k+1] = _VP.step.implicit_linear(rhs=_rhs, lhs_mat=_lhs_mat)
        _v[_k+1] = _v[_k] + (M @ _z[_k+1] - _w[_k+1])

        # optimize over scaled fixed-point residual (KKT residual)
        _VP.set_objective_norm([-sigma*(_z[_k+1]-_z[_k]) - rho*M.T@(_v[_k+1]-_v[_k]), _v[_k+1]-_v[_k]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)

    print(all_residuals)
    return (all_residuals,)


@app.cell
def _(K, all_residuals, plt, simulation_residuals_admm):
    plt.plot(range(1, K+1), all_residuals, label='ADMM (MIP worst-case)')
    plt.plot(range(1, K+1), simulation_residuals_admm, '--', label='ADMM (in-sample avg)')
    plt.suptitle('optimal control example (MIP)')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compare our results to the worst-case performance of the algorithm returned by the performance estimation problem (PEP), using the package [`PEPit`](https://pepit.readthedocs.io/en/0.5.1/). Specifically, we measure the performance of ADMM solving

    \begin{array}{ll}
        \textnormal{minimize} & f(x) + g(x),
    \end{array}

    where $f$ is a smooth strongly-convex function and $g$ is an indicator function to a convex set.
    """)
    return


@app.cell
def _(K, P, np, spa):
    from PEPit import PEP, Point
    from PEPit.operators import LinearOperator
    from PEPit.functions import SmoothStronglyConvexFunction, ConvexIndicatorFunction
    from PEPit.primitive_steps import proximal_step

    P_eigs, _ = spa.linalg.eigs(P)
    P_eigs = np.real(P_eigs)
    mu = np.min(P_eigs)
    L = np.max(P_eigs)
    print(mu, L)

    all_residuals_PEP = []
    for _k_PEP in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        _g = _problem.declare_function(ConvexIndicatorFunction)
        _F = _f + _g

        # Define optimal solution xs
        _xs = _F.stationary_point()

        # Declare initial point x0
        _u0 = _problem.set_initial_point()

        _x = [None for _ in range(K+1)]
        _y = [None for _ in range(K+1)]
        _u = [None for _ in range(K+1)]
        _u[0] = _u0
        _x[0], _, _ = proximal_step(_u[0], _f, 1.)
        _y[0], _, _ = proximal_step(2*_x[0] - _u[0], _g, 1.)

        for _k in range(_k_PEP):
            _u[_k+1] = _u[_k] + (_y[_k] - _x[_k])
            _x[_k+1], _, _ = proximal_step(_u[_k+1], _f, 1.)
            _y[_k+1], _, _ = proximal_step(2*_x[_k+1] - _u[_k+1], _g, 1.)

        _problem.set_initial_condition((_x[0] - _xs)**2 <= 1.0)
        _problem.set_initial_condition((_u0 - _xs)**2 <= 1.0)

        _problem.set_performance_metric((_x[_k_PEP] - _x[_k_PEP-1])**2 + (_u[_k_PEP] - _u[_k_PEP-1])**2)
        _res = _problem.solve(solver='mosek')
        all_residuals_PEP.append(_res) # account for L-infinity norm of VP
    return (all_residuals_PEP,)


@app.cell
def _(K, all_residuals_PEP, plt):
    H_norm = 14.14213564
    r = 8.31227985242667

    plt.figure()
    plt.plot(range(1, K+1), all_residuals_PEP, label='OSQP (ADMM)')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.legend()
    plt.suptitle('optimal control example (PEP)')
    return


if __name__ == "__main__":
    app.run()
