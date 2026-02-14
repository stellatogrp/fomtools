import marimo

__generated_with = "0.14.16"
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
    return Verifier, np, plt, spa


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Optimal control problem:

    \begin{array}{lll}
        \text{minimize} & s_T^T Q_T s_T + \sum_{t=1}^{T-1} s_t^T Q s_t + u_t^T R u_t \\
        \text{subject to} & s_{t+1} = A s_t + B u_t, & t=1,\dots,T-1 \\
        & s_{\min} \le s_t \le s_{\max}, & t=1,\dots,T \\
        & u_{\min} \le u_t \le u_{\max}, & t=1,\dots,T-1 \\
        & s_1 = x,
    \end{array}

    with optimization variable $z = (u_1, \dots, u_{T-1}) \in \R^{(T-1) n_u}$ and parameter $x \in \R^{n_s}$.
    """
    )
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
    mo.md(
        r"""
    Now, consider the equivalent problem of the form:

    \begin{array}{ll}
        \text{minimize} & (1/2) z^T P z + q(x)^T z \\
        \text{subject to} & l(x) \le Mz \le u(x),
    \end{array}

    where $z = (u_1, \dots, u_{T-1})$. Note that for $t=0,\dots,T-1$,

    $$ s_{t+1} = A s_t + B u_t = A^t x + \sum_{m=1}^{t} A^{t-m} B u_m = A^t x + \begin{bmatrix} A^{t-1} B & \dots & B & 0 & \dots & 0 \end{bmatrix} z. $$

    In verification problem (VP) implementation, we use the optimization variable `xz_stack` defined as $(x, u_1, \dots, u_{T-1}) \in \mathbf{R}^{n_s + (T-1) n_u}$.
    """
    )
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
    x_lb,
    x_ub,
):
    # Solve VP for optimal control example
    fix_initial_x = True # True if x is fixed, False if x is in set X

    solver_params = {
        'OutputFlag': 0,
    }
    VP = Verifier(solver='highs', params=solver_params)

    if fix_initial_x :
        offset = 0.0
        x_param = VP.add_param(n_s, lb=x_init-offset, ub=x_init+offset)
    else :
        x_param = VP.add_param(n_s, lb=x_lb, ub=x_ub)

    # l(x) <= M@z <= u(x)
    l_param = VP.add_param((T-1)*n_u, lb=u_stack_min, ub=u_stack_min)
    u_param = VP.add_param((T-1)*n_u, lb=u_stack_max, ub=u_stack_max)
    l_x = VP.add_param_stack([x_param, l_param])
    u_x = VP.add_param_stack([x_param, u_param])

    z = [None for _ in range(K+1)]
    v = [None for _ in range(K+1)]
    w = [None for _ in range(K+1)]

    z[0] = np.zeros(n_xz)
    v[0] = VP.add_param(n_xz, lb=np.zeros(n_xz), ub=np.ones(n_xz))
    w[0] = np.zeros(n_xz)

    # Box constraint
    # from fomtools.proxable_functions import IndBox
    # indbox = IndBox(l=l_x, u=u_x)

    # ADMM (OSQP) iterations
    lhs_mat = P + sigma*np.eye(n_xz) + rho*(M.T@M)
    lhs_mat_factorization = spa.linalg.factorized(lhs_mat)

    all_residuals = []
    for k in range(K):
        # w[k+1] = VP.step.satlin(v[k], l=l_x, u=u_x)
        w[k+1] = VP.step.satlin(v[k], l=u_x, u=l_x)
        # w[k+1] = VP.prox(function=indbox, rhs=v[k])
        rhs = sigma*z[k] - q + rho*M.T@(2*w[k+1]-v[k])
        z[k+1] = VP.step.implicit_linear(rhs=rhs, lhs_mat=lhs_mat, lhs_mat_factorization=lhs_mat_factorization)
        v[k+1] = v[k] + M@z[k+1] - w[k+1]
        VP.set_infinity_norm_objective([z[k+1] - z[k]])
        res = VP.solve()
        all_residuals.append(res)

    print(all_residuals)
    return (all_residuals,)


@app.cell
def _(K, all_residuals, plt):
    plt.plot(range(1, K+1), all_residuals, label='VP')
    plt.suptitle('optimal control example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
