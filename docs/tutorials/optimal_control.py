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

    from algoverify.verifier import Verifier
    from algoverify import LinearExpression
    import scipy.sparse as spa
    from scipy.sparse import csr_matrix

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return LinearExpression, Verifier, csr_matrix, np, spa


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
    # problem data
    # A_dym = np.zeros((n_s, n_s))
    # B_dym = np.zeros((n_s, n_u))
    # n_s = 12     # dimension of state
    # n_u = 10     # dimension of input
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
    (n_s, n_u) = B_dym.shape # n_s: state dimension, n_u: input dimension
    T = 5        # finite horizon t=1,2,...,T

    Q = spa.diags([0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5]) # state cost matrix Q_T
    R = 0.1*spa.eye(n_u) # input cost matrix

    s_min = -np.inf*np.ones(n_s)
    s_max =  np.inf*np.ones(n_s)

    # initial state s_init as parameter x
    x_lb = np.array([-np.pi/6, -np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_ub = np.array([ np.pi/6,  np.pi/6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # input constraint
    u_min = -0.99*np.ones(n_u)
    u_max =  2.41*np.ones(n_u)
    return A_dym, B_dym, Q, R, T, n_s, n_u, u_max, u_min, x_lb, x_ub


@app.cell
def _():
    # ADMM parameters
    sigma = 1e-4
    rho = 0.1  # 0.1, 1, 10
    K = 10
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
    """
    )
    return


@app.cell
def _(
    A_dym,
    B_dym,
    Q,
    R,
    T,
    n_s,
    n_u,
    np,
    rho,
    sigma,
    spa,
    u_max,
    u_min,
    x_lb,
    x_ub,
):
    # Define state s in terms of z: s = SX@x_param + SZ@z
    SX = [] # linear part of x in state s
    for i in range(T):
        SX.append(spa.linalg.matrix_power(A_dym, i))
    SX = spa.vstack(SX)

    SZ = spa.csc_matrix(((T-1) * n_s, (T-1) * n_u)) # linear part of z in state s
    for i in range(T-1):
        AiB = spa.linalg.matrix_power(A_dym, i) @ B_dym
        SZ += spa.kron(spa.eye(T-1, k=-i), AiB)
    SZ = spa.vstack([spa.csc_matrix((n_s, (T-1) * n_u)), SZ])

    P_block = spa.kron(spa.eye(T), Q)
    R_block = spa.kron(spa.eye(T-1), R)

    # Define P and q in objective: minimize (1/2)z.T@P@z + q.T@z
    P = spa.bmat([
            [SX.T @ P_block @ SX, SX.T @ P_block @ SZ],
            [SZ.T @ P_block @ SX, SZ.T @ P_block @ SZ + R_block]
        ])
    q = np.zeros(n_s + (T-1)*n_u)

    # Define M
    M = spa.eye(n_s + (T-1)*n_u)
    lhs_mat = P + sigma*np.eye(n_s + (T-1)*n_u) + rho*(M.T@M)
    lhs_mat_factorization = spa.linalg.factorized(lhs_mat)

    # Define l_x and u_x
    l_x = np.hstack([x_lb, np.kron(np.ones(T-1).T, u_min)])
    u_x = np.hstack([x_ub, np.kron(np.ones(T-1).T, u_max)])
    return M, l_x, lhs_mat, lhs_mat_factorization, q, u_x


@app.cell
def _(
    K,
    M,
    T,
    Verifier,
    l_x,
    lhs_mat,
    lhs_mat_factorization,
    n_s,
    n_u,
    np,
    q,
    rho,
    sigma,
    u_x,
    x_lb,
    x_ub,
):
    solver_params = {
        'OutputFlag': 0,
    }

    # VP = Verifier(solver='gurobi', params=solver_params)
    VP = Verifier(solver='highs', params=solver_params, obbt_handler='gurobi')
    x_param = VP.add_param(n_s, lb=x_lb, ub=x_ub)

    z = [None for _ in range(K+1)]
    v = [None for _ in range(K+1)]
    w = [None for _ in range(K+1)]

    n_z = n_s + (T-1)*n_u
    z[0] = np.zeros(n_z)
    # v[0] = LinearExpression.from_int(ndim=n_z, lb=np.zeros(n_z), ub=np.zeros(n_z)) # error in satlin if lb = 0.0, as bound_propagation requires lb: np.array
    v[0] = VP.add_param(n_z, lb=np.zeros(n_z), ub=np.ones(n_z))
    w[0] = np.zeros(n_z)

    # ADMM (OSQP) iterations
    all_residuals = []
    for k in range(K):
        w[k+1] = VP.step.satlin(v[k], l=l_x, u=u_x) # bound tightening fails
        rhs = sigma*z[k] - q + rho*M.T@(2*w[k+1]-v[k])
        z[k+1] = VP.step.implicit_linear(rhs=rhs, lhs_mat=lhs_mat, lhs_mat_factorization=lhs_mat_factorization)
        v[k+1] = v[k] + M@z[k+1] - w[k+1]
        VP.set_infinity_norm_objective([z[k+1] - z[k]])
        res = VP.solve()
        all_residuals.append(res)

    print(all_residuals)
    return


@app.cell
def _(
    K,
    LinearExpression,
    T,
    Verifier,
    csr_matrix,
    n_s,
    n_u,
    np,
    rho,
    sigma,
    spa,
    x_lb,
    x_ub,
):
    def test() :
        solver_params = {
            'OutputFlag': 0,
        }

        VP = Verifier(solver='highs', params=solver_params)
        # VP = Verifier(solver='gurobi', params=solver_params)
        x_param = VP.add_param(n_s, lb=x_lb, ub=x_ub)

        s_init = x_param
        z = [None for _ in range(K + 1)]
        v = [None for _ in range(K + 1)]
        w = [None for _ in range(K + 1)]
        z[0] = np.zeros((T-1)*n_u)
        # v[0] = np.zeros((T-1)*n_u)  # rhs must be LinearExpression
        # v[0] = LinearExpression.from_int(ndim=(T-1)*n_u, lb=np.zeros((T-1)*n_u), ub=np.zeros((T-1)*n_u)) # v[0] = np.zeros((T-1)*n_u)
        x_param = VP.add_param((T-1)*n_u, lb=0.0, ub=1.0)
        v[0] = x_param

        # ADMM setup
        P = np.eye((T-1)*n_u)
        q_x = np.ones((T-1)*n_u)
        M = csr_matrix(np.eye((T-1)*n_u, (T-1)*n_u))
        # l_x = 0.0 | np.zeros(ndim) unimplemented for satlin
        l_x = LinearExpression.from_int(ndim=(T-1)*n_u, lb=np.zeros((T-1)*n_u), ub=np.zeros((T-1)*n_u))
        # u_x = 1.0 | np.ones(ndim) unimplemented for satlin
        u_x = LinearExpression.from_int(ndim=(T-1)*n_u, lb=np.ones((T-1)*n_u), ub=np.ones((T-1)*n_u))
        # u_x = LinearExpression(ndim=XX, lb=XX, ub=XX) ??
        lhs_mat = P + sigma*np.eye((T-1)*n_u) + rho*(M.T@M)
        lhs_mat_factored = spa.linalg.factorized(lhs_mat)

        # ADMM (OSQP) iterations
        all_residuals = []
        for k in range(K):
            # w[k+1] = VP.step.satlin(v[k], l=0.0, u=1.0)
            # w[k+1] = VP.step.satlin(v[k], l=np.zeros((T-1)*n_u), u=np.ones((T-1)*n_u))
            w[k+1] = VP.step.satlin(v[k], l=l_x, u=u_x)  # rhs=v[k] must be LinearExpression
            rhs = sigma*z[k] - q_x + rho*M.T@(2*w[k+1]-v[k])  # essentially just a constant -> rhs.decomposition_dict has zero matrix
            z[k+1] = VP.step.implicit_linear(rhs=rhs, lhs_mat=lhs_mat, lhs_mat_factorization=lhs_mat_factored)
            v[k+1] = v[k] + M@z[k+1] - w[k+1]
            VP.set_infinity_norm_objective([z[k+1] - z[k]]) # stack (z[k], v[k]) for FPI representation
            res = VP.solve()
            all_residuals.append(res)

        return all_residuals

    print(test())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
