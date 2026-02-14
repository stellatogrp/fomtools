import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    np.random.seed(2)
    return Verifier, cp, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We solve this problem using the standard Chambolle-Pock method, or Primal-dual Hybrid Gradient (PDHG):

    \begin{array}{rcl}
        z^{k+1} &=& \Pi_{[0, c]} \big( z^k - \eta (\mu + A_s^T v^k - A_d^T w^k) \big) \\
        v^{k+1} &=& \max \big\{ v^k + \eta \big( - b_s + A_s (2 z^{k+1} - z^k) \big) \big\} \\
        w^{k+1} &=& w^k + \eta \big( x - A_d (2 z^{k+1} - z^k) \big).
    \end{array}

    We choose $\eta = 0.5 / \| \begin{bmatrix} - A_s^T & A_d^T \end{bmatrix} \|_2$.
    """
    )
    return


@app.cell
def _(np):
    # Generate bipartite graph with n_s supply nodes and n_d demand nodes,
    # whose edges are added with probability 0.5.
    n_s = 15
    n_d = 10

    # Generate bipartite graph G and its incidence matrix A
    p = 0.5

    # import networkx as nx
    # G = nx.bipartite.random_graph(n_s, n_d, p, seed=1, directed=False)
    # A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)
    # A = A.todense()
    # A[n_s:, :] *= -1

    # with open("incidence_matrix.npy", "wb") as write_file:
    #     np.save(write_file, A)

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
    problem.solve()

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
    problem_check.solve()

    print(problem_check.status, problem_check.value)
    return


@app.cell
def _(
    A_d,
    A_s,
    Verifier,
    b_s,
    c,
    c_lb,
    c_ub,
    mu,
    n_d,
    n_e,
    n_s,
    np,
    x_lb,
    x_ub,
    z0_value,
):
    K = 1
    eta = 0.5 / np.linalg.norm(np.hstack([A_s.T, A_d.T]), 2)

    solver_params = {
        'OutputFlag': 0,
    }

    # VP = Verifier(params=solver_params, solver="highs")
    VP = Verifier(params=solver_params, solver="gurobi")
    x_param = VP.add_param(n_d, lb=x_lb, ub=x_ub)
    c_param = VP.add_param(n_e, lb=c_lb*np.ones(n_e), ub=c_ub*np.ones(n_e))
    b_s_param = VP.add_param(n_s, lb=b_s, ub=b_s)
    mu_param = VP.add_param(n_e, lb=mu, ub=mu)

    # z0, v0, w0 = z0_value, np.zeros(n_s), np.zeros(n_d)
    # z0 = VP.add_iterate(n_e, lb=np.zeros(n_e), ub=np.zeros(n_e))
    z0 = VP.add_iterate(n_e, lb=z0_value, ub=z0_value)
    v0 = VP.add_iterate(n_s, lb=np.zeros(n_s), ub=np.zeros(n_s))
    w0 = VP.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))
    z = [None for _ in range(K+1)]
    v = [None for _ in range(K+1)]
    w = [None for _ in range(K+1)]
    z[0] = z0
    v[0] = v0
    w[0] = w0

    all_residuals = []
    for k in range(1, K+1):
        # z[k] = VP.step.satlin(z[k-1] - eta * (mu + A_s.T @ v[k-1] - A_d.T @ w[k-1]), l=np.zeros(n_e), u=c)
        # z[k] = VP.step.satlin(z[k-1] - eta * (mu + A_s.T @ v[k-1] - A_d.T @ w[k-1]), l=c, u=np.zeros(n_e))
        # z_bar = z[k-1] - eta * (mu + A_s.T @ v[k-1] - A_d.T @ w[k-1])
        z_bar = VP.add_param(n_e, lb=np.zeros(n_e), ub=np.ones(n_e))
        z[k] = VP.step.relu(c-z_bar)
        print(np.array([np.subtract(ub, lb) for (lb, ub) in zip(z[k].lb, z[k].ub)]))
        v[k] = VP.step.relu(v[k-1] + eta * (- b_s + A_s @ (2*z[k] - z[k-1])))
        w[k] = w[k-1] + eta * (x_param - A_d @ (2*z[k] - z[k-1]))
        VP.set_infinity_norm_objective([z[k] - z[k-1], v[k] - v[k-1], w[k] - w[k-1]])
        # VP.set_infinity_norm_objective([b_s - A_s @ z[k]])
        res = VP.solve()
        all_residuals.append(res)

    print(all_residuals)
    return K, all_residuals


@app.cell
def _(K, all_residuals, plt):
    plt.plot(range(1, K+1), all_residuals, label='VP')
    # plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
