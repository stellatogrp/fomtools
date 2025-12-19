import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from algoverify.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return Verifier, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this tutorial, we illustrate how to use our package to verify the Primal-Dual Hybrid Gradient (PDHG) method for solving linear programming problems in standard form:

    \begin{array}{ll}
        \text{minimize} & c^T z \\
        \text{subject to} & A z = b, \quad z \ge 0,
    \end{array}

    where $A \in {\bf R}^{m \times n}$, $c \in {\bf R}^n$ is the cost vector, $b \in {\bf R}^m$ is the right-hand side vector, and $z \in {\bf R}^n$ is the decision variable.

    The PDHG algorithm alternates between primal and dual updates. With step size $t > 0$, the iterations are given by:

    \begin{array}{ll}
        z^{k+1} & = \operatorname{prox}_{t I_{\geq 0}}(z^k - t (c - A^T y^k)) \\
        y^{k+1} & = y^k - t (A (2 z^{k+1} - z^k) - b),
    \end{array}

    where $\operatorname{prox}_{t I_{\geq 0}}(x) = \max(x, 0)$ is the projection onto the non-negative orthant (ReLU operation), and $y \in {\bf R}^m$ is the dual variable.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this example, we choose $m = 5$ and $n = 10$, randomly sample $A$, $b$, and $c$, and parameterize $c$ as a box constraint.
    We also choose $t = 0.1$ as the step size.
    We initialize the algorithm with $z^0 = 0$ and $y^0 = 0$.
    """
    )
    return


@app.cell
def _(np):
    np.random.seed(1)
    m, n = 5, 10
    A = np.random.normal(size=(m, n))
    b = np.random.uniform(size=m)
    c_lb, c_ub = 0.5, 1.5
    t = 0.1
    K = 10
    return A, K, c_lb, c_ub, m, n, t


@app.cell
def _(A, K, Verifier, b, c_lb, c_ub, m, n, t):
    from algoverify.proxable_functions import IndNonNegative

    # Indicator function for non-negative constraint
    indicator = IndNonNegative()

    solver_params = {
        'OutputFlag': 0,
    }

    VP = Verifier(params=solver_params)
    c_param = VP.add_param(n, lb=c_lb, ub=c_ub)

    z0 = VP.add_iterate(n, lb=0, ub=0)
    y0 = VP.add_iterate(m, lb=0, ub=0)
    z = [None for _ in range(K + 1)]
    y = [None for _ in range(K + 1)]
    z[0] = z0
    y[0] = y0

    all_residuals = []
    for k in range(1, K+1):
        # Primal update: z^{k+1} = prox_{t I_{\geq 0}}(z^k - t (c - A^T y^k))
        z[k] = VP.prox(indicator, z[k-1] - t * (c_param - A.T @ y[k-1]), gamma=t)
        # Dual update: y^{k+1} = y^k - t (A (2 z^{k+1} - z^k) - b)
        y[k] = y[k-1] - t * (A @ (2 * z[k] - z[k-1]) - b)
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        res = VP.solve()
        all_residuals.append(res)
    return (all_residuals,)


@app.cell
def _(K, all_residuals, plt):
    plt.plot(range(1, K+1), all_residuals, label='VP')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

