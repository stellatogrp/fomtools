import marimo

__generated_with = "0.14.16"
app = marimo.App()


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
    In this tutorial we show how to use our package to verify projected gradient descent for the nonnegative least-squares problem:

    \begin{array}{ll} 
        \text{minimize} & (1/2) \lVert Az - x \rVert^2 \\ \text{subject to} & z \ge 0, 
    \end{array}
    
    
    where $A \in {\bf R}^{m \times n}$, $m \ge n$, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^m$ parameterizes the family.

    For the first order method, we choose projected gradient descent with step size $\eta > 0$, with iterates given by
    
    \begin{array}{ll} 
        z^{k+1} = {\bf ReLU}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}
    
    where $\bf ReLU(v) = \max\{v, 0\}$ element-wise.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ as a unit box.
    We also choose $\eta = 0.5/L$, where $L$ is the maximum eigenvalue of $A^T A$.
    We initialize the algorithm with $z^0 = 0$.
    """
    )
    return


@app.cell
def _(np):
    np.random.seed(0)
    m, n = 5, 3
    A = np.random.normal(size=(m, n))
    x_lb, x_ub = 10, 11
    L = np.max(np.linalg.eigvals(A.T @ A))
    eta = 0.5 / L
    K = 10
    return A, K, eta, m, n, x_lb, x_ub


@app.cell
def _(A, K, Verifier, eta, m, n, x_lb, x_ub):
    solver_params = {
        'OutputFlag': 0,
    }

    VP = Verifier(params=solver_params)
    x_param = VP.add_param(m, lb=x_lb, ub=x_ub)

    z0 = VP.add_iterate(n, lb=1, ub=1)
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_residuals = []
    for k in range(1, K+1):
        z[k] = VP.step.relu(z[k-1] - eta * A.T @ (A @ z[k-1] - x_param))
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
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
