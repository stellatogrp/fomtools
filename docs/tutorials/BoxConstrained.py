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
    In this tutorial, we illustrate how to use our package to verify projected gradient descent for box-constrained optimization problems:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Az - x \rVert^2 \\
        \text{subject to} & \|z\|_\infty \le r,
    \end{array}

    where $A \in {\bf R}^{m \times n}$, $m \ge n$, $z \in {\bf R}^n$ is the decision variable, $x \in {\bf R}^m$ parameterizes the family, and $r > 0$ is the radius of the L-infinity ball constraint.

    For the first order method, we choose projected gradient descent with step size $\eta > 0$, with iterates given by

    \begin{array}{ll}
        z^{k+1} = \operatorname{prox}_{\eta I_{\{z : \|z\|_\infty \leq r\}}}(z^k - \eta A^T(Az^k - x)),
    \end{array}

    where the projection onto the L-infinity ball clips each component to $[-r, r]$:

    $$
        \operatorname{prox}_{\eta I_{\{z : \|z\|_\infty \leq r\}}}(z) = \min(\max(z, -r), r)
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ as a unit box.
    We also choose $\eta = 0.5/L$, where $L$ is the maximum eigenvalue of $A^T A$, and set $r = 1.0$ as the radius.
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
    r = 1.0
    L = np.max(np.linalg.eigvals(A.T @ A))
    eta = 0.5 / L
    K = 10
    return A, K, eta, m, n, r, x_lb, x_ub


@app.cell
def _(A, K, Verifier, eta, m, n, r, x_lb, x_ub):
    from algoverify.proxable_functions import IndBallLinf

    # Indicator function for L-infinity ball constraint
    indicator = IndBallLinf(r=r)

    solver_params = {
        'OutputFlag': 0,
    }

    VP = Verifier(params=solver_params)
    x_param = VP.add_param(m, lb=x_lb, ub=x_ub)

    z0 = VP.add_iterate(n, lb=0, ub=0)
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_residuals = []
    for k in range(1, K+1):
        # Projected gradient step: z[k] = clip(z[k-1] - eta * A^T(Az[k-1] - x), [-r, r])
        z[k] = VP.prox(indicator, z[k-1] - eta * A.T @ (A @ z[k-1] - x_param), gamma=eta)
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

