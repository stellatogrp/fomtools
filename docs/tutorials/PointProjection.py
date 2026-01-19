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
    In this tutorial, we illustrate how to use our package to verify algorithms that project onto a fixed point:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert z - x \rVert^2 \\
        \text{subject to} & z = p,
    \end{array}

    where $z \in {\bf R}^n$ is the decision variable, $x \in {\bf R}^n$ is the problem parameter, and $p \in {\bf R}^n$ is a fixed target point.

    The projection onto a single point is trivial: regardless of the input, the projection always returns the target point $p$. This is useful in algorithms that enforce equality constraints or track a reference point.

    For the first order method, we consider the projection operator:

    \begin{array}{ll}
        z^{k+1} = \operatorname{prox}_{I_{\{p\}}}(z^k) = p,
    \end{array}

    where $I_{\{p\}}$ is the indicator function of the singleton set $\{p\}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this example, we choose $n = 3$, randomly sample the target point $p$, and parameterize $x$ as a box constraint.
    We initialize the algorithm with $z^0 = 0$.
    """
    )
    return


@app.cell
def _(np):
    np.random.seed(0)
    n = 3
    p = np.random.normal(size=n)
    x_lb, x_ub = -2, 2
    K = 10
    return K, n, p, x_lb, x_ub


@app.cell
def _(K, Verifier, n, p, x_lb, x_ub):
    from algoverify.proxable_functions import IndPoint

    # Indicator function for point constraint
    indicator = IndPoint(p=p)

    solver_params = {
        'OutputFlag': 0,
    }

    VP = Verifier(params=solver_params)
    x_param = VP.add_param(n, lb=x_lb, ub=x_ub)

    z0 = VP.add_iterate(n, lb=0, ub=0)
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_residuals = []
    for k in range(1, K+1):
        # Projection onto point p: z[k] = prox_{I_{p}}(z[k-1] - x_param) = p
        # Note: The projection is independent of the input
        z[k] = VP.prox(indicator, z[k-1] - x_param, gamma=1.0)
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

