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

    from fomtools.verifier import Verifier

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
    K = 5
    return A, K, L, eta, m, n, x_lb, x_ub


@app.cell
def _(A, K, Verifier, eta, m, n, x_lb, x_ub):
    solver_params = {
        'OutputFlag': 0,
    }

    _VP = Verifier(solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(m, lb=x_lb, ub=x_ub)

    _z0 = _VP.add_iterate(n, lb=1, ub=1)
    _z = [None for _ in range(K + 1)]
    _z[0] = _z0

    all_residuals = []
    for _k in range(1, K+1):
        _z[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _VP.set_infinity_norm_objective([_z[_k] - _z[_k-1]])
        _res = _VP.solve()
        all_residuals.append(_res)
    return all_residuals, solver_params


@app.cell
def _(A, K, Verifier, eta, m, n, solver_params, x_lb, x_ub):
    # We now run VP for accelerated projected gradient descent
    _VP = Verifier(solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(m, lb=x_lb, ub=x_ub)

    _z0 = _VP.add_iterate(n, lb=1, ub=1)
    _z = [None for _ in range(K + 1)]
    _w = [None for _ in range(K + 1)]
    _z[0] = _z0
    _w[0] = _z0

    all_residuals_momentum = []
    for _k in range(1, K+1):
        _w[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _z[_k] = _w[_k] + (_k-1)/(_k+2) * (_w[_k] - _w[_k-1])
        _VP.set_infinity_norm_objective([_z[_k] - _z[_k-1]])
        _res = _VP.solve()
        all_residuals_momentum.append(_res)
    return (all_residuals_momentum,)


@app.cell
def _(K, all_residuals, all_residuals_momentum, plt):
    plt.plot(range(1, K+1), all_residuals, label='PGD')
    plt.plot(range(1, K+1), all_residuals_momentum, label='momentum PGD')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('nonnegative least-squares example (PEP)')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We compare our results to the worst-case performance of the algorithm returned by the performance estimation problem (PEP), using the package [`PEPit`](https://pepit.readthedocs.io/en/0.5.1/).
    Specifically, we measure the performance of projected gradient descent and its momentum variant applied on smooth convex minimization.
    """
    )
    return


@app.cell
def _(K, L, eta):
    from PEPit import PEP, Point
    from PEPit.functions import SmoothConvexFunction, ConvexIndicatorFunction
    from PEPit.primitive_steps import proximal_step

    all_residuals_PEP = []

    for _k_PEP in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(SmoothConvexFunction, L=L)
        _g = _problem.declare_function(ConvexIndicatorFunction)
        _F = _f + _g

        # Define optimal solution
        _xs = _F.stationary_point()

        # Declare initial point s0
        _x0 = _problem.set_initial_point()
        _problem.set_initial_condition((_x0 - _xs) ** 2 <= 1.0)

        _x = [None for _ in range(K+1)]
        _x[0] = _x0
        for _k in range(1, _k_PEP+1):
            _x[_k], _, _ = proximal_step(_x[_k-1] - eta * _f.gradient(_x[_k-1]), _g, eta)
        _problem.set_performance_metric(_F(_x[_k_PEP]) - _F(_xs))
        # _problem.set_performance_metric((_x[_k_PEP] - _x[_k_PEP-1]) ** 2)
        _res = _problem.solve()
        all_residuals_PEP.append(_res) # account for L-infinity norm of VP
    return (
        ConvexIndicatorFunction,
        PEP,
        SmoothConvexFunction,
        all_residuals_PEP,
        proximal_step,
    )


@app.cell
def _(
    ConvexIndicatorFunction,
    K,
    L,
    PEP,
    SmoothConvexFunction,
    eta,
    proximal_step,
):
    # Momentum PGD
    all_residuals_momentum_PEP = []

    for _k_PEP in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(SmoothConvexFunction, L=L)
        _g = _problem.declare_function(ConvexIndicatorFunction)
        _F = _f + _g

        # Define optimal solution
        _xs = _F.stationary_point()

        # Declare initial point s0
        _x0 = _problem.set_initial_point()
        _problem.set_initial_condition((_x0 - _xs) ** 2 <= 1.0)

        _x = [None for _ in range(K+1)]
        _w = [None for _ in range(K+1)]
        _x[0] = _x0
        _w[0] = _x0
    
        for _k in range(1, _k_PEP+1):
            _w[_k], _, _ = proximal_step(_x[_k-1] - eta * _f.gradient(_x[_k-1]), _g, eta)
            _x[_k] = _w[_k] + (_k-1)/(_k+2) * (_w[_k] - _w[_k-1])
        _problem.set_performance_metric(_F(_x[_k_PEP]) - _F(_xs))
        # _problem.set_performance_metric((_x[_k_PEP] - _x[_k_PEP-1]) ** 2)
        _res = _problem.solve()
        all_residuals_momentum_PEP.append(_res) # account for L-infinity norm of VP
    return (all_residuals_momentum_PEP,)


@app.cell
def _(K, all_residuals_PEP, all_residuals_momentum_PEP, plt):
    plt.figure()
    plt.plot(range(1, K+1), all_residuals_PEP, label='PGD')
    plt.plot(range(1, K+1), all_residuals_momentum_PEP, label='momentum PGD')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Worst-case function-value gap $f(x^K) - f^\\star$')
    plt.xlabel(r'$K$')
    plt.suptitle('PEP bound')
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
