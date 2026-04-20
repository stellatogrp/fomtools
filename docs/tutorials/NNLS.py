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

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return Verifier, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial we show how to use our package to verify projected gradient descent for the nonnegative least-squares problem:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Az - x \rVert^2 \\ \text{subject to} & z \ge 0,
    \end{array}


    where $A \in {\bf R}^{m \times n}$, $m \ge n$, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^m$ parameterizes the family of problems.

    For the first order method, we choose projected gradient descent with step size $\eta > 0$, with iterates given by

    \begin{array}{ll}
        z^{k+1} = {\bf ReLU}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}

    where $\bf ReLU(v) = \max\{v, 0\}$ element-wise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ as a unit box.
    We also choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $A^T A$.
    We initialize the algorithm with $z^0 = 0$.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    m, n = 5, 3
    A = np.random.normal(size=(m, n))

    # x_lb, x_ub = 10, 11
    x_lb, x_ub = 0, 0.5
    # z0_lb, z0_ub = 0.0, 0.0
    z0_lb, z0_ub = 5, 5.1

    L = np.max(np.linalg.eigvals(A.T @ A))
    eta = 1.0 / L
    K = 5

    n_samples = 10000

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }
    return (
        A,
        K,
        L,
        eta,
        m,
        n,
        n_samples,
        solver_params,
        x_lb,
        x_ub,
        z0_lb,
        z0_ub,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate the algorithms on given problem instances, which are parameterized by the set of parameters $x \in \mathcal{X} = [x_{\text{lb}}, x_{\text{ub}}]$. We randomly sample $x$ from uniform distribution over $\mathcal{X}$.
    The output of `VP.solve(K=K)` will be an array of size `(n_samples, K)` whose $n$-th row is a list of infinity norm of residuals `z[k] - z[k-1]` for `k=1,...,K`. After `VP.solve` method has run, the `Verifier` class provides statistics of the residual at last iterate `z[K]`.
    """)
    return


@app.cell
def _(A, K, Verifier, eta, m, n, n_samples, np, x_lb, x_ub, z0_lb, z0_ub):
    _VP = Verifier(method='simulator')

    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, m))
    _x_param = _VP.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=_x_samples)

    _z = _VP.add_iterate(ndim=n, lb=z0_lb, ub=z0_ub, init=np.full(n, (z0_lb+z0_ub)/2))

    _VP.step.relu(_z - eta * A.T @ (A @ _z - _x_param))
    _residuals = _VP.solve(K=K)

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
        # f"residuals: {_residuals}"
    )

    # Take average of residuals for iterate K=1 to K=K
    simulation_residuals = np.mean(_residuals, axis=0)
    return (simulation_residuals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now simulate for momentum PGD as well. Here, the momentum coefficient $(k-1)/(k+2)$ depends on the current iteration number $k$.
    """)
    return


@app.cell
def _(A, K, eta, m, n, n_samples, np, x_lb, x_ub, z0_lb, z0_ub):
    # Momentum coefficients depend on k, so we simulate explicitly instead of
    # using Verifier(method='simulator'), which applies one fixed step map.
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, m))

    _z = np.full((n_samples, n), (z0_lb + z0_ub) / 2.0, dtype=float)
    _w_prev = _z.copy()  # w0 = z0

    _residuals = np.empty((n_samples, K), dtype=float)
    for _k in range(1, K+1):
        _grad = (_z @ A.T - _x_samples) @ A
        _w = np.maximum(_z - eta * _grad, 0.0)

        _beta_k = (_k-1) / (_k+2)
        _z_new = _w + _beta_k * (_w - _w_prev)

        _residuals[:, _k-1] = np.max(np.abs(_z_new - _z), axis=1)
        _w_prev = _w
        _z = _z_new

    simulation_residuals_momentum = np.mean(_residuals, axis=0)
    return (simulation_residuals_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now obtain worst-case guarantee by solving the corresponding verification problem using MIP solver.
    """)
    return


@app.cell
def _(A, K, Verifier, eta, m, n, np, solver_params, x_lb, x_ub, z0_lb, z0_ub):
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(m, lb=x_lb, ub=x_ub)

    _z0 = _VP.add_iterate(n, lb=z0_lb, ub=z0_ub)
    _z = [None for _ in range(K+1)]
    _z[0] = _z0

    all_residuals = []
    worst_case_x_param = []
    for _k in range(1, K+1):
        _z[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _VP.set_objective_norm([_z[_k] - _z[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)
        # Store the horizon-specific worst-case instance for this k.
        worst_case_x_param.append(_VP.get_solution_value(_x_param))
    return all_residuals, worst_case_x_param


@app.cell
def _(
    A,
    K,
    Verifier,
    eta,
    m,
    n,
    np,
    worst_case_x_param,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    # Simulate each horizon k with its own worst-case x from verification at that k.
    worst_case_residuals = []
    for _k in range(1, K + 1):
        _VP = Verifier(method='simulator')
        _x_param = _VP.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=[worst_case_x_param[_k - 1]])
        _z = _VP.add_iterate(ndim=n, lb=z0_lb, ub=z0_ub, init=np.full(n, (z0_lb + z0_ub) / 2))
        _VP.step.relu(_z - eta * A.T @ (A @ _z - _x_param))
        worst_case_residuals.append(_VP.solve(K=_k)[0][-1])

    worst_case_residuals = np.array(worst_case_residuals)
    return (worst_case_residuals,)


@app.cell
def _(A, K, Verifier, eta, m, n, np, solver_params, x_lb, x_ub, z0_lb, z0_ub):
    # We now run VP for accelerated projected gradient descent
    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(m, lb=x_lb, ub=x_ub)

    _z0 = _VP.add_iterate(n, lb=z0_lb, ub=z0_ub)
    _z = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]
    _z[0] = _z0
    _w[0] = _z0

    all_residuals_momentum = []
    worst_case_x_param_momentum = []
    for _k in range(1, K+1):
        _w[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _z[_k] = _w[_k] + (_k-1)/(_k+2) * (_w[_k] - _w[_k-1])
        _VP.set_objective_norm([_z[_k] - _z[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals_momentum.append(_res)
    
        # Store the horizon-specific worst-case instance for this k.
        worst_case_x_param_momentum.append(_VP.get_solution_value(_x_param))
    return all_residuals_momentum, worst_case_x_param_momentum


@app.cell
def _(A, K, eta, n, np, worst_case_x_param_momentum, z0_lb, z0_ub):
    # Momentum has k-dependent coefficients, so simulate each horizon explicitly.
    worst_case_residuals_momentum = []
    _z0 = np.full(n, (z0_lb + z0_ub) / 2.0, dtype=float)

    for _k in range(1, K + 1):
        _x_k = np.array(worst_case_x_param_momentum[_k - 1], dtype=float)

        _z = _z0.copy()
        _w_prev = _z0.copy()
        _last_residual = 0.0

        for _j in range(1, _k + 1):
            _grad = (_z @ A.T - _x_k) @ A
            _w = np.maximum(_z - eta * _grad, 0.0)
            _beta_j = (_j-1) / (_j+2)
            _z_new = _w + _beta_j * (_w - _w_prev)

            _last_residual = np.max(np.abs(_z_new - _z))
            _w_prev = _w
            _z = _z_new

        worst_case_residuals_momentum.append(_last_residual)

    worst_case_residuals_momentum = np.array(worst_case_residuals_momentum)
    print(worst_case_residuals_momentum)
    return (worst_case_residuals_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we plot and compare the results to in-sample average from simulator.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_momentum,
    plt,
    simulation_residuals,
    simulation_residuals_momentum,
    worst_case_residuals,
    worst_case_residuals_momentum,
):
    plt.plot(range(1, K+1), all_residuals, label='PGD (MIP worst-case)')
    plt.plot(range(1, K+1), simulation_residuals, '--', label='PGD (in-sample avg)')
    plt.plot(range(1, K+1), worst_case_residuals, '-.', color='k', linewidth=.5, label='worst-case PGD instance')
    plt.plot(range(1, K+1), all_residuals_momentum, label='momentum PGD (MIP worst-case)')
    plt.plot(range(1, K+1), simulation_residuals_momentum, '--', label='momentum PGD (in-sample avg)')
    plt.plot(range(1, K+1), worst_case_residuals_momentum, '-.', color='k', linewidth=.5, label='worst-case momentum PGD instance')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('nonnegative least-squares example (MIP)')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compare our results to the worst-case performance of the algorithm returned by the performance estimation problem (PEP), using the package [`PEPit`](https://pepit.readthedocs.io/en/0.5.1/).
    Specifically, we measure the performance of projected gradient descent and its momentum variant applied on smooth convex minimization.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we compare the result obtained from solving SDP-relaxed verification problem.
    """)
    return


@app.cell
def _(A, K, Verifier, eta, m, n, x_lb, x_ub, z0_lb, z0_ub):
    _VP = Verifier(method='sdp', solver='clarabel', obbt_handler='highs')

    _x_param = _VP.add_param(m, lb=x_lb, ub=x_ub)

    _z0 = _VP.add_iterate(n, lb=z0_lb, ub=z0_ub)
    _z = [None for _ in range(K+1)]
    _z[0] = _z0

    all_residuals_sdp = []
    for _k in range(1, K+1):
        _z[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _VP.set_objective_norm([_z[_k] - _z[_k-1]], 2)
        _res = _VP.solve()
        all_residuals_sdp.append(_res)
    return (all_residuals_sdp,)


@app.cell
def _(A, K, Verifier, eta, m, n, x_lb, x_ub, z0_lb, z0_ub):
    _VP = Verifier(method='sdp', solver='clarabel', obbt_handler='highs')
    _x_param = _VP.add_param(m, lb=x_lb, ub=x_ub)

    _z0 = _VP.add_iterate(n, lb=z0_lb, ub=z0_ub)
    _z = [None for _ in range(K+1)]
    _w = [None for _ in range(K+1)]
    _z[0] = _z0
    _w[0] = _z0

    all_residuals_sdp_momentum = []
    for _k in range(1, K+1):
        _w[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _z[_k] = _w[_k] + (_k-1)/(_k+2) * (_w[_k] - _w[_k-1])
        _VP.set_objective_norm([_z[_k] - _z[_k-1]], 2)
        # _VP.set_objective_norm([_w[_k] - _w[_k-1]], 2)
        _res = _VP.solve()
        all_residuals_sdp_momentum.append(_res)
    return (all_residuals_sdp_momentum,)


@app.cell
def _(K, all_residuals_sdp, all_residuals_sdp_momentum, plt):
    plt.plot(range(1, K+1), all_residuals_sdp, label='PGD')
    plt.plot(range(1, K+1), all_residuals_sdp_momentum, label='momentum PGD')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('nonnegative least-squares example (SDP)')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
