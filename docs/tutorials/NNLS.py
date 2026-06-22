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
    from pepit_runner import PEPitRunner
    from scipy.optimize import nnls

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }
    return PEPitRunner, Verifier, nnls, np, plt, solver_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial, we show how to use our package to verify the performance of projected gradient descent solving the nonnegative least-squares problem:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Az - x \rVert^2 \\ \text{subject to} & z \ge 0,
    \end{array}

    where $A \in {\bf R}^{m \times n}$, $m \ge n$, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^m$ parameterizes the family of problems.

    For the first order method, we choose projected gradient descent with step size $\eta > 0$, with iterates given by

    \begin{array}{ll}
        z^{k+1} = {\bf ReLU}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}

    where $\bf ReLU(v) = \max\{v, 0\}$ element-wise.
    We choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $A^T A$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ as a unit box $[0.0, 1.0]$.

    Our performance metric will be the fixed-point residual `z[k] - z[k-1]` given the iteration budget $k>0$ and initial condition $z^0 \in [\texttt{z\_lb}, \texttt{z\_ub}]$.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    m, n = 5, 3
    A = np.random.normal(size=(m, n))
    x_lb, x_ub = 10, 11
    L = np.max(np.linalg.eigvalsh(A.T @ A))  # symmetric routine → real eigenvalues
    eta = 0.5 / L
    K = 10
    return A, K, eta, m, n, x_lb, x_ub


@app.cell
def _(A, K, Verifier, eta, m, n, nnls, np, x_lb, x_ub, z0_lb, z0_ub):
    n_samples = 10000 # the number of samples to simulate

    _VP = Verifier(method='simulator', obbt_solver=None)

    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, m))
    _z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n))
    _x_param = _VP.add_param(ndim=m, samples=_x_samples)
    _x_param.lb, _x_param.ub = np.full(m, x_lb), np.full(m, x_ub)

    _z = [None] * (K + 1)
    _z[0] = _VP.add_param(ndim=n, samples=_z0_samples)
    _z[0].lb, _z[0].ub = np.full(n, z0_lb), np.full(n, z0_ub)

    for _k in range(1, K + 1):
        _z[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _VP.set_objective_norm(_z[_k] - _z[_k-1], ord=np.inf)
    _residuals = _VP.solve()

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
        f"residuals: {_residuals}"
    )

    # Take average of residuals for iterate K=1 to K=K
    simulation_residuals = np.mean(_residuals, axis=0)
    simulation_residual_quantiles = np.quantile(_residuals, [0.90, 0.50, 0.10], axis=0)
    _z_star_samples = np.array([nnls(A, _x_sample)[0] for _x_sample in _x_samples])
    pepit_radius = np.max(np.linalg.norm(_z0_samples - _z_star_samples, axis=1))
    return (
        n_samples,
        pepit_radius,
        simulation_residual_quantiles,
        simulation_residuals,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also consider a momentum variant of projected gradient descent. This
    method follows the FISTA update, with the proximal operator for the
    nonnegative orthant replaced by the element-wise ReLU projection:

    \begin{array}{rcl}
        w^{k} &=& {\bf ReLU}\left(z^{k-1} - \eta A^T(Az^{k-1} - x)\right), \\
        z^{k} &=& w^{k} + \beta_k (w^{k} - w^{k-1}),
    \end{array}

    with $w^0 = z^0$. Since
    $\mathbf{prox}_{\iota_{\mathbb{R}_+^n}}(v)$ is the projection onto
    the nonnegative orthant, this prox step is exactly
    ${\bf ReLU}(v) = \max\{v, 0\}$ element-wise.

    In this experiment, we choose the momentum coefficient $\beta_k$ as $\beta_k = \frac{k-1}{k+2}$.
    """)
    return


@app.cell
def _(A, K, Verifier, eta, m, n, n_samples, np, x_lb, x_ub, z0_lb, z0_ub):
    _VP = Verifier(method='simulator', obbt_solver=None)

    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, m))
    _z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n))
    _x_param = _VP.add_param(ndim=m, samples=_x_samples)
    _x_param.lb, _x_param.ub = np.full(m, x_lb), np.full(m, x_ub)

    _z = [None] * (K + 1)
    _w = [None] * (K + 1)
    _z[0] = _VP.add_param(ndim=n, samples=_z0_samples)
    _z[0].lb, _z[0].ub = np.full(n, z0_lb), np.full(n, z0_ub)
    _w[0] = _z[0]

    for _k in range(1, K+1):
        _beta_k = (_k-1) / (_k+2)
        _w[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _z[_k] = _w[_k] + _beta_k * (_w[_k] - _w[_k-1])
        _VP.set_objective_norm(_z[_k] - _z[_k-1], ord=np.inf)

    _residuals = _VP.solve()

    simulation_residuals_momentum = np.mean(_residuals, axis=0)
    simulation_residual_quantiles_momentum = np.quantile(
        _residuals, [0.90, 0.50, 0.10], axis=0
    )
    return (
        simulation_residual_quantiles_momentum,
        simulation_residuals_momentum,
    )


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
    worst_case_z0 = []
    for _k in range(1, K+1):
        _z[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _VP.set_objective_norm([_z[_k] - _z[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)
        # Store the horizon-specific worst-case instance for this k.
        worst_case_x_param.append(_VP.get_solution_value(_x_param))
        worst_case_z0.append(_VP.get_solution_value(_z0))
    return all_residuals, worst_case_x_param, worst_case_z0


@app.cell
def _(A, K, Verifier, eta, m, n, np, worst_case_x_param, worst_case_z0):
    # Simulate each horizon k with its own worst-case x and z0 from verification at that k.
    worst_case_residuals = []
    for _k in range(1, K + 1):
        _VP = Verifier(method='simulator', obbt_solver=None)
        _x_param = _VP.add_param(ndim=m, samples=[worst_case_x_param[_k - 1]])
        _x_param.lb = _x_param.ub = np.asarray(worst_case_x_param[_k - 1])
        _z = [None] * (_k + 1)
        _z0_k = np.asarray(worst_case_z0[_k - 1])
        _z[0] = _VP.add_iterate(ndim=n, init=_z0_k)
        _z[0].lb = _z[0].ub = _z0_k
        for k in range(1, _k + 1):
            _z[k] = _VP.step.relu(_z[k-1] - eta * A.T @ (A @ _z[k-1] - _x_param))
        worst_case_residuals.append(_VP.solve()[0][-1])

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
    worst_case_z0_momentum = []
    for _k in range(1, K+1):
        _w[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _z[_k] = _w[_k] + (_k-1)/(_k+2) * (_w[_k] - _w[_k-1])
        _VP.set_objective_norm([_z[_k] - _z[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals_momentum.append(_res)

        # Store the horizon-specific worst-case instance for this k.
        worst_case_x_param_momentum.append(_VP.get_solution_value(_x_param))
        worst_case_z0_momentum.append(_VP.get_solution_value(_z0))
    return (
        all_residuals_momentum,
        worst_case_x_param_momentum,
        worst_case_z0_momentum,
    )


@app.cell
def _(A, K, eta, np, worst_case_x_param_momentum, worst_case_z0_momentum):
    # Momentum has k-dependent coefficients, so simulate each horizon explicitly.
    worst_case_residuals_momentum = []

    for _k in range(1, K + 1):
        _x_k = np.array(worst_case_x_param_momentum[_k - 1], dtype=float)
        _z0 = np.array(worst_case_z0_momentum[_k - 1], dtype=float)

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
    We compare our results to the worst-case performance of projected gradient descent
    returned by the performance estimation problem (PEP), using the shared `PEPitRunner`.
    """)
    return


@app.cell
def _(K, L, PEPitRunner, eta, pepit_radius):
    all_residuals_PEP, _ = PEPitRunner().proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius, nonsmooth="indicator")
    return (all_residuals_PEP,)


@app.cell
def _(K, L, PEPitRunner, eta, pepit_radius):
    all_residuals_PEP_momentum, _ = PEPitRunner().accelerated_proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius, nonsmooth="indicator")
    return (all_residuals_PEP_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we compare the result obtained from solving SDP-relaxed verification problem.
    """)
    return


@app.cell
def _(A, K, Verifier, eta, m, n, x_lb, x_ub, z0_lb, z0_ub):
    _VP = Verifier(method='sdp', solver='clarabel')

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
    _VP = Verifier(method='sdp', solver='clarabel')
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
        _res = _VP.solve()
        all_residuals_sdp_momentum.append(_res)
    return (all_residuals_sdp_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we plot the verification results against the PEPit bound and the
    horizon-specific worst-case instances extracted from MIP.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    all_residuals_sdp,
    plt,
    simulation_residual_quantiles,
    simulation_residuals,
    worst_case_residuals,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='PGD (PEPit)')
    plt.plot(_ks, all_residuals, label='PGD (MIP worst-case)')
    plt.plot(_ks, all_residuals_sdp, label='PGD (SDP bound)')
    plt.plot(_ks, simulation_residuals, '--', label='PGD (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles[2], simulation_residual_quantiles[0], alpha=.18, label='PGD simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals, '-.', color='k', linewidth=.7, label='worst-case PGD instance')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('nonnegative least-squares example')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also plot the verification results for momentum PGD.
    """)
    return


@app.cell
def _(
    K,
    all_residuals_PEP_momentum,
    all_residuals_momentum,
    all_residuals_sdp_momentum,
    plt,
    simulation_residual_quantiles_momentum,
    simulation_residuals_momentum,
    worst_case_residuals_momentum,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_momentum, label='momentum PGD (PEPit)')
    plt.plot(_ks, all_residuals_momentum, label='momentum PGD (MIP worst-case)')
    plt.plot(_ks, all_residuals_sdp_momentum, label='momentum PGD (SDP bound)')
    plt.plot(_ks, simulation_residuals_momentum, '--', label='momentum PGD (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_momentum[2], simulation_residual_quantiles_momentum[0], alpha=.18, label='momentum PGD simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals_momentum, '-.', color='k', linewidth=.7, label='worst-case momentum PGD instance')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('nonnegative least-squares momentum PGD example')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
