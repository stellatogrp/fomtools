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
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner
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
    # Problem setup: Nonnegative Least-squares (NNLS)

    In this tutorial, we show how to use our package to verify the performance of first-order methods solving the nonnegative least-squares (NNLS) problem:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Az - x \rVert^2 \\ \text{subject to} & z \ge 0,
    \end{array}

    where $A \in {\bf R}^{m \times n}$ with $m \ge n$, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^m$ is the problem parameter.

    ## Problem parameters

    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ as a unit box $[10.0, 11.0]$.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    m, n = 5, 3
    A = np.random.normal(size=(m, n))
    return A, m, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance metric

    Given the sequence of iterates $\{z^k\}_{k=0}^K$ generated from the algorithm of our choice, we quantify its performance through measuring how fast the fixed-point residual $z^K - z^{K-1}$ decreases, given the iteration budget $K>0$.
    We impose the initial condition $z^0 \in \mathcal{Z} := [\texttt{z0\_lb},\, \texttt{z0\_ub}]$.

    This amounts to finding the worst-case performance by solving the verification problem (VP) of the following form:

    \begin{array}{ll}
        \textnormal{maximize} & \|z^K - z^{K-1}\| \\
        \textnormal{subject to} & \{z^k\}_{k=0}^K \text{ given by algorithm} \\
        & z^0 \in \mathcal{Z} \\
        & x \in \mathcal{X},
    \end{array}

    where $\mathcal{X} := [\texttt{x\_lb},\, \texttt{x\_ub}]$ is the space of problem parameters.
    """)
    return


@app.cell
def _():
    z0_lb, z0_ub = 4.0, 6.0    # initial condition
    x_lb, x_ub = 10.0, 11.0    # parameter set
    return x_lb, x_ub, z0_lb, z0_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: Projected gradient descent

    We solve this problem using projected gradient descent with step size $\eta > 0$, whose iterations are given by

    \begin{array}{ll}
        z^{k+1} = {\bf ReLU}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}

    where $\bf ReLU(v) = \max\{v, 0\}$ element-wise.
    We choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $A^T A$.

    We also consider a momentum variant of projected gradient descent, which follows the FISTA update:

    \begin{array}{rcl}
        w^{k+1} &=& {\bf ReLU}\left(z^{k} - \eta A^T(Az^{k} - x)\right), \\
        z^{k+1} &=& w^{k+1} + \beta_k (w^{k+1} - w^{k}),
    \end{array}

    with $w^0 = z^0$ and $\beta_k = k/(k+3)$.
    """)
    return


@app.cell
def _(A, np):
    L = np.max(np.linalg.eigvalsh(A.T @ A))  # symmetric routine → real eigenvalues
    eta = 0.5 / L
    K = 10
    return K, L, eta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    We now describe how we can use our package `fomtools` to verify the worst-case convergence rate of projected gradient descent and its accelerated variant, applied on nonnegative least-squares problem. We compare the simulated results obtained from randomly generated problem instances to the obtained worst-case bound from PEPit and our approach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    The `Verifier` class offers functionality of running simulations of the algorithm given randomly generated problem instances, with the option `method='simulator'`.
    This randomly-generated problem instance consists of two parameters: randomly generated problem parameter $x \in \mathcal{X}$ and initial iterate $z^0 \in \mathcal{Z}$.
    """)
    return


@app.cell
def _(m, n, np, x_lb, x_ub, z0_lb, z0_ub):
    n_samples = 10000 # the number of samples to simulate

    # Generate the sample of problem parameter x and initial iterate z0
    rng = np.random.default_rng(0)
    x_samples = rng.uniform(low=x_lb, high=x_ub, size=(n_samples, m))
    z0_samples = rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n))
    return x_samples, z0_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given the parameter sample `_x_samples` and the initial iterate sample `_z0_samples`, simulator mode runs the algorithm on each problem instance `_x_samples[i]` starting from `_z0_samples[i]`.
    After defining algorithm iterates, running the solve method `_VP.solve()` outputs the history of residuals at objective (fixed-point residual). The Verifier class instance `_VP` further provides the statistics summary which can be accessed via `_VP['mean']`, and so on.
    """)
    return


@app.cell
def _(
    A,
    K,
    Verifier,
    eta,
    m,
    n,
    nnls,
    np,
    x_lb,
    x_samples,
    x_ub,
    z0_lb,
    z0_samples,
    z0_ub,
):
    # Define Verifier instance with simulator mode on
    _VP = Verifier(method='simulator', obbt_solver=None)

    # Register the samples as the parameter of _VP instance
    _x_param = _VP.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=x_samples)
    _z0 = _VP.add_param(ndim=n, lb=z0_lb, ub=z0_ub, samples=z0_samples)

    # Define the algorithm iterates
    _z = [None] * (K + 1)
    _z[0] = _z0
    for _k in range(1, K + 1):
        _z[_k] = _VP.step.relu(_z[_k-1] - eta * A.T @ (A @ _z[_k-1] - _x_param))
        _VP.set_objective_norm(_z[_k] - _z[_k-1], ord=np.inf)

    # Run the simulation method
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
    _z_star_samples = np.array([nnls(A, _x_sample)[0] for _x_sample in x_samples])
    pepit_radius = np.max(np.linalg.norm(z0_samples - _z_star_samples, axis=1))
    return pepit_radius, simulation_residual_quantiles, simulation_residuals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the simulation of momentum-accelerated projected gradient descent as well.
    """)
    return


@app.cell
def _(
    A,
    K,
    Verifier,
    eta,
    m,
    n,
    np,
    x_lb,
    x_samples,
    x_ub,
    z0_lb,
    z0_samples,
    z0_ub,
):
    _VP = Verifier(method='simulator', obbt_solver=None)

    _x_param = _VP.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=x_samples)
    _z0 = _VP.add_param(ndim=n, lb=z0_lb, ub=z0_ub, samples=z0_samples)

    _z = [None] * (K + 1)
    _w = [None] * (K + 1)
    _z[0] = _z0
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
    ## Algorithm verification through verification problem (VP)

    We now obtain worst-case guarantee by solving the corresponding verification problem (VP).
    There are two approaches to formulating VP: mixed-integer linear program (MILP) formulation and (relaxed) semidefinite program (SDP) formulation.
    Passing the option `method='mip'` and `method='sdp'` will formulate respective verification problem forms, which are based on the following papers:

    - MIP-VP: [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming.](https://arxiv.org/abs/2412.11330)
    - SDP-VP: [Verification of First-Order Methods for Parametric Quadratic Optimization.](https://link.springer.com/article/10.1007/s10107-025-02261-w)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MIP-VP

    The following example formulates the verification problem as a mixed-integer linear program (MILP) and solves it to obtain exact worst-case fixed-point residual after $K$ iterations.
    We no longer have sampled $x$ and $z^0$ values, but have constraints on each parameter set $\mathcal{X}$ and initial iterate set $\mathcal{Z}$ defined using the parameter bounds: $\mathcal{X} = [\texttt{x\_lb},\, \texttt{x\_ub}]$ and $\mathcal{Z} = [\texttt{z0\_lb},\, \texttt{z0\_ub}]$.
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

        # Store the horizon-specific worst-case instance for each k.
        worst_case_x_param.append(_VP.get_solution_value(_x_param))
        worst_case_z0.append(_VP.get_solution_value(_z0))
    return all_residuals, worst_case_x_param, worst_case_z0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use simulator mode to further observe the algorithmic performance of obtained worst-case instance from verification problem. Adding the worst-case parameters and initial point as parameters, we can run the same algorithm and check if the obtained worst-case instance gives explicitly matching MIP bound.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We repeat the same procedure for accelerated projected gradient descent as well.
    """)
    return


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
def _(
    A,
    K,
    Verifier,
    eta,
    m,
    n,
    np,
    worst_case_x_param_momentum,
    worst_case_z0_momentum,
):
    # Simulate each horizon k with its own worst-case x and z0 from verification at that k.
    worst_case_residuals_momentum = []

    for _k in range(1, K + 1):
        _VP = Verifier(method='simulator', obbt_solver=None)
        _x_param = _VP.add_param(ndim=m, samples=[worst_case_x_param_momentum[_k - 1]])
        _x_k = np.asarray(worst_case_x_param_momentum[_k - 1])
        _x_param.lb = _x_param.ub = _x_k

        _z = [None] * (_k + 1)
        _w = [None] * (_k + 1)
        _z0_k = np.asarray(worst_case_z0_momentum[_k - 1])
        _z[0] = _VP.add_iterate(ndim=n, init=_z0_k)
        _z[0].lb = _z[0].ub = _z0_k
        _w[0] = _z[0]

        for _j in range(1, _k + 1):
            _beta_j = (_j - 1) / (_j + 2)
            _w[_j] = _VP.step.relu(_z[_j - 1] - eta * A.T @ (A @ _z[_j - 1] - _x_param))
            _z[_j] = _w[_j] + _beta_j * (_w[_j] - _w[_j - 1])

        worst_case_residuals_momentum.append(_VP.solve()[0][-1])

    worst_case_residuals_momentum = np.array(worst_case_residuals_momentum)
    return (worst_case_residuals_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    As we compare our results to the worst-case performance of projected gradient descent, we use performance estimation problem (PEP) to obtain such worst-case rate using the shared `PEPitRunner`.
    """)
    return


@app.cell
def _(K, L, PEPitRunner, eta, pepit_radius):
    all_residuals_PEP, _ = PEPitRunner().proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius, nonsmooth="indicator"
    )
    all_residuals_PEP_momentum, _ = PEPitRunner().accelerated_proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius, nonsmooth="indicator"
    )
    return all_residuals_PEP, all_residuals_PEP_momentum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SDP-VP

    The following example formulates the verification problem as a semidefinite program (SDP) and solves it to obtain valid upper bound to the worst-case fixed-point residual after $K$ iterations. Although the constraints, objective function, and iterate updates are quadratic, we use semidefinite lifting to relax the (nonconvex) quadratic constraints into convex linear matrix inequality.
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
    ### Numerical result: comparison against baselines

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
