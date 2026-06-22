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

    from fomtools.proxable_functions import NormL1
    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
    }
    return NormL1, PEPitRunner, Verifier, np, plt, solver_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial, we illustrate the sparse coding example in the Section 4.1 of the paper: [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

    Consider the following problem of recovering the signal $z^\star$: from its noisy measurement $x = D z^\star + \varepsilon \in {\bf R}^p$:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Dz - x \rVert_2^2 + \lambda \lVert z \rVert_1,
    \end{array}

    where $D \in {\bf R}^{p \times n}$ is the dictionary matrix, $\varepsilon \in {\bf R}^p$, $\lambda > 0$ is a sparsity-inducing regularization parameter, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^p$ is the problem parameter.

    This problem can be solved with ISTA and its accelerated variant FISTA.
    We provide the definition of ISTA iteration below:

    \begin{array}{ll}
        s^{k+1} = {\cal T}_{\lambda \eta} ( (I - \eta D^T D) s^{k} + \eta D^T x),
    \end{array}

    where ${\cal T}_{\lambda}$ is a soft-thresholding operator applied element-wisely.
    We choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $D^T D$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the strongly convex example, we choose $p=20$ and $n=15$, and for the nonstrongly convex example, we choose $p = 15$, $n = 20$, and $\lambda = 10^{-1}$.
    To construct $D$, we randomly sample from i.i.d. Gaussian ${\cal N}(0, 1/p)$ the $20$% nonzeros of $D$ and normalize its columns.
    The true signal $z^\star$ is constructed by sampling each entry from i.i.d. Gaussian ${\cal N}(0, 1)$ with probability $0.1$ and setting $0$ otherwise. We sample $m = 100$ of those random sparse signals $z^\star$.
    Finally, we let the parameter set $X$ to be the tightest hypercube containing all $x$'s.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    # p, n, lambd = 15, 20, 1e-1
    p, n, lambd = 3, 4, 1e-1
    D_value = np.random.normal(size=(p, n))
    D_mask = np.random.binomial(n=1, p=0.2, size=(p, n))
    D = np.multiply(D_value, D_mask)

    # signal z_star
    # num_star = 100
    num_star = 20
    z_star_value = np.random.normal(size=(n, num_star))
    z_star_mask = np.random.binomial(n=1, p=0.1, size=(n, num_star))
    z_star = np.multiply(z_star_value, z_star_mask)

    # noise err
    err = np.random.normal(scale=1e-2, size=(p, num_star))

    # parameter x = D z_star + err
    x = D @ z_star + err

    # parameter set X = [x_lb, x_ub]
    x_lb, x_ub = np.min(x, axis=1), np.max(x, axis=1)

    L = np.real(np.max(np.linalg.eigvals(D.T @ D)))
    eta = 1 / L
    s0_lb, s0_ub = -1.0, 1.0
    return D, L, eta, lambd, n, p, s0_lb, s0_ub, x_lb, x_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this experiment, we verify the performance of first-order methods solving sparse coding problem given the iteration budget $K>0$.
    """)
    return


@app.cell
def _():
    K = 7
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate ISTA on given problem instances, which are parameterized by the parameter $x$.
    Each problem instance is generated from randomly sampling $x$ from uniform distribution over $[\texttt{x\_lb}, \texttt{x\_ub}]$ and $s^0$ from uniform distribution over $[\texttt{s0\_lb}, \texttt{s0\_ub}]$.
    The output of `VP.solve(K=K)` will be an array of size `(n_samples, K)` whose $n$-th row is a list of infinity norm of residuals $s[k] - s[k-1]$ for $k=1,\ldots,K$ of $n$-th problem instance.
    """)
    return


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, s0_lb, s0_ub, x_lb, x_ub):
    n_samples = 10000

    _rng = np.random.default_rng(0)
    _VP = Verifier(method='simulator', obbt_solver=None)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))
    _s0_samples = _rng.uniform(low=s0_lb, high=s0_ub, size=(n_samples, n))
    _x_param = _VP.add_param(ndim=p, samples=_x_samples)
    _x_param.lb, _x_param.ub = x_lb, x_ub

    _s = [None] * (K + 1)
    _s[0] = _VP.add_param(n, samples=_s0_samples)
    _s[0].lb, _s[0].ub = np.full(n, s0_lb), np.full(n, s0_ub)

    for _k in range(1, K + 1):
        _s[_k] = _VP.step.soft_threshold(_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), lambd=lambd*eta)
        _VP.set_objective_norm(_s[_k] - _s[_k-1], ord=np.inf)
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

    simulation_residuals_ista = np.mean(_residuals, axis=0)
    simulation_residual_quantiles_ista = np.quantile(_residuals, [0.90, 0.50, 0.10], axis=0)

    _s_star_samples = np.zeros((n_samples, n), dtype=float)
    for _ in range(2000):
        _grad = (_s_star_samples @ D.T - _x_samples) @ D
        _s_next_samples = np.sign(_s_star_samples - eta * _grad) * np.maximum(
            np.abs(_s_star_samples - eta * _grad) - lambd * eta,
            0.0,
        )
        if np.max(np.linalg.norm(_s_next_samples - _s_star_samples, axis=1)) <= 1e-10:
            _s_star_samples = _s_next_samples
            break
        _s_star_samples = _s_next_samples
    pepit_radius = np.max(np.linalg.norm(_s0_samples - _s_star_samples, axis=1))
    return (
        n_samples,
        pepit_radius,
        simulation_residual_quantiles_ista,
        simulation_residuals_ista,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also consider FISTA, the accelerated variant of ISTA. The method is
    written as

    \begin{array}{rcl}
        y^{k} &=& {\cal T}_{\lambda \eta}\left(z^{k-1} - \eta D^T(Dz^{k-1} - x)\right), \\
        z^{k} &=& y^{k} + \frac{t_{k-1}-1}{t_k}(y^{k} - y^{k-1}),
    \end{array}

    where $t_0 = 1$ and $t_k = (1 + \sqrt{1 + 4t_{k-1}^2})/2$.
    """)
    return


@app.cell
def _(
    D,
    K,
    Verifier,
    eta,
    lambd,
    n,
    n_samples,
    np,
    p,
    s0_lb,
    s0_ub,
    x_lb,
    x_ub,
):
    _VP = Verifier(method='simulator', obbt_solver=None)
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))
    _s0_samples = _rng.uniform(low=s0_lb, high=s0_ub, size=(n_samples, n))
    _x_param = _VP.add_param(ndim=p, samples=_x_samples)
    _x_param.lb, _x_param.ub = x_lb, x_ub

    _y = [None] * (K + 1)
    _z = [None] * (K + 1)
    _y[0] = _VP.add_param(n, samples=_s0_samples)
    _y[0].lb, _y[0].ub = np.full(n, s0_lb), np.full(n, s0_ub)
    _z[0] = _y[0]
    _t = [1.0]
    for _k in range(1, K+1):
        _y[_k] = _VP.step.soft_threshold(
            _z[_k-1] - eta * D.T @ (D @ _z[_k-1] - _x_param),
            lambd=lambd*eta,
        )
        _t_new = (1.0 + np.sqrt(1.0 + 4.0 * _t[-1]**2)) / 2.0
        _t.append(_t_new)
        _z[_k] = _y[_k] + ((_t[_k-1] - 1.0) / _t[_k]) * (_y[_k] - _y[_k-1])
        _VP.set_objective_norm(_y[_k] - _y[_k-1], ord=np.inf)

    _residuals = _VP.solve()
    simulation_residuals_fista = np.mean(_residuals, axis=0)
    simulation_residual_quantiles_fista = np.quantile(_residuals, [0.90, 0.50, 0.10], axis=0)
    return simulation_residual_quantiles_fista, simulation_residuals_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now solve verification problem with MIP solver.
    """)
    return


@app.cell
def _(
    D,
    K,
    NormL1,
    Verifier,
    eta,
    lambd,
    n,
    np,
    p,
    s0_lb,
    s0_ub,
    solver_params,
    x_lb,
    x_ub,
):
    regularizer = NormL1(lambd=lambd)

    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(p, lb=x_lb, ub=x_ub)

    _s = [None for _ in range(K+1)]
    _s[0] = _VP.add_iterate(n, lb=s0_lb, ub=s0_ub)

    all_residuals = []
    worst_case_s0_ista = []
    worst_case_x_param_ista = []
    for _k in range(1, K+1):
        _s[_k] = _VP.prox(regularizer, _s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), gamma=eta)
        _VP.set_objective_norm([_s[_k] - _s[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)
        worst_case_s0_ista.append(_VP.get_solution_value(_s[0]))
        worst_case_x_param_ista.append(_VP.get_solution_value(_x_param))
    return (
        all_residuals,
        regularizer,
        worst_case_s0_ista,
        worst_case_x_param_ista,
    )


@app.cell
def _(
    D,
    K,
    Verifier,
    eta,
    n,
    np,
    p,
    regularizer,
    worst_case_s0_ista,
    worst_case_x_param_ista,
    x_lb,
    x_ub,
):
    # Simulate each horizon with its own worst-case x from ISTA verification.
    worst_case_residuals_ista = []
    for _k in range(1, K + 1):
        _VP = Verifier(method='simulator')
        _x_param = _VP.add_param(ndim=p, lb=x_lb, ub=x_ub, samples=[worst_case_x_param_ista[_k - 1]])
        _s = [None] * (_k + 1)
        _s0_k = np.asarray(worst_case_s0_ista[_k - 1])
        _s[0] = _VP.add_iterate(ndim=n, init=_s0_k)
        _s[0].lb = _s[0].ub = _s0_k
        for _j in range(1, _k + 1):
            _s[_j] = _VP.prox(regularizer, _s[_j-1] - eta * D.T @ (D @ _s[_j-1] - _x_param), gamma=eta)
        worst_case_residuals_ista.append(_VP.solve()[0][-1])

    worst_case_residuals_ista = np.array(worst_case_residuals_ista)
    return (worst_case_residuals_ista,)


@app.cell
def _(
    D,
    K,
    Verifier,
    eta,
    n,
    np,
    p,
    regularizer,
    s0_lb,
    s0_ub,
    solver_params,
    x_lb,
    x_ub,
):
    # FISTA MIP verification
    _VP_fista = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param_fista = _VP_fista.add_param(p, lb=x_lb, ub=x_ub)

    _y = [None for _ in range(K+1)]
    _z = [None for _ in range(K+1)]
    _y[0] = _VP_fista.add_iterate(n, lb=s0_lb, ub=s0_ub)
    _z[0] = _y[0]

    _t = [1.0]  # momentum parameter sequence
    all_residuals_fista = []
    worst_case_s0_fista = []
    worst_case_x_param_fista = []

    for _k in range(1, K+1):
        _y[_k] = _VP_fista.prox(regularizer, _z[_k-1] - eta * D.T @ (D @ _z[_k-1] - _x_param_fista), gamma=eta)


        # Update momentum parameter
        _t_new = (1.0 + np.sqrt(1.0 + 4.0 * _t[-1]**2)) / 2.0
        _t.append(_t_new)

        # Momentum update: z := y + ((t - 1) / t_new) * (y - y_prev)
        _z[_k] = _y[_k] + ((_t[_k-1] - 1.0) / _t[_k]) * (_y[_k] - _y[_k-1])

        _VP_fista.set_objective_norm([_y[_k] - _y[_k-1]], np.inf)
        _res = _VP_fista.solve()
        all_residuals_fista.append(_res)
        worst_case_s0_fista.append(_VP_fista.get_solution_value(_y[0]))
        worst_case_x_param_fista.append(_VP_fista.get_solution_value(_x_param_fista))
    return all_residuals_fista, worst_case_s0_fista, worst_case_x_param_fista


@app.cell
def _(D, K, eta, lambd, np, worst_case_s0_fista, worst_case_x_param_fista):
    # Simulate each horizon with its own worst-case x from FISTA verification.
    worst_case_residuals_fista = []

    def _soft_threshold(_v, _threshold):
        return np.sign(_v) * np.maximum(np.abs(_v) - _threshold, 0.0)

    for _k in range(1, K+1):
        _x_k = np.array(worst_case_x_param_fista[_k-1], dtype=float)
        _s0_k = np.array(worst_case_s0_fista[_k-1], dtype=float)

        _y = _s0_k.copy()
        _z = _s0_k.copy()
        _y_prev = _y.copy()
        _t = 1.0
        _last_residual = 0.0

        for _j in range(1, _k + 1):
            _y_new = _soft_threshold(_z - eta * (D @ _z - _x_k) @ D, lambd * eta)
            _t_new = (1.0 + np.sqrt(1.0 + 4.0 * _t**2)) / 2.0
            _z_new = _y_new + ((_t - 1.0) / _t_new) * (_y_new - _y_prev)

            _last_residual = np.max(np.abs(_y_new - _y))
            _y_prev = _y_new.copy()
            _y = _y_new
            _z = _z_new
            _t = _t_new

        worst_case_residuals_fista.append(_last_residual)

    worst_case_residuals_fista = np.array(worst_case_residuals_fista)
    return (worst_case_residuals_fista,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compare our results to the worst-case performance of ISTA and FISTA
    returned by the performance estimation problem (PEP), using the shared
    `PEPitRunner`.
    """)
    return


@app.cell
def _(K, L, PEPitRunner, eta, pepit_radius):
    all_residuals_PEP, _ = PEPitRunner().proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius, nonsmooth="convex")
    return (all_residuals_PEP,)


@app.cell
def _(K, L, PEPitRunner, eta, pepit_radius):
    all_residuals_PEP_fista, _ = PEPitRunner().accelerated_proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=pepit_radius, nonsmooth="convex")
    return (all_residuals_PEP_fista,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we compare the result obtained from solving SDP-relaxed verification problem.
    """)
    return


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, s0_lb, s0_ub, x_lb, x_ub):
    _VP = Verifier(method='sdp', solver='clarabel')
    _x_param = _VP.add_param(p, lb=x_lb, ub=x_ub)

    _s = [None for _ in range(K+1)]
    _s[0] = _VP.add_iterate(n, lb=s0_lb, ub=s0_ub)

    all_residuals_sdp = []
    for _k in range(1, K+1):
        _s[_k] = _VP.step.soft_threshold(
            _s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param),
            lambd=lambd * eta,
        )
        _VP.set_objective_norm([_s[_k] - _s[_k-1]], np.inf)
        all_residuals_sdp.append(_VP.solve())
    return (all_residuals_sdp,)


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, s0_lb, s0_ub, x_lb, x_ub):
    _VP = Verifier(method='sdp', solver='clarabel')
    _x_param = _VP.add_param(p, lb=x_lb, ub=x_ub)

    _y = [None for _ in range(K+1)]
    _z = [None for _ in range(K+1)]
    _y[0] = _VP.add_iterate(n, lb=s0_lb, ub=s0_ub)
    _z[0] = _y[0]
    _t = [1.0]

    all_residuals_sdp_fista = []
    for _k in range(1, K+1):
        _y[_k] = _VP.step.soft_threshold(
            _z[_k-1] - eta * D.T @ (D @ _z[_k-1] - _x_param),
            lambd=lambd * eta,
        )
        _t_new = (1.0 + np.sqrt(1.0 + 4.0 * _t[-1]**2)) / 2.0
        _t.append(_t_new)
        _z[_k] = _y[_k] + ((_t[_k-1] - 1.0) / _t[_k]) * (_y[_k] - _y[_k-1])
        _VP.set_objective_norm([_y[_k] - _y[_k-1]], np.inf)
        all_residuals_sdp_fista.append(_VP.solve())
    return (all_residuals_sdp_fista,)


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
    simulation_residual_quantiles_ista,
    simulation_residuals_ista,
    worst_case_residuals_ista,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='ISTA (PEPit)')
    plt.plot(_ks, all_residuals, label='ISTA (MIP worst-case)')
    plt.plot(_ks, all_residuals_sdp, label='ISTA (SDP bound)')
    plt.plot(_ks, simulation_residuals_ista, '--', label='ISTA (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_ista[2], simulation_residual_quantiles_ista[0], alpha=.18, label='ISTA simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals_ista, '-.', color='k', linewidth=.7, label='worst-case ISTA instance')
    plt.suptitle('sparse coding example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.ylim([1e-2, 1e2])
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also plot the verification results for FISTA.
    """)
    return


@app.cell
def _(
    K,
    all_residuals_PEP_fista,
    all_residuals_fista,
    all_residuals_sdp_fista,
    plt,
    simulation_residual_quantiles_fista,
    simulation_residuals_fista,
    worst_case_residuals_fista,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_fista, label='FISTA (PEPit)')
    plt.plot(_ks, all_residuals_fista, label='FISTA (MIP worst-case)')
    plt.plot(_ks, all_residuals_sdp_fista, label='FISTA (SDP bound)')
    plt.plot(_ks, simulation_residuals_fista, '--', label='FISTA (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_fista[2], simulation_residual_quantiles_fista[0], alpha=.18, label='FISTA simulation 10%-90% band')
    plt.plot(_ks, worst_case_residuals_fista, '-.', color='k', linewidth=.7, label='worst-case FISTA instance')
    plt.suptitle('sparse coding FISTA example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.ylim([1e-3, 5e1])
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
