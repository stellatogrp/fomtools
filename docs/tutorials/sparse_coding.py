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

    from PEPit import PEP, Point
    from PEPit.functions import ConvexLipschitzFunction, SmoothConvexFunction, ConvexFunction
    from PEPit.primitive_steps import proximal_step

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return (
        ConvexFunction,
        PEP,
        SmoothConvexFunction,
        Verifier,
        np,
        plt,
        proximal_step,
    )


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
    return D, lambd, n, p, x_lb, x_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the algorithm, we also choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $D^T D$.
    We initialize the algorithm with $S^0 = \{ s^0 \}$ where $s^0$ is the solution of unregularized problem ($\lambda = 0$) for a random sample.
    """)
    return


@app.cell
def _(D, np):
    L = np.real(np.max(np.linalg.eigvals(D.T @ D)))
    eta = 1 / L
    K = 10
    n_samples = 10000

    solver_params = {
        'OutputFlag': 0,
    }
    return K, L, eta, n_samples, solver_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first simulate ISTA on given problem instances, which are parameterized by the parameter $x \in \mathcal{X} = [x_{\text{lb}}, x_{\text{ub}}]$.
    We randomly sample $x$ from uniform distribution over $\mathcal{X}$.
    The output of `VP.solve(K=K)` will be an array of size `(n_samples, K)` whose $n$-th row is a list of infinity norm of residuals $s[k] - s[k-1]$ for $k=1,\ldots,K$.
    """)
    return


@app.cell
def _(D, K, Verifier, eta, lambd, n, n_samples, np, p, x_lb, x_ub):
    # The Simulator backend does not support soft_threshold, so we simulate 
    # ISTA explicitly using batched NumPy operations over sampled parameter instances.
    _rng = np.random.default_rng(0)
    _VP = Verifier(method='simulator')
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))
    _x_param = _VP.add_param(ndim=p, lb=x_lb, ub=x_ub, samples=_x_samples)

    _s = _VP.add_iterate(ndim=n, lb=0.0, ub=0.0, init=np.zeros(n))

    _VP.step.soft_threshold(_s - eta * D.T @ (D @ _s - _x_param), lambd=lambd*eta)
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

    simulation_residuals_ista = np.mean(_residuals, axis=0)

    # # numpy version
    # # Take average of residuals for iterate K=1 to K=K
    # simulation_residuals_ista = np.mean(_residuals, axis=0)
    # residuals = np.empty((n_samples, K), dtype=float)

    # # Soft-thresholding operator
    # def soft_threshold(v, threshold):
    #     return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    # for k in range(1, K+1):
    #     # ISTA step: s := ST_{lambda*eta}(s - eta * D^T(D*s - x))
    #     grad = (s @ D.T - _x_samples) @ D
    #     s_new = soft_threshold(s - eta * grad, lambd * eta)

    #     residuals[:, k - 1] = np.max(np.abs(s_new - s), axis=1)
    #     s = s_new

    # simulation_residuals_ista = np.mean(residuals, axis=0)
    return (simulation_residuals_ista,)


@app.cell
def _(D, K, eta, lambd, n, n_samples, np, p, x_lb, x_ub):
    # FISTA simulation: Fast Iterative Shrinkage-Thresholding Algorithm
    # Uses acceleration with momentum term
    _rng = np.random.default_rng(0)
    _x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))

    _y = np.zeros((n_samples, n), dtype=float)  # y variable
    _z = np.zeros((n_samples, n), dtype=float)  # z variable (momentum)
    _residuals_fista = np.empty((n_samples, K), dtype=float)

    _t = 1.0  # momentum parameter
    _y_prev = np.copy(_y)

    # Soft-thresholding operator
    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    for _k in range(1, K+1):
        # FISTA step:
        # y := ST_{lambda*eta}(z - eta * D^T(D*z - x))
        _y_new = soft_threshold(_z - eta * (_z @ D.T - _x_samples) @ D, lambd * eta)

        # Update momentum
        _t_new = (1.0 + np.sqrt(1.0 + 4.0 * _t**2)) / 2.0

        # Momentum update: z := y + ((t - 1) / t_new) * (y - y_prev)
        _z_new = _y_new + ((_t - 1.0) / _t_new) * (_y_new - _y_prev)

        _residuals_fista[:, _k-1] = np.max(np.abs(_y_new - _y), axis=1)
        _y_prev = np.copy(_y_new)
        _y = _y_new
        _z = _z_new
        _t = _t_new

    simulation_residuals_fista = np.mean(_residuals_fista, axis=0)
    return (simulation_residuals_fista,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now solve verification problem with MIP solver.
    """)
    return


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, solver_params, x_lb, x_ub):
    # from fomtools.proxable_functions import NormL1
    # regularizer = lambd * NormL1
    # regularizer = NormL1(lambd=lambd)

    _VP = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param = _VP.add_param(p, lb=x_lb, ub=x_ub)

    _s0 = np.zeros(n) # initialized at zero
    _s = [None for _ in range(K+1)]
    _s[0] = _s0

    all_residuals = []
    worst_case_x_param_ista = []
    for _k in range(1, K+1):
        _s[_k] = _VP.step.soft_threshold(_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), lambd=lambd*eta)
        # _s[_k] = VP.prox(function=regularizer, rhs=_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), gamma=eta)
        _VP.set_objective_norm([_s[_k] - _s[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)
        worst_case_x_param_ista.append(_VP.get_solution_value(_x_param))
    return all_residuals, worst_case_x_param_ista


@app.cell
def _(
    D,
    K,
    Verifier,
    eta,
    lambd,
    n,
    np,
    p,
    worst_case_x_param_ista,
    x_lb,
    x_ub,
):
    # Simulate each horizon with its own worst-case x from ISTA verification.
    worst_case_residuals_ista = []
    for _k in range(1, K + 1):
        _VP = Verifier(method='simulator')
        _x_param = _VP.add_param(ndim=p, lb=x_lb, ub=x_ub, samples=[worst_case_x_param_ista[_k - 1]])
        _s = _VP.add_iterate(ndim=n, lb=0.0, ub=0.0, init=np.zeros(n))
        _VP.step.soft_threshold(_s - eta * D.T @ (D @ _s - _x_param), lambd=lambd * eta)
        worst_case_residuals_ista.append(_VP.solve(K=_k)[0][-1])

    worst_case_residuals_ista = np.array(worst_case_residuals_ista)
    return (worst_case_residuals_ista,)


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, solver_params, x_lb, x_ub):
    # FISTA MIP verification
    _VP_fista = Verifier(method='mip', solver='gurobi', params=solver_params)
    _x_param_fista = _VP_fista.add_param(p, lb=x_lb, ub=x_ub)

    _y0 = np.zeros(n)
    _z0 = np.zeros(n)
    _y = [None for _ in range(K+1)]
    _z = [None for _ in range(K+1)]
    _y[0] = _y0
    _z[0] = _z0

    _t = [1.0]  # momentum parameter sequence
    all_residuals_fista = []
    worst_case_x_param_fista = []

    for _k in range(1, K+1):
        # FISTA step: y := ST_{lambda*eta}(z - eta * D^T(D*z - x))
        _y[_k] = _VP_fista.step.soft_threshold(_z[_k-1] - eta * D.T @ (D @ _z[_k-1] - _x_param_fista), lambd=lambd*eta)

        # Update momentum parameter
        _t_new = (1.0 + np.sqrt(1.0 + 4.0 * _t[-1]**2)) / 2.0
        _t.append(_t_new)

        # Momentum update: z := y + ((t - 1) / t_new) * (y - y_prev)
        _z[_k] = _y[_k] + ((_t[_k-1] - 1.0) / _t[_k]) * (_y[_k] - _y[_k-1])

        _VP_fista.set_objective_norm([_y[_k] - _y[_k-1]], np.inf)
        _res = _VP_fista.solve()
        all_residuals_fista.append(_res)
        worst_case_x_param_fista.append(_VP_fista.get_solution_value(_x_param_fista))
    return all_residuals_fista, worst_case_x_param_fista


@app.cell
def _(D, K, eta, lambd, n, np, worst_case_x_param_fista):
    # Simulate each horizon with its own worst-case x from FISTA verification.
    worst_case_residuals_fista = []

    def _soft_threshold(_v, _threshold):
        return np.sign(_v) * np.maximum(np.abs(_v) - _threshold, 0.0)

    for _k in range(1, K+1):
        _x_k = np.array(worst_case_x_param_fista[_k-1], dtype=float)

        _y = np.zeros(n, dtype=float)
        _z = np.zeros(n, dtype=float)
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


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_fista,
    plt,
    simulation_residuals_fista,
    simulation_residuals_ista,
    worst_case_residuals_fista,
    worst_case_residuals_ista,
):
    plt.plot(range(1, K+1), all_residuals, label='ISTA MIP bound')
    plt.plot(range(1, K+1), simulation_residuals_ista, '--', label='ISTA (sample avg)')
    plt.plot(range(1, K+1), worst_case_residuals_ista, '-.', color='k', linewidth=.6, label='worst-case ISTA instance')
    plt.plot(range(1, K+1), all_residuals_fista, label='FISTA MIP bound')
    plt.plot(range(1, K+1), simulation_residuals_fista, '--', label='FISTA (sample avg)')
    plt.plot(range(1, K+1), worst_case_residuals_fista, ':', color='k', linewidth=.8, label='worst-case FISTA instance')
    plt.suptitle('sparse coding example (MIP)')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


@app.cell
def _(ConvexFunction, K, L, PEP, SmoothConvexFunction, eta, proximal_step):
    all_residuals_PEP = []

    for _k in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(SmoothConvexFunction, L=L)
        # _g = _problem.declare_function(ConvexLipschitzFunction, M=lambd)
        _g = _problem.declare_function(ConvexFunction)
        _F = _f + _g

        # Define optimal solution
        _xs = _F.stationary_point()

        # Declare initial point s0
        _s0 = _problem.set_initial_point()
        _problem.set_initial_condition((_s0 - _xs)**2 <= 1.0)

        _s = [None for _ in range(K+1)]
        _s[0] = _s0
        for _k in range(1, _k+1):
            _s[_k], _, _ = proximal_step(_s[_k-1] - eta * _f.gradient(_s[_k-1]), _g, eta)
        # _problem.set_performance_metric(_F(_s[_k]) - _F(_xs))
        _problem.set_performance_metric((_s[_k] - _s[_k-1]) ** 2)
        _res = _problem.solve()
        all_residuals_PEP.append(_res)
    return (all_residuals_PEP,)


@app.cell
def _(ConvexFunction, K, L, PEP, SmoothConvexFunction, eta, proximal_step):
    all_residuals_PEP_fista = []

    for _k in range(1, K+1):
        _problem = PEP()
        _f = _problem.declare_function(SmoothConvexFunction, L=L)
        _g = _problem.declare_function(ConvexFunction)
        _F = _f + _g

        # Define optimal solution
        _xs = _F.stationary_point()

        # Declare initial point y0, z0
        _y0 = _problem.set_initial_point()
        _problem.set_initial_condition((_y0 - _xs)**2 <= 1.0)

        _y = [None for _ in range(K+1)]
        _z = [None for _ in range(K+1)]
        _y[0] = _y0
        _z[0] = _y0

        _t = [1.0]

        for _i in range(1, _k+1):
            # FISTA step: y := prox_g(z - eta * grad_f(z))
            _y[_i], _, _ = proximal_step(_z[_i-1] - eta * _f.gradient(_z[_i-1]), _g, eta)

            # Update momentum parameter
            _t_new = (1.0 + (1.0 + 4.0 * _t[-1]**2)**0.5) / 2.0
            _t.append(_t_new)

            # Momentum update: z := y + ((t - 1) / t_new) * (y - y_prev)
            _z[_i] = _y[_i] + ((_t[_i-1] - 1.0) / _t[_i]) * (_y[_i] - _y[_i-1])

        # _problem.set_performance_metric(_F(_y[_k]) - _F(_xs))
        _problem.set_performance_metric((_y[_k] - _y[_k-1]) ** 2)
        _res = _problem.solve()
        all_residuals_PEP_fista.append(_res)
    return (all_residuals_PEP_fista,)


@app.cell
def _(K, all_residuals_PEP, all_residuals_PEP_fista, plt):
    plt.figure()
    plt.plot(range(1, K+1), all_residuals_PEP, label='ISTA PEP bound')
    plt.plot(range(1, K+1), all_residuals_PEP_fista, label='FISTA PEP bound')
    plt.suptitle('sparse coding example (PEP)')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
