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
def _(D, K, eta, lambd, n, n_samples, np, p, x_lb, x_ub):
    # The Simulator backend does not support soft_threshold, so we simulate 
    # ISTA explicitly using batched NumPy operations over sampled parameter instances.
    _rng = np.random.default_rng(0)
    x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))

    s = np.zeros((n_samples, n), dtype=float)
    residuals = np.empty((n_samples, K), dtype=float)

    # Soft-thresholding operator
    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    for k in range(1, K + 1):
        # ISTA step: s := ST_{lambda*eta}(s - eta * D^T(D*s - x))
        grad = (s @ D.T - x_samples) @ D
        s_new = soft_threshold(s - eta * grad, lambd * eta)

        residuals[:, k - 1] = np.max(np.abs(s_new - s), axis=1)
        s = s_new

    simulation_residuals = np.mean(residuals, axis=0)
    return (simulation_residuals,)


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
    for _k in range(1, K+1):
        _s[_k] = _VP.step.soft_threshold(_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), lambd=lambd*eta)
        # _s[_k] = VP.prox(function=regularizer, rhs=_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), gamma=eta)
        _VP.set_objective_norm([_s[_k] - _s[_k-1]], np.inf)
        _res = _VP.solve()
        all_residuals.append(_res)
    return (all_residuals,)


@app.cell
def _(K, all_residuals, plt, simulation_residuals):
    plt.plot(range(1, K+1), all_residuals, label='ISTA MIP bound')
    plt.plot(range(1, K+1), simulation_residuals, '--', label='ISTA (in-sample avg)')
    plt.suptitle('sparse coding example (MIP)')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


@app.cell
def _(K, L, eta):
    from PEPit import PEP, Point
    from PEPit.functions import ConvexLipschitzFunction, SmoothConvexFunction, ConvexFunction
    from PEPit.primitive_steps import proximal_step

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
def _(K, all_residuals_PEP, plt):
    plt.figure()
    plt.plot(range(1, K+1), all_residuals_PEP, label='PEP bound')
    plt.suptitle('sparse coding example (PEP)')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, x_lb, x_ub):
    # from fomtools.proxable_functions import NormL1
    # _regularizer = lambd * NormL1
    # _regularizer = NormL1(lambd=lambd)

    _VP = Verifier(method='sdp', solver='clarabel', obbt_handler='highs')
    _x_param = _VP.add_param(p, lb=x_lb, ub=x_ub)

    _s0 = np.zeros(n) # initialized at zero
    _s = [None for _ in range(K + 1)]
    _s[0] = _s0

    all_residuals_sdp = []
    for _k in range(1, K+1):
        _s[_k] = _VP.step.soft_threshold(_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), lambd=lambd*eta)
        # _s[_k] = _VP.prox(function=_regularizer, rhs=_s[_k-1] - eta * D.T @ (D @ _s[_k-1] - _x_param), gamma=eta)
        _VP.set_objective_norm([_s[_k] - _s[_k-1]], 2)
        _res = _VP.solve()
        all_residuals_sdp.append(_res)
    return (all_residuals_sdp,)


@app.cell
def _(K, all_residuals_sdp, plt):
    plt.figure()
    plt.plot(range(1, K+1), all_residuals_sdp, label='SDP-VP bound')
    plt.suptitle('sparse coding example (SDP)')
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
