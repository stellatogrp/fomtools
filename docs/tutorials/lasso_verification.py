import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "fomtools").is_dir():
            repo_root = candidate
            break
    else:
        repo_root = start

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from docs.tutorials.helper_functions.lasso import (
        estimate_lasso_pepit_radius,
        ista_step,
        make_lasso_data,
    )

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }
    return (
        estimate_lasso_pepit_radius,
        ista_step,
        make_lasso_data,
        mo,
        np,
        plt,
        solver_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem setup: Sparse coding

    In this tutorial, we verify first-order methods for the sparse coding example from Section 4.1 of [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

    Consider the following problem of recovering the signal $z^\star$: from its noisy measurement $x = D z^\star + \varepsilon \in {\bf R}^p$:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Dz - x \rVert_2^2 + \lambda \lVert z \rVert_1,
    \end{array}

    where $D \in {\bf R}^{p \times n}$ is the dictionary matrix, $\varepsilon \in {\bf R}^p$, $\lambda > 0$ is a sparsity-inducing regularization parameter, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^p$ is the problem parameter.

    ## First-order method: ISTA and FISTA

    This problem can be solved with ISTA and its accelerated variant FISTA.
    We provide the definition of ISTA iteration below:

    \begin{array}{ll}
        s^{k+1} = {\cal T}_{\lambda \eta} ( (I - \eta D^T D) s^{k} + \eta D^T x),
    \end{array}

    where ${\cal T}_{\lambda}$ is a soft-thresholding operator applied element-wisely.
    We choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $D^T D$.
    The FISTA iteration can be written as

    \begin{array}{rcl}
        y^{k} &=& {\cal T}_{\lambda \eta}\left(z^{k-1} - \eta D^T(Dz^{k-1} - x)\right), \\
        z^{k} &=& y^{k} + \frac{t_{k-1}-1}{t_k}(y^{k} - y^{k-1}),
    \end{array}

    where $t_0 = 1$ and $t_k = (1 + \sqrt{1 + 4t_{k-1}^2})/2$.

    The performance metric is the infinity norm of the fixed-point residual between consecutive iterates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem parameters

    For the strongly convex example, we choose $p=20$ and $n=15$, and for the nonstrongly convex example, we choose $p = 15$, $n = 20$, and $\lambda = 10^{-1}$.
    To construct $D$, we randomly sample from i.i.d. Gaussian ${\cal N}(0, 1/p)$ the $20$% nonzeros of $D$ and normalize its columns.
    The true signal $z^\star$ is constructed by sampling each entry from i.i.d. Gaussian ${\cal N}(0, 1)$ with probability $0.1$ and setting $0$ otherwise. We sample $m = 100$ of those random sparse signals $z^\star$.
    Finally, we let the parameter set $X$ to be the tightest hypercube containing all $x$'s.
    """)
    return


@app.cell
def _(make_lasso_data):
    # p, n, lambd = 15, 20, 1e-1
    p, n, lambd = 3, 4, 1e-1
    # num_star = 100
    num_star = 20
    D, x_lb, x_ub, L, eta, lambd = make_lasso_data(
        p=p, n=n, lambd=lambd, num_signals=num_star, seed=0
    )
    s0_lb, s0_ub = -1.0, 1.0
    return D, L, eta, lambd, n, p, s0_lb, s0_ub, x_lb, x_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every verification mode below (simulator, MIP-VP, and SDP-VP) drives this
    *same* `ista_step` recurrence and residual; only the `Verifier` backend and
    whether $(x, s^0)$ are sampled instances or bounded search variables change.
    Setting `momentum=True` selects the FISTA variant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    In this experiment, we verify the performance of ISTA and FISTA for sparse coding given the iteration budget $K>0$. We compare simulator statistics, exact MIP verification, SDP verification, PEPit bounds, and the in-sample worst-case instances.
    """)
    return


@app.cell
def _():
    K = 15  # Maximum iteration number
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    We first simulate ISTA on given problem instances, which are parameterized by the parameter $x$.
    Each problem instance is generated from randomly sampling $x$ from uniform distribution over $[\texttt{x\_lb}, \texttt{x\_ub}]$ and $s^0$ from uniform distribution over $[\texttt{s0\_lb}, \texttt{s0\_ub}]$. The samples are drawn once below and reused by every simulation cell so that all comparisons share the same instances.
    """)
    return


@app.cell
def _(n, np, p, s0_lb, s0_ub, x_lb, x_ub):
    n_samples = 10000
    _rng = np.random.default_rng(0)
    x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))
    s0_samples = _rng.uniform(low=s0_lb, high=s0_ub, size=(n_samples, n))
    return s0_samples, x_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulation is run through the following function. The output of
    `_VP.solve()` is an array of size `(n_samples, K)` whose $i$-th row lists
    the infinity norm of residuals for $k=1,\ldots,K$ of the $i$-th problem
    instance.
    """)
    return


@app.cell
def _(D, eta, ista_step, lambd, n, np, p):
    from fomtools.verifier import Verifier

    def ista_simulation(x_samples, s0_samples, K, momentum=False, ord=np.inf):
        """Simulate ISTA on sampled instances.

        Builds a simulator-mode :class:`Verifier`, registers the parameter
        ``x`` and initial iterate ``s0`` as sampled parameters, rolls out the
        recurrence with ``ista_step``, and simulates via ``VP.solve()``. Each
        ``*_samples`` array has shape ``(n_samples, dim)`` (a single-row array
        replays one instance), and the returned ``residuals`` has shape
        ``(n_samples, K)``. Summary statistics (``mean``, ``min``, ``max``, ...)
        are available on the returned instance ``VP`` via ``VP[...]``.
        """
        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=p, samples=x_samples)
        s0 = VP.add_param(n, samples=s0_samples)

        z, y, t = s0, s0, 1.0
        for k in range(1, K + 1):
            z, y, t, residual = ista_step(
                VP, D, eta, lambd, z, y, x_param, t, k=k, momentum=momentum
            )
            VP.set_objective_norm(residual, ord)

        residuals = VP.solve()
        return VP, residuals

    return Verifier, ista_simulation


@app.cell
def _(
    K,
    ista_simulation,
    np,
    s0_samples,
    x_samples,
):
    _VP, _residuals = ista_simulation(x_samples, s0_samples, K, momentum=False, ord=np.inf)

    print(
        f"Summary statistics:\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
    )

    simulation_residuals_ista = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_ista = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return (
        simulation_residual_quantiles_ista,
        simulation_residuals_ista,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We further simulate FISTA, reusing the same `ista_simulation` helper and samples with `momentum=True`.
    """)
    return


@app.cell
def _(K, ista_simulation, np, s0_samples, x_samples):
    _, _residuals = ista_simulation(x_samples, s0_samples, K, momentum=True, ord=np.inf)
    simulation_residuals_fista = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_fista = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return simulation_residual_quantiles_fista, simulation_residuals_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm verification through verification problem (VP)

    We now obtain worst-case guarantees by solving the verification problem
    with a MIP formulation (exact) and an SDP relaxation (valid upper bound).
    Both formulations use the following function, which rolls out the same
    `ista_step` recurrence over the horizon $K$ and maximizes the fixed-point
    residual at each iteration.
    """)
    return


@app.cell
def _(D, Verifier, eta, ista_step, lambd, n, np, p, s0_lb, s0_ub, x_lb, x_ub):
    def ista_verification(K, method, momentum=False, ord=np.inf, **verifier_kwargs):
        """Solve the ISTA/FISTA verification problem for one backend.

        Builds a ``Verifier`` in ``method`` mode, registers the parameter box
        ``[x_lb, x_ub]`` and the initial iterate box ``[s0_lb, s0_ub]``, rolls
        out ``ista_step`` over the horizon ``K``, and returns ``(VP, residuals)``
        with one worst-case residual per iteration. Extra ``verifier_kwargs``
        (``solver``, ``params``, ...) are forwarded to the ``Verifier``
        constructor.
        """
        VP = Verifier(method=method, **verifier_kwargs)
        x_param = VP.add_param(p, lb=x_lb, ub=x_ub)
        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n, lb=s0_lb, ub=s0_ub)
        y, t = z[0], 1.0
        residuals = []
        for k in range(1, K + 1):
            z, y, t, residual = ista_step(
                VP, D, eta, lambd, z, y, x_param, t, k=k, momentum=momentum
            )
            VP.set_objective_norm(residual, ord)
            residuals.append(VP.solve())
        return VP, residuals

    return (ista_verification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MIP-VP
    We solve the exact MIP verification problem for both ISTA and its
    accelerated variant FISTA. The MIP searches over the measurement parameter
    $x$ and initial iterate $s^0$, and maximizes the fixed-point residual at
    each horizon.
    """)
    return


@app.cell
def _(K, ista_verification, solver_params):
    _, all_residuals = ista_verification(
        K, 'mip', momentum=False, solver='gurobi', params=solver_params,
    )
    _, all_residuals_fista = ista_verification(
        K, 'mip', momentum=True, solver='gurobi', params=solver_params,
    )
    return all_residuals, all_residuals_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SDP-VP

    Here, we compare against valid upper bounds obtained from the SDP-relaxed verification problem, for both ISTA and FISTA.
    """)
    return


@app.cell
def _(K, ista_verification):
    _, all_residuals_sdp = ista_verification(
        K, 'sdp', momentum=False, solver='clarabel'
    )
    _, all_residuals_sdp_fista = ista_verification(
        K, 'sdp', momentum=True, solver='clarabel'
    )
    return all_residuals_sdp, all_residuals_sdp_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    We compare our results to the worst-case performance of ISTA and FISTA
    returned by the performance estimation problem (PEP), using the shared
    `PEPitRunner`.
    """)
    return


@app.cell(hide_code=True)
def _(
    D,
    K,
    L,
    estimate_lasso_pepit_radius,
    eta,
    lambd,
    s0_samples,
    x_samples,
):
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

    # Empirical initial-condition radius for the PEPit comparison.
    _pepit_radius = estimate_lasso_pepit_radius(
        D,
        eta,
        lambd,
        s0_samples,
        x_samples,
    )
    all_residuals_PEP, _ = PEPitRunner().proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius, nonsmooth="convex"
    )
    all_residuals_PEP_fista, _ = PEPitRunner().accelerated_proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius, nonsmooth="convex"
    )
    return all_residuals_PEP, all_residuals_PEP_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical Result

    Now we plot the verification results against the PEPit bound, the SDP
    bound, and the in-sample simulation statistics, including the sample
    worst-case instance.
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
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='ISTA (PEPit)')
    plt.plot(_ks, all_residuals, label='ISTA (MIP worst-case)')
    plt.plot(_ks, all_residuals_sdp, label='ISTA (SDP bound)')
    _avg_line, = plt.plot(_ks, simulation_residuals_ista, '--', label='ISTA (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_ista[2], simulation_residual_quantiles_ista[0], alpha=.18, label='ISTA sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_ista[3], ':', color=_avg_line.get_color(), label='ISTA sample worst-case')
    plt.suptitle('sparse coding example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.ylim([1e-3, 1e2])
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
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_fista, label='FISTA (PEPit)')
    plt.plot(_ks, all_residuals_fista, label='FISTA (MIP worst-case)')
    plt.plot(_ks, all_residuals_sdp_fista, label='FISTA (SDP bound)')
    _avg_line, = plt.plot(_ks, simulation_residuals_fista, '--', label='FISTA (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_fista[2], simulation_residual_quantiles_fista[0], alpha=.18, label='FISTA sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_fista[3], ':', color=_avg_line.get_color(), label='FISTA sample worst-case')
    plt.suptitle('sparse coding FISTA example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    plt.ylim([1e-3, 5e1])
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
