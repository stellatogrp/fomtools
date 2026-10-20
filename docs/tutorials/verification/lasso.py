import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import contextlib
    import io
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
        certify_lasso_pepit_radius,
        ista_step,
        make_lasso_data,
    )

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }

    # Gurobi prints a license banner through sys.stdout the first time its
    # default environment starts, before any OutputFlag can apply. Start the
    # environment here, in this hidden cell, with stdout swallowed so the
    # banner never lands in a visible cell.
    try:
        import gurobipy as _gp
    except ImportError:
        pass
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            _gp.Model()

    return (
        certify_lasso_pepit_radius,
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

    [Back to the verification tutorials](./)

    In this tutorial, we verify first-order methods for the sparse coding example from Section 4.1 of [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

    Consider the following problem of recovering the signal $z_\mathrm{true}$ from its noisy measurement $x = D z_\mathrm{true} + \varepsilon \in \mathbf{R}^p$:

    \begin{array}{ll}
        \text{minimize} & (1/2) \|Dz - x\|_2^2 + \lambda \|z\|_1,
    \end{array}

    where $D \in \mathbf{R}^{p \times n}$ is the dictionary matrix, $\varepsilon \in \mathbf{R}^p$, $\lambda > 0$ is a sparsity-inducing regularization parameter, $z \in \mathbf{R}^n$ is the decision variable, and $x \in \mathbf{R}^p$ is the problem parameter.

    ## First-order method: ISTA and FISTA

    This problem can be solved with ISTA and its accelerated variant FISTA.
    We provide the definition of ISTA iteration below:

    \begin{array}{ll}
        z^{k+1} = \mathbf{soft}_{\lambda \eta}\left(z^{k} - \eta D^T(Dz^{k} - x)\right),
    \end{array}

    where $\mathbf{soft}_\kappa(v)_i = \mathbf{sign}(v_i)\max\{|v_i| - \kappa, 0\}$ is the soft-thresholding operator, applied element-wise (the `soft_threshold` step).
    We choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $D^T D$.
    The FISTA iteration can be written as

    \begin{array}{rcl}
        y^{k+1} &=& \mathbf{soft}_{\lambda \eta}\left(z^{k} - \eta D^T(Dz^{k} - x)\right), \\
        z^{k+1} &=& y^{k+1} + \beta_{k+1}(y^{k+1} - y^{k}),
    \end{array}

    where $y^0 = z^0$, $t_0 = 1$, $t_k = (1 + \sqrt{1 + 4t_{k-1}^2})/2$, and $\beta_k = (t_{k-1}-1)/t_k$, so that $\beta_1 = 0$.

    The performance metric is the infinity norm $\|r^k\|_\infty$ of the fixed-point residual $r^k = y^k - y^{k-1}$ between consecutive proximal iterates, as in the paper; it keeps the verification problem a mixed-integer *linear* program.
    For ISTA, $y^k = z^k$, so $r^k = z^k - z^{k-1}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem parameters

    For the strongly convex example, we choose $p=20$ and $n=15$, and for the nonstrongly convex example, we choose $p = 15$, $n = 20$, and $\lambda = 10^{-1}$.
    To construct $D$, we randomly sample from i.i.d. Gaussian $\mathcal{N}(0, 1/p)$ the $20$% nonzeros of $D$ and normalize its columns.
    The true signal $z_\mathrm{true}$ is constructed by sampling each entry from i.i.d. Gaussian $\mathcal{N}(0, 1)$ with probability $0.1$ and setting $0$ otherwise. We sample $m = 100$ of those random sparse signals $z_\mathrm{true}$.
    Finally, we let the parameter set $\mathcal{X} = [x_\mathrm{lb}, x_\mathrm{ub}]$ be the tightest box containing all $x$'s.
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
    Every verification mode below (simulator and MIP-VP) drives this
    *same* `ista_step` recurrence and residual; only the `Verifier` backend and
    whether $(x, z^0)$ are sampled instances or bounded search variables change.
    Setting `momentum=True` selects the FISTA variant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    In this experiment, we verify the performance of ISTA and FISTA for sparse coding given the iteration budget $K>0$. We compare simulator statistics, exact MIP verification, PEPit bounds, and the in-sample worst-case instances.
    """)
    return


@app.cell
def _():
    K = 22  # Maximum iteration number
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    We first simulate ISTA on given problem instances, which are parameterized by the parameter $x$.
    Each problem instance is generated from randomly sampling $x$ from uniform distribution over $\mathcal{X} = [x_\mathrm{lb}, x_\mathrm{ub}]$ and the initial iterate $z^0$ from uniform distribution over $\mathcal{Z} = [z^0_\mathrm{lb}, z^0_\mathrm{ub}]$ (`s0_lb`, `s0_ub`, and `s0_samples` in the code). The samples are drawn once below and reused by every simulation cell so that all comparisons share the same instances.
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
    the residual norms $\|r^k\|_\infty$ for $k=1,\ldots,K$ of the $i$-th
    problem instance.
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
        ``(n_samples, K)``. ``ord`` defaults to ``inf``, matching
        :func:`ista_verification`; ``set_objective_norm(..., 2)`` would maximize
        the *squared* Euclidean norm, so with ``ord=2`` the returned residuals
        are in squared units. Summary statistics (``mean``, ``min``, ``max``,
        ...) are available on the returned instance ``VP`` via ``VP[...]``.
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
def _(K, ista_simulation, np, s0_samples, x_samples):
    _VP, _residuals = ista_simulation(x_samples, s0_samples, K, momentum=False, ord=np.inf)

    print(
        f"Summary statistics (at K={K}):\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
    )

    simulation_residuals_ista = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_ista = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return simulation_residual_quantiles_ista, simulation_residuals_ista


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
    with a MIP formulation, which gives the exact worst case. The following
    function rolls out the same `ista_step` recurrence over the horizon $K$
    and maximizes $\|r^k\|_\infty$ at each iteration.
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

        ``ord`` defaults to ``inf``, which keeps the MIP a MILP. A 2-norm
        objective is squared, so its value is square-rooted on return.
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
            val = VP.solve()
            residuals.append(np.sqrt(val) if ord == 2 else val)
        return VP, residuals

    return (ista_verification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MIP-VP
    We solve the exact MIP verification problem for both ISTA and its
    accelerated variant FISTA. The MIP searches over the measurement parameter
    $x$ and initial iterate $z^0$, and maximizes $\|r^k\|_\infty$ at each
    horizon.
    """)
    return


@app.cell
def _(K, ista_verification, np, solver_params):
    _, all_residuals = ista_verification(
        K, 'mip', momentum=False, ord=np.inf, solver='gurobi', params=solver_params,
    )
    _, all_residuals_fista = ista_verification(
        K, 'mip', momentum=True, ord=np.inf, solver='gurobi', params=solver_params,
    )
    return all_residuals, all_residuals_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    We compare our ISTA results to the worst-case performance returned by the
    performance estimation problem (PEP), using the shared `PEPitRunner`. The
    radius is certified rather than measured on the samples, so the curve is a
    valid upper bound for the whole box the verification problem searches; the
    price is that it is far looser than the sampled radius used previously
    ($R \approx 23.7$ instead of $3.0$, and the PEP scales as $R^2$).

    The FISTA curve uses a PEP written to match `ista_step` rather than the
    generic accelerated template. `PEPitRunner` exposes this through
    `variant="fista"`, which changes two things relative to the default
    `variant="simple"` used by the NNLS tutorial: the extrapolation weights come
    from the classical sequence $t_0 = 1$,
    $t_{k} = (1+\sqrt{1+4t_{k-1}^2})/2$, $\beta_k = (t_{k-1}-1)/t_{k}$ instead
    of the hardcoded $\beta_k = (k-1)/(k+2)$ (they differ precisely over the
    small $K$ plotted here: $\beta_2 = 0.282$ vs $0.250$,
    $\beta_3 = 0.434$ vs $0.400$), and the performance metric is the difference
    of the *proximal* iterates $y^k - y^{k-1}$ rather than of the extrapolated
    ones, which is the residual `ista_step` returns. Since $y^0 = z^0$ in
    the notebook, the same initial condition $\|z^0-z^\star\|_2 \le R$ covers
    both sequences.
    """)
    return


@app.cell(hide_code=True)
def _(K, L, certify_lasso_pepit_radius, eta, lambd, n, np, s0_lb, s0_ub, x_lb, x_ub):
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

    # Certified initial-condition radius: covers the whole box, not just the
    # drawn samples, so the PEP value below is a genuine upper bound.
    _pepit_radius = certify_lasso_pepit_radius(lambd, s0_lb, s0_ub, x_lb, x_ub, n)

    _runner = PEPitRunner()
    _pep, _ = _runner.proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius, nonsmooth="convex"
    )
    # variant="fista" matches ``ista_step``: classical t_k weights, and the
    # metric taken on the proximal iterates y^k rather than the extrapolated z^k.
    _pep_fista, _ = _runner.accelerated_proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius, nonsmooth="convex",
        variant="fista",
    )
    # PEPit's metric is ||x[k] - x[k-1]||^2, so its square root bounds
    # ||r^k||_2, and hence ||r^k||_inf, the norm the MIP maximizes.
    all_residuals_PEP = np.sqrt(np.asarray(_pep, dtype=float))
    all_residuals_PEP_fista = np.sqrt(np.asarray(_pep_fista, dtype=float))
    return all_residuals_PEP, all_residuals_PEP_fista


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical Result

    Now we plot the verification results against the PEPit bound and the
    in-sample simulation statistics, including the sample worst-case instance.
    The simulation and MIP curves report $\|r^k\|_\infty$. PEPit bounds
    $\|r^k\|_2$, which is also a bound on $\|r^k\|_\infty$ because
    $\|v\|_\infty \le \|v\|_2$.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    np,
    plt,
    simulation_residual_quantiles_ista,
    simulation_residuals_ista,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label=r'ISTA (PEPit, bounds $\|r^k\|_2 \geq \|r^k\|_\infty$)')
    plt.plot(_ks, all_residuals, label='ISTA (MIP worst-case)')
    _avg_line, = plt.plot(_ks, simulation_residuals_ista, '--', label='ISTA (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_ista[2], simulation_residual_quantiles_ista[0], alpha=.18, label='ISTA sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_ista[3], ':', color=_avg_line.get_color(), label='ISTA sample worst-case')
    # Bound the axis by the plotted *lines*, not by the 10%-90% band: the band's
    # lower edge decays to ~0, which would otherwise squeeze every curve into the
    # top of the figure. Nothing drawn as a line is clipped.
    _lines = np.concatenate([
        all_residuals,
        simulation_residuals_ista,
        simulation_residual_quantiles_ista[3],
    ])
    plt.ylim(bottom=0.5 * float(np.min(_lines)))
    plt.suptitle('sparse coding example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel(r'Worst-case fixed-point residual $\|r^k\|_\infty$')
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
    np,
    plt,
    simulation_residual_quantiles_fista,
    simulation_residuals_fista,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_fista, label=r'FISTA (PEPit, bounds $\|r^k\|_2 \geq \|r^k\|_\infty$)')
    plt.plot(_ks, all_residuals_fista, label='FISTA (MIP worst-case)')
    _avg_line, = plt.plot(_ks, simulation_residuals_fista, '--', label='FISTA (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_fista[2], simulation_residual_quantiles_fista[0], alpha=.18, label='FISTA sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_fista[3], ':', color=_avg_line.get_color(), label='FISTA sample worst-case')
    # Bound the axis by the plotted *lines*, not by the 10%-90% band: the band's
    # lower edge decays to ~0, which would otherwise squeeze every curve into the
    # top of the figure. Nothing drawn as a line is clipped.
    _lines = np.concatenate([
        all_residuals_fista,
        simulation_residuals_fista,
        simulation_residual_quantiles_fista[3],
    ])
    plt.ylim(bottom=0.5 * float(np.min(_lines)))
    plt.suptitle('sparse coding FISTA example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel(r'Worst-case fixed-point residual $\|r^k\|_\infty$')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
