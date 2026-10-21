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

    from docs.tutorials.helper_functions.nnls import (
        certify_nnls_pepit_radius,
        make_nnls_matrix,
        pgd_step,
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
        certify_nnls_pepit_radius,
        make_nnls_matrix,
        mo,
        np,
        pgd_step,
        plt,
        solver_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem setup: Nonnegative Least-squares (NNLS)

    [Back to the verification tutorials](./)

    In this tutorial, we show how to use our package to verify the performance of first-order methods solving the nonnegative least-squares (NNLS) problem:

    \begin{array}{ll}
        \text{minimize} & (1/2) \|Az - x\|_2^2 \\ \text{subject to} & z \ge 0,
    \end{array}

    where $A \in \mathbf{R}^{m \times n}$ with $m \ge n$, $z \in \mathbf{R}^n$ is the decision variable, and $x \in \mathbf{R}^m$ is the problem parameter.

    ## Problem parameters

    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ by the box $\mathcal{X} = [10, 11]^m$.
    """)
    return


@app.cell
def _(make_nnls_matrix):
    m, n = 5, 3
    A = make_nnls_matrix(m=m, n=n)
    x_lb, x_ub = 10.0, 11.0    # parameter set
    return A, m, n, x_lb, x_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance metric

    Given the sequence of iterates $\{z^k\}_{k=0}^K$ generated from the algorithm of our choice, we quantify its performance through measuring how fast the fixed-point residual $r^K = z^K - z^{K-1}$ decreases, given the iteration budget $K>0$.
    We impose the initial condition $z^0 \in \mathcal{Z} := [z^0_\mathrm{lb},\, z^0_\mathrm{ub}]$ (`z0_lb`, `z0_ub` in the code).

    This amounts to finding the worst-case performance by solving the verification problem (VP) of the following form:

    \begin{array}{ll}
        \text{maximize} & \|r^K\|_\infty \\
        \text{subject to} & \{z^k\}_{k=0}^K \text{ given by algorithm} \\
        & z^0 \in \mathcal{Z} \\
        & x \in \mathcal{X},
    \end{array}

    where $\mathcal{X} := [x_\mathrm{lb},\, x_\mathrm{ub}]$ (`x_lb`, `x_ub` in the code) is the space of problem parameters.
    As in [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330), we measure the residual in the infinity norm, which keeps the VP a mixed-integer *linear* program.
    """)
    return


@app.cell
def _():
    z0_lb, z0_ub = 4.0, 6.0    # initial condition
    return z0_lb, z0_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: Projected gradient descent

    We solve this problem using projected gradient descent with step size $\eta > 0$, whose iterations are given by

    \begin{array}{ll}
        z^{k+1} = \Pi_{\mathbf{R}^n_+}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}

    where $\Pi_{\mathbf{R}^n_+}(v) = \max\{v, 0\}$, taken element-wise, is the projection onto the nonnegative orthant (the `relu` step).
    We choose $\eta = 1/(2L)$, where $L$ is the maximum eigenvalue of $A^T A$.

    We also consider a momentum variant of projected gradient descent with Nesterov-type extrapolation:

    \begin{array}{rcl}
        w^{k+1} &=& \Pi_{\mathbf{R}^n_+}\left(z^{k} - \eta A^T(Az^{k} - x)\right), \\
        z^{k+1} &=& w^{k+1} + \beta_{k+1} (w^{k+1} - w^{k}),
    \end{array}

    with $w^0 = z^0$ and $\beta_k = (k-1)/(k+2)$, so that $\beta_1 = 0$.

    The shared implementation used throughout this tutorial is:

    ```python
    def pgd_step(vp, A, eta, z_prev, w_prev, x_param, k, momentum=False):
        w_next = vp.relu(
            z_prev - eta * A.T @ (A @ z_prev - x_param)
        )
        if momentum:
            beta_k = (k - 1) / (k + 2)
            z_next = w_next + beta_k * (w_next - w_prev)
        else:
            z_next = w_next
        residual = z_next - z_prev
        return z_next, w_next, residual
    ```

    Thus, `momentum=False` gives ordinary projected gradient descent, while
    `momentum=True` applies the momentum extrapolation before the next iteration.
    """)
    return


@app.cell
def _(A, np):
    L = np.max(np.linalg.eigvalsh(A.T @ A))  # symmetric routine → real eigenvalues
    eta = 0.5 / L
    return L, eta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every verification mode below (simulator and MIP-VP) drives this
    *same* `pgd_step` recurrence and residual; only the `Verifier` backend and
    whether $(x, z^0)$ are sampled instances or bounded search variables change.
    Setting `momentum=True` selects the momentum variant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    We now describe how we can use our package `fomtools` to verify the worst-case convergence rate of projected gradient descent and its accelerated variant, applied on nonnegative least-squares problem. We compare the simulated results obtained from randomly generated problem instances to the obtained worst-case bound from PEPit and our approach, up to $K$ iterations.
    """)
    return


@app.cell
def _():
    K = 25  # Maximum iteration number
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    The `Verifier` class offers functionality of running simulations of the algorithm given randomly generated problem instances, with the option `method='simulator'`.
    This randomly-generated problem instance consists of two parameters: randomly generated problem parameter $x \in \mathcal{X}$ and initial iterate $z^0 \in \mathcal{Z}$. The samples are drawn once below and reused by every simulation cell so that all comparisons share the same instances.
    """)
    return


@app.cell
def _(m, n, np, x_lb, x_ub, z0_lb, z0_ub):
    n_samples = 10000  # the number of samples to simulate

    # Generate the sample of problem parameter x and initial iterate z0
    _rng = np.random.default_rng(0)
    x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, m))
    z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n))
    return x_samples, z0_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Meanwhile, the simulation will be run through following function.
    """)
    return


@app.cell
def _(A, eta, m, n, np, pgd_step):
    from fomtools.verifier import Verifier

    def pgd_simulation(x_samples, z0_samples, K, momentum=False, ord=np.inf):
        """Simulate projected gradient descent``.

        Builds a simulator-mode :class:`Verifier`, registers the parameter ``x`` and
        initial iterate ``z0`` as sampled parameters, rolls out the recurrence with
        ``pgd_step``, and simulates via ``vp.solve()``. Each ``*_samples`` array
        has shape ``(n_samples, dim)`` (a single-row array replays one instance),
        and the returned ``residuals`` has shape ``(n_samples, K)``. Summary statistics
        (``mean``, ``min``, ``max``, ...) are available on the returned instance
        ``vp`` via ``vp[...]``.

        ``ord`` defaults to ``inf``, matching :func:`pgd_verification`. Note that
        ``set_objective_norm(..., 2)`` would maximize the *squared* Euclidean norm,
        so with ``ord=2`` the returned residuals are in squared units.
        """
        vp = Verifier(method='simulator', obbt_solver=None)
        x_param = vp.parameter(ndim=m, samples=x_samples)
        z0 = vp.parameter(ndim=n, samples=z0_samples)

        z, w = z0, z0
        for k in range(1, K + 1):
            z, w, residual = pgd_step(
                vp, A, eta, z, w, x_param, k=k, momentum=momentum,
            )
            vp.set_objective_norm(residual, ord)

        residuals = vp.solve()
        return vp, residuals

    return Verifier, pgd_simulation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given the parameter sample `x_samples` and the initial iterate sample `z0_samples`, simulator mode runs the algorithm on each problem instance `x_samples[i]` starting from `z0_samples[i]`.
    Running the solve method `_VP.solve()` outputs the history of residuals at objective (fixed-point residual). After calling `solve()`, summary statistics are also available directly on `_VP`, including `mean`, `std`, `min`, `max`, `median`, `empirical_risk`, `risk_bound`, `n_samples`, `n_iterations`, `runtime`, `time_per_sample`, and `throughput`.
    """)
    return


@app.cell
def _(K, np, pgd_simulation, x_samples, z0_samples):
    _VP, _residuals = pgd_simulation(x_samples, z0_samples, K, momentum=False, ord=np.inf)

    print(
        f"Summary statistics (at K={K}):\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
    )

    # Take average of residuals for iterate K=1 to K=K
    simulation_residuals = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return simulation_residual_quantiles, simulation_residuals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the simulation of momentum-accelerated projected gradient descent as well, reusing the same `pgd_simulation` helper and samples with `momentum=True`.
    """)
    return


@app.cell
def _(K, np, pgd_simulation, x_samples, z0_samples):
    _, _residuals = pgd_simulation(x_samples, z0_samples, K, momentum=True, ord=np.inf)
    simulation_residuals_momentum = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_momentum = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return (
        simulation_residual_quantiles_momentum,
        simulation_residuals_momentum,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm verification through verification problem (VP)

    We now obtain worst-case guarantee by solving the corresponding verification problem (VP).
    Passing the option `method='mip'` formulates the VP as a mixed-integer linear program (MILP), based on [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming.](https://arxiv.org/abs/2412.11330)


    The following function rolls out the same `pgd_step` recurrence over the
    horizon $K$ and maximizes $\|r^k\|_\infty$ at each iteration.
    """)
    return


@app.cell
def _(A, Verifier, eta, m, n, np, pgd_step, x_lb, x_ub, z0_lb, z0_ub):
    def pgd_verification(K, method, momentum=False, ord=np.inf, **verifier_kwargs):
        """Solve the projected-gradient verification problem for one backend.

        Builds a ``Verifier`` in ``method`` mode, registers the parameter box
        ``[x_lb, x_ub]`` and the initial iterate box ``[z0_lb, z0_ub]``,
        rolls out ``pgd_step`` over the horizon ``K``, and returns ``(vp, residuals)``
        with one worst-case residual per iteration. Extra ``verifier_kwargs``
        (``solver``, ``params``, ...) are forwarded to the ``Verifier`` constructor.

        ``ord`` defaults to ``inf``, which keeps the MIP a MILP. A 2-norm
        objective is squared, so its value is square-rooted on return.
        """
        vp = Verifier(method=method, **verifier_kwargs)
        x_param = vp.parameter(m, lb=x_lb, ub=x_ub)
        z = vp.iterate_trajectory(K+1, collapse_expr='auto')
        z[0] = vp.iterate(n, lb=z0_lb, ub=z0_ub)
        w = z[0]

        residuals = []
        for k in range(1, K + 1):
            z, w, residual = pgd_step(
                vp, A, eta, z, w, x_param, k=k, momentum=momentum,
            )
            vp.set_objective_norm(residual, ord)
            val = vp.solve()
            residuals.append(np.sqrt(val) if ord == 2 else val)
        return vp, residuals

    return (pgd_verification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MIP-VP

    The following example formulates the verification problem as a mixed-integer linear program (MILP) and solves it to obtain exact worst-case fixed-point residual after $K$ iterations, for both projected gradient descent and its accelerated variant.
    """)
    return


@app.cell
def _(K, np, pgd_verification, solver_params):
    _, all_residuals = pgd_verification(
        K, 'mip', momentum=False, ord=np.inf, solver='gurobi', params=solver_params,
    )
    _, all_residuals_momentum = pgd_verification(
        K, 'mip', momentum=True, ord=np.inf, solver='gurobi', params=solver_params,
    )
    return all_residuals, all_residuals_momentum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    As we compare our results to the worst-case performance of projected gradient descent, we use performance estimation problem (PEP) to obtain such worst-case rate using the shared `PEPitRunner`.

    #### Certified initial-condition radius

    PEP bounds the worst case over every $L$-smooth convex $f$ and convex indicators $g$ whose initial iterate lies in a Euclidean ball of radius $R$ about the fixed point: $\|z^0 - z^\star\|_2 \le R$.
    For this PEP bound to be a valid upper bound on our VP with $z^0 \in \mathcal{Z}$, $x \in \mathcal{X}$, the ball must cover the box:

    $$R \;\ge\; \sup_{z^0 \in \mathcal{Z},\, x \in \mathcal{X}} \|z^0 - z^\star(x)\|_2 .$$

    We find $R$ via solving optimization problem which gives maximum value of $\|z^0 - z^\star(x)\|_2$ for all $z^0 \in \mathcal{Z}$ and $x \in \mathcal{X}$.
    Note that the NNLS solution map $x \mapsto z^\star(x)$ is piecewise linear; on each active set $\mathcal{I}$, its Jacobian is $(A_{\mathcal{I}}^T A_{\mathcal{I}})^{-1} A_{\mathcal{I}}^T$, so $\sigma_{\min}(A_{\mathcal{I}}) \ge \sigma_{\min}(A)$.
    This mapping is globally $\gamma$-Lipschitz with $\gamma = 1/\sigma_{\min}(A)$.
    Splitting the two box contributions with the triangle inequality gives the certified radius

    $$R = \max_{z^0 \in \mathcal{Z}} \|z^0 - z^\star(\bar{x})\|_2
        \;+\; \gamma \cdot \mathbf{rad}(\mathcal{X}),$$

    where $\bar{x}$ is the center of the $x$-box and $\mathbf{rad}(\mathcal{X}) = \tfrac12\sqrt{\sum_j (x_\mathrm{ub}-x_\mathrm{lb})_j^2}$ is its circumradius.
    The first term is the distance from $z^\star(\bar{x})$ to the farthest box vertex, evaluated coordinatewise.
    This requires a single NNLS solve (at $\bar{x}$) and no sampling.
    """)
    return


@app.cell(hide_code=True)
def _(A, K, L, certify_nnls_pepit_radius, eta, np, x_lb, x_ub, z0_lb, z0_ub):
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

    _pepit_radius = certify_nnls_pepit_radius(
        A,
        x_lb,
        x_ub,
        z0_lb,
        z0_ub,
    )
    # PEPitRunner defaults to Clarabel: SCS fails on the k=1 PEP here and
    # returns None, which used to show up as a missing K=1 point in the figure.
    _runner = PEPitRunner()

    # PEPit's performance metric is ||x[k] - x[k-1]||^2, so its square root
    # bounds ||r^k||_2, and hence ||r^k||_inf, the norm the MIP maximizes.
    _pep, _ = _runner.proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius, nonsmooth="indicator"
    )
    _pep_momentum, _ = _runner.accelerated_proximal_gradient_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius, nonsmooth="indicator"
    )
    all_residuals_PEP = np.sqrt(np.asarray(_pep, dtype=float))
    all_residuals_PEP_momentum = np.sqrt(np.asarray(_pep_momentum, dtype=float))
    return all_residuals_PEP, all_residuals_PEP_momentum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numerical result: comparison against baselines

    Now we plot the verification results against the PEPit bound and the
    in-sample simulation statistics, including the sample
    worst-case instance. The simulation and MIP curves report
    $\|r^k\|_\infty$. PEPit bounds $\|r^k\|_2$, which is also a bound on
    $\|r^k\|_\infty$ because $\|v\|_\infty \le \|v\|_2$. The curves answer
    different questions, so they are not interchangeable:

    - **sample worst-case** — a *lower* bound on the true worst case. The
      samples are uniform over the same boxes the verifiers search, but uniform
      sampling in $8$ dimensions concentrates away from the vertices and faces
      where the maximizer sits, so this curve is loose by construction.
    - **MIP-VP** — the *exact* worst case over the boxes, for this problem
      instance.
    - **PEPit** — the worst case of $\|r^k\|_2$ over *every* $L$-smooth convex
      $f$ and convex indicator $g$ with $\|z^0-z^\star\|_2 \le R$. It bounds a
      whole function class rather than this instance, and a larger norm, so it
      should sit above MIP-VP; here $R$
      comes from `certify_nnls_pepit_radius`, which covers the entire
      initial-condition box rather than just the drawn samples.

    So the expected ordering at every $K$ is
    $\text{sample worst-case} \le \text{MIP-VP} \le \text{PEPit}$.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    plt,
    simulation_residual_quantiles,
    simulation_residuals,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label=r'PEPit (bounds $\|r^k\|_2 \geq \|r^k\|_\infty$)')
    plt.plot(_ks, all_residuals, label='MIP-VP')
    _avg_line, = plt.plot(_ks, simulation_residuals, '--', label='sample average')
    plt.fill_between(_ks, simulation_residual_quantiles[2], simulation_residual_quantiles[0], alpha=.18, label='sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles[3], ':', color=_avg_line.get_color(), label='sample worst-case')
    plt.yscale('log')
    plt.ylabel(r'Worst-case fixed-point residual $\|r^k\|_\infty$')
    plt.xlabel(r'$K$')
    plt.suptitle('NNLS: projected gradient descent (PGD)')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the simulation of momentum-accelerated projected gradient descent as well, reusing the same `pgd_simulation` helper and samples with `momentum=True`.
    """)
    return


@app.cell
def _(
    K,
    all_residuals_PEP_momentum,
    all_residuals_momentum,
    plt,
    simulation_residual_quantiles_momentum,
    simulation_residuals_momentum,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_momentum, label=r'PEPit (bounds $\|r^k\|_2 \geq \|r^k\|_\infty$)')
    plt.plot(_ks, all_residuals_momentum, label='MIP-VP')
    _avg_line, = plt.plot(_ks, simulation_residuals_momentum, '--', label='sample average')
    plt.fill_between(_ks, simulation_residual_quantiles_momentum[2], simulation_residual_quantiles_momentum[0], alpha=.18, label='sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_momentum[3], ':', color=_avg_line.get_color(), label='sample worst-case')
    plt.yscale('log')
    plt.ylabel(r'Worst-case fixed-point residual $\|r^k\|_\infty$')
    plt.xlabel(r'$K$')
    plt.suptitle('NNLS: momentum PGD')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we overlay PGD and momentum PGD to compare the two methods
    directly. Methods are distinguished by color, and the data series (MIP-VP
    and sample average with its 10%-90% band) by line style. PEPit is
    left out here: it bounds a function class rather than this instance, so it
    is the one curve that does not distinguish the two methods *on this
    problem*. It is shown in the per-method figures above.
    """)
    return


@app.cell(hide_code=True)
def _(
    K,
    all_residuals,
    all_residuals_momentum,
    plt,
    simulation_residual_quantiles,
    simulation_residual_quantiles_momentum,
    simulation_residuals,
    simulation_residuals_momentum,
):
    from matplotlib.lines import Line2D

    plt.figure()
    _ks = range(1, K+1)

    _methods = [
        ('PGD', 'C0', all_residuals,
         simulation_residuals, simulation_residual_quantiles),
        ('momentum PGD', 'C1', all_residuals_momentum,
         simulation_residuals_momentum, simulation_residual_quantiles_momentum),
    ]
    for _, _color, _mip, _avg, _quant in _methods:
        plt.plot(_ks, _mip, '--', color=_color)
        plt.plot(_ks, _avg, ':', color=_color)
        plt.fill_between(_ks, _quant[2], _quant[0], color=_color, alpha=.18)

    _method_handles = [
        Line2D([], [], color=_color, label=_name)
        for _name, _color, *_ in _methods
    ]
    _style_handles = [
        Line2D([], [], color='gray', linestyle='--', label='MIP-VP'),
        Line2D([], [], color='gray', linestyle=':', label='sample average'),
    ]

    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel(r'Worst-case fixed-point residual $\|r^k\|_\infty$')
    _leg = plt.legend(handles=_method_handles, loc='upper right')
    plt.gca().add_artist(_leg)
    plt.legend(handles=_style_handles, loc='lower left')
    plt.suptitle('NNLS: PGD vs. momentum PGD')
    return


if __name__ == "__main__":
    app.run()
