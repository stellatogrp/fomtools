import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    from pathlib import Path

    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "fomtools").is_dir():
            repo_root = candidate
            break
    else:
        repo_root = start

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from docs.tutorials.helper_functions.nnls import make_nnls_matrix, pgd_step
    from scipy.optimize import nnls

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    solver_params = {
        'OutputFlag': 0,
        'LogToConsole': 0,
    }
    return make_nnls_matrix, mo, nnls, np, pgd_step, plt, solver_params


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
def _(make_nnls_matrix):
    m, n = 5, 3
    A = make_nnls_matrix(m=m, n=n)
    x_lb, x_ub = 10.0, 11.0    # parameter set
    return A, m, n, x_lb, x_ub


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
    return z0_lb, z0_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: Projected gradient descent

    We solve this problem using projected gradient descent with step size $\eta > 0$, whose iterations are given by

    \begin{array}{ll}
        z^{k+1} = {\bf ReLU}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}

    where $\bf ReLU(v) = \max\{v, 0\}$ element-wise.
    We choose $\eta = 1/(2L)$, where $L$ is the maximum eigenvalue of $A^T A$.

    We also consider a momentum variant of projected gradient descent, which follows the FISTA update:

    \begin{array}{rcl}
        w^{k+1} &=& {\bf ReLU}\left(z^{k} - \eta A^T(Az^{k} - x)\right), \\
        z^{k+1} &=& w^{k+1} + \beta_k (w^{k+1} - w^{k}),
    \end{array}

    with $w^0 = z^0$ and $\beta_k = (k-1)/(k+2)$.

    The shared implementation used throughout this tutorial is:

    ```python
    def pgd_step(VP, A, eta, z_prev, w_prev, x_param, k, momentum=False):
        w_next = VP.step.relu(
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
    `momentum=True` applies the FISTA extrapolation before the next iteration.
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
    Every verification mode below (simulator, MIP-VP, and SDP-VP) drives this
    *same* `pgd_step` recurrence and residual; only the `Verifier` backend and
    whether $(x, z^0)$ are sampled instances or bounded search variables change.
    Setting `momentum=True` selects the FISTA variant.
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
    K = 10  # Maximum iteration number
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
        ``pgd_step``, and simulates via ``VP.solve()``. Each ``*_samples`` array
        has shape ``(n_samples, dim)`` (a single-row array replays one instance),
        and the returned ``residuals`` has shape ``(n_samples, K)``. Summary statistics
        (``mean``, ``min``, ``max``, ...) are available on the returned instance
        ``VP`` via ``VP[...]``.
        """
        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=m, samples=x_samples)
        z0 = VP.add_param(ndim=n, samples=z0_samples)

        z, w = z0, z0
        for k in range(1, K + 1):
            z, w, residual = pgd_step(
                VP, A, eta, z, w, x_param, k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, ord)

        residuals = VP.solve()
        return VP, residuals

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
        f"Summary statistics:\n"
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
    There are two approaches to formulating VP: mixed-integer linear program (MILP) formulation and (relaxed) semidefinite program (SDP) formulation.
    Passing the option `method='mip'` and `method='sdp'` will formulate respective verification problem forms, which are based on the following papers:

    - MIP-VP: [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming.](https://arxiv.org/abs/2412.11330)
    - SDP-VP: [Verification of First-Order Methods for Parametric Quadratic Optimization.](https://link.springer.com/article/10.1007/s10107-025-02261-w)

    Both formulations use the following function, which rolls out the same
    `pgd_step` recurrence over the horizon $K$ and maximizes the residual at
    each iteration. The MIP backend uses the infinity-norm objective while the
    SDP backend uses the squared Euclidean objective.
    """)
    return


@app.cell
def _(A, Verifier, eta, m, n, np, pgd_step, x_lb, x_ub, z0_lb, z0_ub):
    def pgd_verification(K, method, momentum=False, ord=None, **verifier_kwargs):
        """Solve the projected-gradient verification problem for one backend.

        Builds a ``Verifier`` in ``method`` mode, registers the parameter box
        ``[x_lb, x_ub]`` and the initial iterate box ``[z0_lb, z0_ub]``,
        rolls out ``pgd_step`` over the horizon ``K``, and returns ``(VP, residuals)``
        with one worst-case residual per iteration. ``ord`` defaults to ``inf``
        for MIP and ``2`` for SDP. Extra ``verifier_kwargs`` (``solver``, ``params``, ...)
        are forwarded to the ``Verifier`` constructor.
        """
        ord = ord if ord is not None else (np.inf if method == 'mip' else 2)

        VP = Verifier(method=method, **verifier_kwargs)
        x_param = VP.add_param(m, lb=x_lb, ub=x_ub)
        z = VP.iterate_trajectory(K+1, collapse_expr='auto')
        z[0] = VP.add_iterate(n, lb=z0_lb, ub=z0_ub)
        w = z[0]

        residuals = []
        for k in range(1, K + 1):
            z, w, residual = pgd_step(
                VP, A, eta, z, w, x_param, k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, ord)
            residuals.append(VP.solve())
        return VP, residuals

    return (pgd_verification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MIP-VP

    The following example formulates the verification problem as a mixed-integer linear program (MILP) and solves it to obtain exact worst-case fixed-point residual after $K$ iterations, for both projected gradient descent and its accelerated variant.
    """)
    return


@app.cell
def _(K, pgd_verification, solver_params):
    _, all_residuals = pgd_verification(
        K, 'mip', momentum=False, solver='gurobi', params=solver_params,
    )
    _, all_residuals_momentum = pgd_verification(
        K, 'mip', momentum=True, solver='gurobi', params=solver_params,
    )
    return all_residuals, all_residuals_momentum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SDP-VP

    The following example formulates the verification problem as a semidefinite program (SDP) and solves it to obtain valid upper bound to the worst-case fixed-point residual after $K$ iterations. Although the constraints, objective function, and iterate updates are quadratic, we use semidefinite lifting to relax the (nonconvex) quadratic constraints into convex linear matrix inequality.
    """)
    return


@app.cell
def _(K, pgd_verification):
    _, all_residuals_sdp = pgd_verification(
        K, 'sdp', momentum=False, solver='clarabel'
    )
    _, all_residuals_sdp_momentum = pgd_verification(
        K, 'sdp', momentum=True, solver='clarabel',
    )
    return all_residuals_sdp, all_residuals_sdp_momentum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    As we compare our results to the worst-case performance of projected gradient descent, we use performance estimation problem (PEP) to obtain such worst-case rate using the shared `PEPitRunner`.

    #### Certified initial-condition radius

    PEP bounds the worst case over every $L$-smooth convex $f$ and convex indicators $g$ whose initial iterate lies in a Euclidean ball of radius $R$ about the fixed point: $\|z^0 - z^\star\|_2 \le R$.
    For this PEP bound to be a valid upper bound on our VP with $z^0 \in \mathcal{Z}$, $x \in \mathcal{X}$, the ball must cover the box:

    $$R \;\ge\; \sup_{z^0 \in \mathcal{Z},\, x \in \mathcal{X}} \|z^0 - z^\star(x)\|_2 .$$

    We find $R$ via solving optimization problem which gives maximum value of $\|z^0 - z^\star(x)\|$ for all $z^0 \in \mathcal{Z}$ and $x \in \mathcal{X}$.
    Note that the NNLS solution map $x \mapsto z^\star(x)$ is piecewise linear; on each active set $\mathcal{I}$, its Jacobian is $(A_{\mathcal{I}}^\top A_{\mathcal{I}})^{-1} A_{\mathcal{I}}^\top$, so $\sigma_{\min}(A_{\mathcal{I}}) \ge \sigma_{\min}(A)$.
    This mapping is globally $\gamma$-Lipschitz with $\gamma = 1/\sigma_{\min}(A)$.
    Splitting the two box contributions with the triangle inequality gives the certified radius

    $$R = \max_{z^0 \in \mathcal{Z}} \|z^0 - z^\star(\bar{x})\|_2
        \;+\; \gamma \cdot \operatorname{rad}(\mathcal{X}),$$

    where $\bar{x}$ is the center of the $x$-box and $\operatorname{rad}(\mathcal{X}) = \tfrac12\sqrt{\sum_j (\texttt{x\_ub}-\texttt{x\_lb})^2}$ is its circumradius.
    The first term is the distance from $z^\star(\bar{x})$ to the farthest box vertex, evaluated coordinatewise.
    This requires a single NNLS solve (at $\bar{x}$) and no sampling.
    """)
    return


@app.cell
def _(A, m, nnls, np, x_lb, x_ub, z0_lb, z0_ub):
    # Certified upper bound on sup_{z0 in Z, x in X} ||z0 - z*(x)||_2 (see above).
    _sigma_min = np.sqrt(np.min(np.linalg.eigvalsh(A.T @ A)))
    _x_center = np.full(m, 0.5 * (x_lb + x_ub))
    _z_star_center = nnls(A, _x_center)[0]
    # Farthest z0 in [z0_lb, z0_ub]^n from z*(xbar): pick the extreme per axis.
    _z0_far = np.maximum(
        np.abs(z0_ub - _z_star_center), np.abs(z0_lb - _z_star_center)
    )
    _x_circumradius = 0.5 * np.linalg.norm(np.full(m, x_ub - x_lb))
    pepit_radius = np.linalg.norm(_z0_far) + _x_circumradius / _sigma_min
    return (pepit_radius,)


@app.cell
def _(K, L, eta, pepit_radius):
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

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
    ### Numerical result: comparison against baselines

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
    simulation_residual_quantiles,
    simulation_residuals,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='PEPit')
    plt.plot(_ks, all_residuals, label='MIP-VP')
    plt.plot(_ks, all_residuals_sdp, label='SDP-VP')
    _avg_line, = plt.plot(_ks, simulation_residuals, '--', label='sample average')
    plt.fill_between(_ks, simulation_residual_quantiles[2], simulation_residual_quantiles[0], alpha=.18, label='sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles[3], ':', color=_avg_line.get_color(), label='sample worst-case')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
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
    all_residuals_sdp_momentum,
    plt,
    simulation_residual_quantiles_momentum,
    simulation_residuals_momentum,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_momentum, label='PEPit')
    plt.plot(_ks, all_residuals_momentum, label='MIP-VP')
    plt.plot(_ks, all_residuals_sdp_momentum, label='SDP-VP')
    _avg_line, = plt.plot(_ks, simulation_residuals_momentum, '--', label='sample average')
    plt.fill_between(_ks, simulation_residual_quantiles_momentum[2], simulation_residual_quantiles_momentum[0], alpha=.18, label='sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_momentum[3], ':', color=_avg_line.get_color(), label='sample worst-case')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.suptitle('NNLS: momentum PGD')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we overlay PGD and momentum PGD to compare the two methods
    directly. Methods are distinguished by color, and the data series (PEPit,
    MIP-VP, SDP-VP, and sample average with its 10%-90% band) by line style.
    """)
    return


@app.cell(hide_code=True)
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    all_residuals_PEP_momentum,
    all_residuals_momentum,
    all_residuals_sdp,
    all_residuals_sdp_momentum,
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
        ('PGD', 'C0', all_residuals_PEP, all_residuals, all_residuals_sdp,
         simulation_residuals, simulation_residual_quantiles),
        ('momentum PGD', 'C1', all_residuals_PEP_momentum, all_residuals_momentum,
         all_residuals_sdp_momentum, simulation_residuals_momentum,
         simulation_residual_quantiles_momentum),
    ]
    for _, _color, _pep, _mip, _sdp, _avg, _quant in _methods:
        # plt.plot(_ks, _pep, '-', color=_color)
        plt.plot(_ks, _mip, '--', color=_color)
        plt.plot(_ks, _sdp, '-.', color=_color)
        plt.plot(_ks, _avg, ':', color=_color)
        plt.fill_between(_ks, _quant[2], _quant[0], color=_color, alpha=.18)

    _method_handles = [
        Line2D([], [], color=_color, label=_name)
        for _name, _color, *_ in _methods
    ]
    _style_handles = [
        # Line2D([], [], color='gray', linestyle='-', label='PEPit'),
        Line2D([], [], color='gray', linestyle='--', label='MIP-VP'),
        Line2D([], [], color='gray', linestyle='-.', label='SDP-VP'),
        Line2D([], [], color='gray', linestyle=':', label='sample average'),
    ]

    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case fixed-point residual')
    _leg = plt.legend(handles=_method_handles, loc='upper right')
    plt.gca().add_artist(_leg)
    plt.legend(handles=_style_handles, loc='lower left')
    plt.suptitle('NNLS: PGD vs. momentum PGD')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backend-aware collapsing with `iterate_trajectory`

    Writing an affine update such as `z[k] = M @ z[k-1] + c` produces a
    *collapsed* linear expression: it references existing variables and mints no
    new one. This is ideal for the MIP and simulator backends. For the **SDP**
    backend, however, collapsing couples many variables in a single dense moment
    matrix; keeping each iterate as its own leaf (linked by an equality) yields a
    much sparser matrix that decomposes into many small PSD cones and is often
    substantially faster to solve.

    `verifier.iterate_trajectory(K + 1, collapse_expr="auto")` picks the right
    behavior automatically: collapse on MIP/simulator, keep a separate leaf on
    SDP. Pass `collapse_expr=True`/`False` (on the trajectory or per slot via
    `z.assign(k, expr, collapse_expr=...)`) to force it. Note the policy only
    applies to *collapsed* expressions — a value that is already a fresh variable
    (a `relu`/`satlin`/`soft_threshold` output on MIP, or a `relu` on SDP) is
    stored as-is regardless.

    The cell below illustrates the SDP effect on a soft-threshold (ISTA) chain,
    where the collapsed and separated formulations are genuinely different SDP
    relaxations.
    """)
    return


@app.cell
def _(np):
    import time

    from fomtools.verifier import Verifier as _V

    def _sdp_ista_bound(collapse_expr, K=2, seed=0):
        m2, n2 = 4, 3
        t, lambd = 0.05, 0.1
        np.random.seed(seed)
        A2 = np.random.normal(size=(m2, n2))
        vp = _V(method="sdp", solver="clarabel", obbt_solver=None)
        xp = vp.add_param(m2, lb=-1, ub=1)
        zt = vp.iterate_trajectory(K + 1, collapse_expr=collapse_expr)
        zt[0] = vp.add_iterate(n2, lb=-0.5, ub=0.5)
        for kk in range(1, K + 1):
            zt[kk] = vp.step.soft_threshold(
                zt[kk - 1] - t * A2.T @ (A2 @ zt[kk - 1] - xp), lambd=lambd * t
            )
        vp.set_objective_norm(zt[K] - zt[K - 1], 2)
        t0 = time.perf_counter()
        bound = vp.solve()
        return bound, time.perf_counter() - t0

    try:
        b_collapse, s_collapse = _sdp_ista_bound(True, K=7)
        b_auto, s_auto = _sdp_ista_bound("auto")  # -> no-collapse on SDP
        print(f"SDP collapsed    : bound={b_collapse:.4f}  solve={s_collapse:.3f}s")
        print(f"SDP no-collapse  : bound={b_auto:.4f}  solve={s_auto:.3f}s")
        print("no-collapse is a sparser (>=) relaxation; timing favors it as K/n grow")
    except Exception as exc:  # missing SDP solver in this environment
        print(f"(skipping SDP comparison: {exc})")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
