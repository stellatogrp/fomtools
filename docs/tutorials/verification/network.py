import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import contextlib
    import io
    import sys
    from pathlib import Path

    import cvxpy as cp
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

    from docs.tutorials.helper_functions.network import (
        estimate_network_pepit_radius,
        make_network_data,
        pdhg_step,
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
        cp,
        estimate_network_pepit_radius,
        make_network_data,
        mo,
        np,
        pdhg_step,
        plt,
        solver_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem setup: Network flow optimization

    In this tutorial, we verify first-order methods for the minimum-cost network flow example from Section 4.2 of [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

    Suppose that we are given a set of $n_s$ supply nodes and $n_d$ demand nodes, which constructs a bipartite graph with capacitated directed edges from supply nodes to demand nodes.
    Let $A_s$ and $A_d$ be submatrices corresponding to supply nodes and demand nodes, respectively.
    For each edge $i$, let $c_i$ be the flow capacity, $\mu_i$ be the cost of sending a unit of flow, and $f_i$ be the variable representing the amount of flow sent.
    Finally, let $b_s \ge 0$ be the vector of supply amounts available at each supply node and $x \le 0$ be the vector of requested demand amount for each demand node.

    Consider the following minimum cost network flow problem:

    \begin{array}{ll}
        \text{minimize} & \mu^T f \\
        \text{subject to}
        & A_s f \le b_s \\
        & A_d f = x \\
        & 0 \le f \le c.
    \end{array}

    The demand vector $x$ is treated as the problem parameter. The verification problem will search over $x \in \mathcal{X}$ and over bounded initial primal-dual iterates.
    """)
    return


@app.cell(hide_code=True)
def _(make_network_data, np):
    # Generate bipartite graph with n_s supply nodes and n_d demand nodes,
    # whose edges are added with probability p = 0.5.
    n_s = 6
    n_d = 3
    p = 0.5

    A_s, A_d, b_s, c, mu = make_network_data(
        n_supply=n_s,
        n_demand=n_d,
        edge_probability=p,
        graph_seed=1,
        data_seed=2,
    )
    n_e = A_s.shape[1]

    d_lb, d_ub = -7., -6.  # demand
    c_lb, c_ub =  5.,  5.  # capacity

    # Parameter set X = [x_lb, x_ub] of demands
    x_lb = d_lb * np.ones(n_d)
    x_ub = d_ub * np.ones(n_d)
    z0_lb, z0_ub = np.zeros(n_e), c.copy()
    v0_lb, v0_ub = np.zeros(n_s), np.ones(n_s)
    w0_lb, w0_ub = -np.ones(n_d), np.ones(n_d)
    return (
        A_d,
        A_s,
        b_s,
        c,
        c_lb,
        c_ub,
        mu,
        n_d,
        n_e,
        n_s,
        v0_lb,
        v0_ub,
        w0_lb,
        w0_ub,
        x_lb,
        x_ub,
        z0_lb,
        z0_ub,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We make sure that the optimization problem with randomly generated data has a solution.
    """)
    return


@app.cell
def _(A_d, A_s, b_s, c, cp, mu, n_e, x_ub):
    # Check the feasibility of the generated min cost flow problem
    f = cp.Variable(n_e)

    constraints_check = [
        A_s @ f <= b_s,
        A_d @ f == x_ub,
        0 <= f, f <= c,
    ]

    problem_check = cp.Problem(objective=cp.Minimize(mu.T@f), constraints=constraints_check)
    problem_check.solve(solver=cp.SCS)

    print(f"The problem is {problem_check.status}.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: PDHG

    We solve this problem using the standard Chambolle-Pock method, or Primal-dual Hybrid Gradient (PDHG):

    \begin{array}{rcl}
        z^{k+1} &=& \Pi_{[0, c]} \big( z^k - \eta (\mu + A_s^T v^k - A_d^T w^k) \big) \\
        v^{k+1} &=& \mathbf{prox}_{\mathbb{I}_{\mathbb{R}_+}} \big( v^k + \eta \big( - b_s + A_s (2 z^{k+1} - z^k) \big) \big) \\
        w^{k+1} &=& w^k + \eta \big( x - A_d (2 z^{k+1} - z^k) \big).
    \end{array}

    Here $\Pi_{[0, c]}$ is projection onto the flow-capacity box and $\Pi_{\mathbb{R}_+}$ is pojection onto the nonnegative orthant.
    We choose $\eta = 0.5 / \| \begin{bmatrix} - A_s^T & A_d^T \end{bmatrix} \|_2$.

    The performance metric is the infinity norm of the scaled KKT fixed-point residual induced by the PDHG update.
    """)
    return


@app.cell
def _(A_d, A_s, np):
    # Step size choice
    L = np.linalg.norm(np.hstack([- A_s.T, A_d.T]), 2)
    eta = 0.5 / L
    return L, eta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every verification mode below (simulator, MIP-VP, SDP-VP, and the worst-case replays) drives this *same* recurrence and residual. In the current experiment, capacity, cost, and supply have zero-width bounds, so MIP and SDP use the same nominal problem data. The MIP represents those values as fixed symbolic parameters, whereas the SDP passes them as numeric constants to avoid an unnecessarily larger lifted matrix.
    Each cell rolls out the recurrence directly by iterating `pdhg_step` from the initial iterates $(z^0, v^0, w^0)$ over the horizon $K$, registering the scaled KKT residual at each step as an objective along the way.
    Setting `momentum=True` applies the extrapolation $\tilde z^k = z^k + \frac{k-1}{k+2}(z^k - z^{k-1})$
    (see below) before the dual updates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    We verify the performance of PDHG and its momentum variant up to a fixed iteration budget $K>0$. As in the NNLS tutorial, we compare randomly sampled simulations, exact MIP verification, SDP-relaxed verification, a PEPit bound, and the trajectories obtained by replaying the MIP worst-case instances.
    """)
    return


@app.cell
def _():
    K = 4  # Maximum iteration number
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    We first simulate the PDHG algorithm on given problem instances by randomly sampling the parameter $x$ (demand vector) over the parameter set $\mathcal{X}$ and the initial iterates $(z^0, v^0, w^0)$ over their prescribed boxes. The samples are drawn once below and reused by every simulation cell so that all comparisons share the same instances.
    """)
    return


@app.cell
def _(n_d, n_s, np, v0_lb, v0_ub, w0_lb, w0_ub, x_lb, x_ub, z0_lb, z0_ub):
    _n_samples = 10000
    _rng = np.random.default_rng(0)
    x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(_n_samples, n_d))
    z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(_n_samples, len(z0_lb)))
    v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(_n_samples, n_s))
    w0_samples = _rng.uniform(low=w0_lb, high=w0_ub, size=(_n_samples, n_d))
    return v0_samples, w0_samples, x_samples, z0_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Meanwhile, the simulation will be run through following function.
    """)
    return


@app.cell
def _(A_d, A_s, b_s, c, eta, mu, n_d, n_e, n_s, np, pdhg_step):
    from fomtools.verifier import Verifier

    def pdhg_simulation(
        x_samples, z0_samples, v0_samples, w0_samples, K,
        momentum=False, ord=np.inf,
    ):
        """Simulate PDHG on sampled instances.

        Builds a simulator-mode :class:`Verifier`, registers the demand
        parameter and the initial iterates ``(z0, v0, w0)`` as sampled
        parameters, rolls out the PDHG recurrence with ``pdhg_step``,
        and simulates via ``VP.solve()``. Each of the four ``*_samples`` arrays has shape
        ``(n_samples, dim)`` (a single-row array replays one instance), and the
        returned ``residuals`` has shape ``(n_samples, K)``. Summary statistics
        (``mean``, ``min``, ``max``, ``final_residuals``, ...) are available on
        the returned instance ``VP`` via ``VP[...]``.
        """
        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=n_d, samples=x_samples)
        z0 = VP.add_param(ndim=n_e, samples=z0_samples)
        v0 = VP.add_param(ndim=n_s, samples=v0_samples)
        w0 = VP.add_param(ndim=n_d, samples=w0_samples)

        z, v, w = z0, v0, w0
        for k in range(1, K + 1):
            z, v, w, residual = pdhg_step(
                VP, A_s, A_d, eta, z, v, w, x_param, c, mu, b_s,
                k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, ord)

        residuals = VP.solve()
        return VP, residuals

    return Verifier, pdhg_simulation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running `_VP.solve()` returns one residual history per sampled instance, which we summarize by averages and empirical quantiles. After calling `solve()`, summary statistics are also available directly on `_VP`, including `mean`, `std`, `min`, `max`, `median`, `empirical_risk`, `risk_bound`, `n_samples`, `n_iterations`, `runtime`, `time_per_sample`, and `throughput`.
    """)
    return


@app.cell
def _(K, np, pdhg_simulation, v0_samples, w0_samples, x_samples, z0_samples):
    _VP, _residuals = pdhg_simulation(
        x_samples, z0_samples, v0_samples, w0_samples, K,
        momentum=False, ord=np.inf,
    )

    print(
        f"Summary statistics:\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
    )

    simulation_residuals_pdhg = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_pdhg = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return simulation_residual_quantiles_pdhg, simulation_residuals_pdhg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Momentum PDHG simulation

    We also consider a momentum variant of PDHG. After the primal update,
    the method extrapolates the flow iterate as

    $$
        \tilde z^{k} = z^{k} + \frac{k-1}{k+2}(z^{k} - z^{k-1}),
    $$

    and uses $\tilde z^k$ in the dual updates. This reuses the same
    `pdhg_simulation` helper and the samples drawn above, with `momentum=True`.
    """)
    return


@app.cell
def _(K, np, pdhg_simulation, v0_samples, w0_samples, x_samples, z0_samples):
    _, _residuals = pdhg_simulation(
        x_samples, z0_samples, v0_samples, w0_samples, K,
        momentum=True, ord=np.inf,
    )
    simulation_residuals_pdhg_mom = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_pdhg_mom = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return (
        simulation_residual_quantiles_pdhg_mom,
        simulation_residuals_pdhg_mom,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We plot the in-sample mean and 10%-90% empirical quantile band of the residual against the horizon $K$, together with the sample worst-case instance, for PDHG and its momentum variant together.
    """)
    return


@app.cell(hide_code=True)
def _(
    K,
    plt,
    simulation_residual_quantiles_pdhg,
    simulation_residual_quantiles_pdhg_mom,
    simulation_residuals_pdhg,
    simulation_residuals_pdhg_mom,
):
    plt.figure()
    _ks = range(1, K + 1)

    plt.plot(_ks, simulation_residuals_pdhg, label='PDHG (in-sample avg)')
    plt.fill_between(
        _ks, simulation_residual_quantiles_pdhg[2], simulation_residual_quantiles_pdhg[0],
        alpha=.18, label='PDHG sample 10%-90% band',
    )
    plt.plot(
        _ks, simulation_residual_quantiles_pdhg[3], ':', color='C0', linewidth=.7,
        label='PDHG sample worst-case',
    )

    plt.plot(_ks, simulation_residuals_pdhg_mom, label='momentum PDHG (in-sample avg)')
    plt.fill_between(
        _ks, simulation_residual_quantiles_pdhg_mom[2], simulation_residual_quantiles_pdhg_mom[0],
        alpha=.18, label='momentum PDHG sample 10%-90% band',
    )
    plt.plot(
        _ks, simulation_residual_quantiles_pdhg_mom[3], ':', color='C1', linewidth=.7,
        label='momentum PDHG sample worst-case',
    )

    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('In-sample KKT residual')
    plt.legend()
    plt.suptitle('PDHG vs. momentum PDHG in-sample residual statistics')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm verification through verification problem (VP)

    ### MIP-VP
    We now obtain worst-case guarantees by solving the corresponding verification problem with a MIP formulation. The demand vector and initial iterates vary over their prescribed bounds. Capacities, costs, and supplies are represented as parameters too, but their lower and upper bounds coincide in this experiment, so they remain fixed at the same nominal values used by the SDP.
    """)
    return


@app.cell
def _(
    A_d,
    A_s,
    Verifier,
    b_s,
    c,
    c_lb,
    c_ub,
    eta,
    mu,
    n_d,
    n_e,
    n_s,
    np,
    pdhg_step,
    v0_lb,
    v0_ub,
    w0_lb,
    w0_ub,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    def pdhg_verification(
        K, method, momentum=False, ord=None, parametrize_data=None, **verifier_kwargs,
    ):
        """Solve the PDHG verification problem for one backend.

        Builds a ``Verifier`` in ``method`` mode, registers the demand parameter
        and the ``(z0, v0, w0)`` iterate boxes, rolls out ``pdhg_step`` over the
        horizon ``K``, and returns ``(VP, residuals)`` with one worst-case
        residual per iteration.

        By default, the MIP backend represents capacity, cost, and supply as
        symbolic parameters while the SDP backend uses numeric constants. In
        this tutorial those parameter bounds have zero width, so both choices
        describe the same nominal problem; the numeric SDP representation
        merely avoids redundant lifted variables. Pass ``parametrize_data``
        to override this backend-derived default. ``ord`` selects the residual
        norm (defaults to ``inf`` for MIP and ``2`` for SDP); a 2-norm
        objective is squared, so its value is square-rooted on return. Extra
        ``verifier_kwargs`` (``solver``, ``obbt_solver``, ``params``, ...) are
        forwarded to the ``Verifier`` constructor.
        """
        VP = Verifier(method=method, **verifier_kwargs)
        x_param = VP.add_param(n_d, lb=x_lb, ub=x_ub)

        if parametrize_data is None:
            parametrize_data = method == 'mip'
        if parametrize_data:
            capacity = VP.add_param(n_e, lb=c_lb * np.ones(n_e), ub=c_ub * np.ones(n_e))
            supply = VP.add_param(n_s, lb=b_s, ub=b_s)
            cost = VP.add_param(n_e, lb=mu, ub=mu)
        else:
            capacity, supply, cost = c, b_s, mu

        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        v = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        w = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n_e, lb=z0_lb, ub=z0_ub)
        v[0] = VP.add_iterate(n_s, lb=v0_lb, ub=v0_ub)
        w[0] = VP.add_iterate(n_d, lb=w0_lb, ub=w0_ub)

        ord = ord if ord is not None else (np.inf if method == 'mip' else 2)

        residuals = []
        for k in range(1, K + 1):
            z, v, w, residual = pdhg_step(
                VP, A_s, A_d, eta, z, v, w, x_param, capacity, cost, supply,
                k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, ord)
            val = VP.solve()
            residuals.append(np.sqrt(val) if ord == 2 else val)
        return VP, residuals

    return (pdhg_verification,)


@app.cell
def _(K, pdhg_verification, solver_params):
    _, all_residuals = pdhg_verification(
        K, 'mip', solver='gurobi', params=solver_params,
    )
    print(all_residuals)
    return (all_residuals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same MIP verification procedure is then applied to the momentum PDHG recurrence, using the residual expression induced by the extrapolated primal iterate.
    """)
    return


@app.cell
def _(K, pdhg_verification, solver_params):
    # Solve verification problem for momentum PDHG
    _, all_residuals_momentum = pdhg_verification(
        K, 'mip', momentum=True, solver='gurobi', params=solver_params,
    )
    print(all_residuals_momentum)
    return (all_residuals_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SDP-VP

    We also solve an SDP relaxation of each verification problem using the squared Euclidean KKT residual, which is substantially faster than an SDP infinity-norm objective. We report the resulting $\|r^k\|_2$ directly, without normalizing by dimension.
    """)
    return


@app.cell
def _(K, pdhg_verification):
    _, all_residuals_sdp = pdhg_verification(
        K, 'sdp', solver='clarabel', obbt_solver='highs',
    )
    return (all_residuals_sdp,)


@app.cell
def _(K, pdhg_verification):
    _, all_residuals_sdp_momentum = pdhg_verification(
        K, 'sdp', momentum=True, solver='clarabel', obbt_solver='highs',
    )
    return (all_residuals_sdp_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we plot the verification results against the SDP comparison,
    a worst-case rate from PEPit's performance estimation problem, and the
    horizon-specific worst-case instances extracted from MIP.
    """)
    return


@app.cell(hide_code=True)
def _(
    A_d,
    A_s,
    K,
    L,
    all_residuals,
    all_residuals_sdp,
    b_s,
    c,
    estimate_network_pepit_radius,
    eta,
    mu,
    plt,
    simulation_residual_quantiles_pdhg,
    simulation_residuals_pdhg,
    v0_samples,
    w0_samples,
    x_samples,
    z0_samples,
):
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

    # Empirical primal-dual radius over the shared simulation samples.
    _pepit_radius = estimate_network_pepit_radius(
        A_s,
        A_d,
        b_s,
        mu,
        c,
        eta,
        z0_samples,
        v0_samples,
        w0_samples,
        x_samples,
    )
    _runner = PEPitRunner()
    all_residuals_PEP, _ = _runner.pdhg_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius
    )
    all_residuals_PEP_momentum, _ = _runner.pdhg_momentum_residuals(
        K=K, L=float(L), eta=float(eta), R=_pepit_radius
    )

    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='PEPit')
    plt.plot(_ks, all_residuals, label='MIP-VP')
    plt.plot(_ks, all_residuals_sdp, label='SDP-VP')
    _avg_line, = plt.plot(_ks, simulation_residuals_pdhg, '--', label='sample average')
    plt.fill_between(_ks, simulation_residual_quantiles_pdhg[2], simulation_residual_quantiles_pdhg[0], alpha=.18, label='sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_pdhg[3], ':', color=_avg_line.get_color(), label='sample worst-case')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case KKT residual')
    plt.legend()
    plt.suptitle('network optimization example')
    return (all_residuals_PEP_momentum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also plot the verification results for momentum PDHG.
    """)
    return


@app.cell(hide_code=True)
def _(
    K,
    all_residuals_PEP_momentum,
    all_residuals_momentum,
    all_residuals_sdp_momentum,
    plt,
    simulation_residual_quantiles_pdhg_mom,
    simulation_residuals_pdhg_mom,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP_momentum, label='PEPit')
    plt.plot(_ks, all_residuals_momentum, label='MIP-VP')
    plt.plot(_ks, all_residuals_sdp_momentum, label='SDP-VP')
    _avg_line, = plt.plot(_ks, simulation_residuals_pdhg_mom, '--', label='sample average')
    plt.fill_between(_ks, simulation_residual_quantiles_pdhg_mom[2], simulation_residual_quantiles_pdhg_mom[0], alpha=.18, label='sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_pdhg_mom[3], ':', color=_avg_line.get_color(), label='sample worst-case')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case KKT residual')
    plt.legend()
    plt.suptitle('network optimization momentum PDHG example')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we overlay PDHG and momentum PDHG to compare the two methods
    directly.
    """)
    return


@app.cell(hide_code=True)
def _(
    K,
    all_residuals,
    all_residuals_momentum,
    all_residuals_sdp,
    all_residuals_sdp_momentum,
    plt,
    simulation_residual_quantiles_pdhg,
    simulation_residual_quantiles_pdhg_mom,
    simulation_residuals_pdhg,
    simulation_residuals_pdhg_mom,
):
    from matplotlib.lines import Line2D

    plt.figure()
    _ks = range(1, K+1)

    _methods = [
        ('PDHG', 'C0', all_residuals_sdp, all_residuals,
         simulation_residuals_pdhg, simulation_residual_quantiles_pdhg),
        ('momentum PDHG', 'C1', all_residuals_sdp_momentum, all_residuals_momentum,
         simulation_residuals_pdhg_mom, simulation_residual_quantiles_pdhg_mom),
    ]
    for _, _color, _sdp, _mip, _avg, _quant in _methods:
        plt.plot(_ks, _sdp, '-', color=_color)
        plt.plot(_ks, _mip, '--', color=_color)
        plt.plot(_ks, _avg, ':', color=_color)
        plt.fill_between(_ks, _quant[2], _quant[0], color=_color, alpha=.18)

    _method_handles = [
        Line2D([], [], color=_color, label=_name)
        for _name, _color, *_ in _methods
    ]
    _style_handles = [
        Line2D([], [], color='gray', linestyle='-', label='SDP-VP'),
        Line2D([], [], color='gray', linestyle='--', label='MIP-VP'),
        Line2D([], [], color='gray', linestyle=':', label='sample average'),
    ]

    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel('Worst-case KKT residual')
    _leg = plt.legend(handles=_method_handles, loc='upper right')
    plt.gca().add_artist(_leg)
    plt.legend(handles=_style_handles, loc='lower left')
    plt.suptitle('network optimization: PDHG vs. momentum PDHG')
    return


if __name__ == "__main__":
    app.run()
