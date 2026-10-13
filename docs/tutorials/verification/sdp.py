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
    import scipy.sparse as spa
    from scipy.sparse import csr_matrix

    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "fomtools").is_dir():
            repo_root = candidate
            break
    else:
        repo_root = start

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from docs.tutorials.helper_functions.control import admm_step as control_admm_step
    from docs.tutorials.helper_functions.control import (
        build_admm_lhs,
        build_control_qp,
        make_control_dynamics,
    )
    from docs.tutorials.helper_functions.lasso import ista_step, make_lasso_data
    from docs.tutorials.helper_functions.network import make_network_data, pdhg_step
    from docs.tutorials.helper_functions.nnls import make_nnls_matrix, pgd_step
    from docs.tutorials.helper_functions.portfolio import admm_step as portfolio_admm_step
    from docs.tutorials.helper_functions.portfolio import make_portfolio_data
    from fomtools.proxable_functions import IndBox
    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return (
        IndBox,
        Verifier,
        build_admm_lhs,
        build_control_qp,
        control_admm_step,
        csr_matrix,
        ista_step,
        make_control_dynamics,
        make_lasso_data,
        make_network_data,
        make_nnls_matrix,
        make_portfolio_data,
        mo,
        np,
        pdhg_step,
        pgd_step,
        plt,
        portfolio_admm_step,
        spa,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SDP verification problem

    The other verification tutorials ([NNLS](nnls.html), [sparse coding](lasso.html),
    [portfolio](portfolio.html), [network flow](network.html), and
    [optimal control](control.html)) solve the verification problem (VP) *exactly*
    as a mixed-integer linear program with `method='mip'`. This notebook solves the
    same verification problems with `method='sdp'`, following
    [Verification of First-Order Methods for Parametric Quadratic Optimization](https://link.springer.com/article/10.1007/s10107-025-02261-w).

    For each example, the VP has the form

    \begin{array}{ll}
        \text{maximize} & \|r^K\|_2^2 \\
        \text{subject to} & \{z^k\}_{k=0}^K \text{ given by algorithm} \\
        & z^0 \in \mathcal{Z} \\
        & x \in \mathcal{X},
    \end{array}

    where $r^K$ is the algorithm's fixed-point (or KKT) residual at iteration $K$.
    The objective and the iterate updates (including projections onto the
    nonnegative orthant and onto boxes, and soft-thresholding) are quadratic, so the VP is a nonconvex quadratically
    constrained problem. The SDP formulation uses semidefinite lifting to relax
    these nonconvex quadratic constraints into convex linear matrix inequalities.
    Its optimal value is therefore a valid *upper* bound on the worst case,
    rather than the exact worst case the MIP returns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Switching the backend

    The VP is written exactly as for the MIP. You declare parameters and
    initial iterates with bounds, roll out the algorithm with the step
    operators, and set the objective. Only the `Verifier` constructor changes:

    ```python
    VP = Verifier(method='sdp', solver='clarabel', obbt_solver='highs')
    ```

    A few practical points apply to every section below:

    - **Objective.** Every SDP uses the squared Euclidean objective
      `set_objective_norm(r, 2)`, and its value is square-rooted so that every
      curve reports $\|r^k\|_2$. An infinity-norm objective is supported but is
      far slower on the SDP backend.
    - **Iterate trajectories.** `iterate_trajectory(K + 1, collapse_expr='auto')`
      keeps each iterate as its own variable on the SDP backend. This gives a
      sparser lifted matrix that solves faster; see the appendix.
    - **One SDP per horizon.** Each horizon $K$ is a separate relaxation. The
      lifting accumulates one relaxed nonconvexity per iteration, so the bound
      typically loosens as $K$ grows.
    - **Solvers.** The SDP is solved by Clarabel. Optimization-based bound
      tightening (OBBT) uses HiGHS, so this notebook does not need Gurobi.

    We compare each SDP bound with simulations of the same algorithm on
    $10^4$ instances drawn uniformly from the same sets the verifier
    searches. The sample worst case is a *lower* bound on the true worst case,
    so at every $K$ the expected ordering is

    $$\text{sample worst-case} \;\le\; \text{true worst case} \;\le\; \text{SDP-VP}.$$

    The exact worst case, given by the MIP, lies in between. Each example's own
    notebook plots it.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def plot_sdp_vs_samples(K, methods, title, ylabel=r'Worst-case fixed-point residual $\|r^k\|_2$'):
        """Overlay SDP-VP bounds and sample statistics for each method.

        ``methods`` is a list of ``(name, sdp_bounds, sample_residuals)``, where
        ``sdp_bounds`` has length ``K`` (or is ``None`` to plot the samples only)
        and ``sample_residuals`` has shape ``(n_samples, K)``. Methods are
        distinguished by color and data series by line style.
        """
        fig = plt.figure(figsize=(8, 4.5), layout='constrained')
        ks = range(1, K + 1)
        lines = []
        for i, (_name, sdp, samples) in enumerate(methods):
            color = f'C{i}'
            avg = np.mean(samples, axis=0)
            # Quantile indices are [90%, 10%, max]; 1.0 is the in-sample worst case.
            q90, q10, worst = np.quantile(samples, [0.90, 0.10, 1.0], axis=0)
            if sdp is not None:
                plt.plot(ks, sdp, '-', color=color)
                lines.append(np.asarray(sdp, dtype=float))
            plt.plot(ks, worst, ':', color=color)
            plt.plot(ks, avg, '--', color=color)
            plt.fill_between(ks, q10, q90, color=color, alpha=.18)
            lines += [worst, avg]

        method_handles = [
            Line2D([], [], color=f'C{i}', label=name)
            for i, (name, *_) in enumerate(methods)
        ]
        style_handles = [
            Line2D([], [], color='gray', linestyle=':', label='sample worst-case'),
            Line2D([], [], color='gray', linestyle='--', label='sample average'),
            Patch(color='gray', alpha=.18, label='sample 10%-90% band'),
        ]
        if any(sdp is not None for _, sdp, _ in methods):
            style_handles.insert(0, Line2D([], [], color='gray', linestyle='-', label='SDP-VP'))

        # Bound the axis by the plotted *lines*, not by the 10%-90% band, whose
        # lower edge can decay to ~0 and squeeze every curve into the top.
        plt.yscale('log')
        plt.ylim(bottom=0.5 * float(np.min(np.concatenate(lines))))
        plt.xticks(list(ks))
        plt.xlabel(r'$K$')
        plt.ylabel(ylabel)
        fig.legend(handles=method_handles + style_handles, loc='outside right center')
        plt.suptitle(title)
        return plt.gca()

    return (plot_sdp_vs_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nonnegative least squares

    Projected gradient descent (PGD) and its momentum variant applied to
    $\min_{z \ge 0}\ (1/2)\|Az - x\|_2^2$, with $x \in [10, 11]^5$,
    $z^0 \in [4, 6]^3$, and step size $\eta = 1/(2L)$. See the
    [NNLS tutorial](nnls.html) for the full setup.
    """)
    return


@app.cell
def _(make_nnls_matrix, np):
    A_nnls = make_nnls_matrix(m=5, n=3)
    m_nnls, n_nnls = A_nnls.shape
    x_lb_nnls, x_ub_nnls = 10.0, 11.0  # parameter set
    z0_lb_nnls, z0_ub_nnls = 4.0, 6.0  # initial condition
    eta_nnls = 0.5 / np.max(np.linalg.eigvalsh(A_nnls.T @ A_nnls))
    K_nnls = 7
    return (
        A_nnls,
        K_nnls,
        eta_nnls,
        m_nnls,
        n_nnls,
        x_lb_nnls,
        x_ub_nnls,
        z0_lb_nnls,
        z0_ub_nnls,
    )


@app.cell
def _(
    A_nnls,
    Verifier,
    eta_nnls,
    m_nnls,
    n_nnls,
    np,
    pgd_step,
    x_lb_nnls,
    x_ub_nnls,
    z0_lb_nnls,
    z0_ub_nnls,
):
    def pgd_sdp_verification(K, momentum=False, **verifier_kwargs):
        """Solve the PGD verification problem as an SDP for horizons 1, ..., K.

        Returns ``(VP, residuals)``, with one upper bound on $\\|r^k\\|_2$ per
        horizon. Extra ``verifier_kwargs`` are forwarded to ``Verifier``.
        """
        VP = Verifier(method='sdp', **verifier_kwargs)
        x_param = VP.add_param(m_nnls, lb=x_lb_nnls, ub=x_ub_nnls)
        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n_nnls, lb=z0_lb_nnls, ub=z0_ub_nnls)
        w = z[0]

        residuals = []
        for k in range(1, K + 1):
            z, w, residual = pgd_step(
                VP, A_nnls, eta_nnls, z, w, x_param, k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, 2)
            residuals.append(np.sqrt(VP.solve()))  # squared objective -> ||r^k||_2
        return VP, residuals

    return (pgd_sdp_verification,)


@app.cell
def _(K_nnls, pgd_sdp_verification):
    _, sdp_pgd = pgd_sdp_verification(
        K_nnls, momentum=False, solver='clarabel', obbt_solver='highs',
    )
    _, sdp_pgd_momentum = pgd_sdp_verification(
        K_nnls, momentum=True, solver='clarabel', obbt_solver='highs',
    )
    return sdp_pgd, sdp_pgd_momentum


@app.cell
def _(
    A_nnls,
    K_nnls,
    Verifier,
    eta_nnls,
    m_nnls,
    n_nnls,
    np,
    pgd_step,
    x_lb_nnls,
    x_ub_nnls,
    z0_lb_nnls,
    z0_ub_nnls,
):
    def _simulate(momentum, n_samples=10000):
        """Simulate PGD on uniform samples; returns ||r^k||_2, shape (n_samples, K)."""
        rng = np.random.default_rng(0)
        x_samples = rng.uniform(x_lb_nnls, x_ub_nnls, size=(n_samples, m_nnls))
        z0_samples = rng.uniform(z0_lb_nnls, z0_ub_nnls, size=(n_samples, n_nnls))

        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=m_nnls, samples=x_samples)
        z = w = VP.add_param(ndim=n_nnls, samples=z0_samples)
        for k in range(1, K_nnls + 1):
            z, w, residual = pgd_step(
                VP, A_nnls, eta_nnls, z, w, x_param, k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, 2)
        return np.sqrt(VP.solve())  # squared objective -> ||r^k||_2

    samples_pgd = _simulate(momentum=False)
    samples_pgd_momentum = _simulate(momentum=True)
    return samples_pgd, samples_pgd_momentum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For plain PGD, the SDP bound still decreases with $K$. For momentum PGD it
    *increases* with $K$ and becomes uninformative after a few iterations, as
    the relaxation gap compounds. We do not replace such a curve by its running
    minimum. Each $K$ bounds a different quantity, $\|r^K\|_2$, and only for a
    nonexpansive iteration is $\|r^k\|_2$ guaranteed to be non-increasing in
    $k$. The momentum variant is not nonexpansive.
    """)
    return


@app.cell
def _(
    K_nnls,
    plot_sdp_vs_samples,
    samples_pgd,
    samples_pgd_momentum,
    sdp_pgd,
    sdp_pgd_momentum,
):
    plot_sdp_vs_samples(
        K_nnls,
        [
            ('PGD', sdp_pgd, samples_pgd),
            ('momentum PGD', sdp_pgd_momentum, samples_pgd_momentum),
        ],
        'NNLS: SDP-VP for PGD and momentum PGD',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sparse coding

    ISTA and FISTA applied to
    $\min_z\ (1/2)\|Dz - x\|_2^2 + \lambda\|z\|_1$, with $x$ in the tightest
    box containing the sampled measurements and $z^0 \in [-1, 1]^n$. See the
    [sparse coding tutorial](lasso.html) for the full setup.
    """)
    return


@app.cell
def _(make_lasso_data):
    p_lasso, n_lasso = 3, 4
    D_lasso, x_lb_lasso, x_ub_lasso, _, eta_lasso, lambd_lasso = make_lasso_data(
        p=p_lasso, n=n_lasso, lambd=1e-1, num_signals=20, seed=0
    )
    s0_lb_lasso, s0_ub_lasso = -1.0, 1.0
    K_lasso = 7
    return (
        D_lasso,
        K_lasso,
        eta_lasso,
        lambd_lasso,
        n_lasso,
        p_lasso,
        s0_lb_lasso,
        s0_ub_lasso,
        x_lb_lasso,
        x_ub_lasso,
    )


@app.cell
def _(
    D_lasso,
    Verifier,
    eta_lasso,
    ista_step,
    lambd_lasso,
    n_lasso,
    np,
    p_lasso,
    s0_lb_lasso,
    s0_ub_lasso,
    x_lb_lasso,
    x_ub_lasso,
):
    def ista_sdp_verification(K, momentum=False, **verifier_kwargs):
        """Solve the ISTA/FISTA verification problem as an SDP for horizons 1, ..., K.

        Returns ``(VP, residuals)``, with one upper bound on $\\|r^k\\|_2$ per
        horizon. Extra ``verifier_kwargs`` are forwarded to ``Verifier``.
        """
        VP = Verifier(method='sdp', **verifier_kwargs)
        x_param = VP.add_param(p_lasso, lb=x_lb_lasso, ub=x_ub_lasso)
        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n_lasso, lb=s0_lb_lasso, ub=s0_ub_lasso)
        y, t = z[0], 1.0

        residuals = []
        for k in range(1, K + 1):
            z, y, t, residual = ista_step(
                VP, D_lasso, eta_lasso, lambd_lasso, z, y, x_param, t,
                k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, 2)
            residuals.append(np.sqrt(VP.solve()))  # squared objective -> ||r^k||_2
        return VP, residuals

    return (ista_sdp_verification,)


@app.cell
def _(K_lasso, ista_sdp_verification):
    _, sdp_ista = ista_sdp_verification(
        K_lasso, momentum=False, solver='clarabel', obbt_solver='highs',
    )
    _, sdp_fista = ista_sdp_verification(
        K_lasso, momentum=True, solver='clarabel', obbt_solver='highs',
    )
    return sdp_fista, sdp_ista


@app.cell
def _(
    D_lasso,
    K_lasso,
    Verifier,
    eta_lasso,
    ista_step,
    lambd_lasso,
    n_lasso,
    np,
    p_lasso,
    s0_lb_lasso,
    s0_ub_lasso,
    x_lb_lasso,
    x_ub_lasso,
):
    def _simulate(momentum, n_samples=10000):
        """Simulate ISTA/FISTA on uniform samples; returns ||r^k||_2, shape (n_samples, K)."""
        rng = np.random.default_rng(0)
        x_samples = rng.uniform(x_lb_lasso, x_ub_lasso, size=(n_samples, p_lasso))
        s0_samples = rng.uniform(s0_lb_lasso, s0_ub_lasso, size=(n_samples, n_lasso))

        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=p_lasso, samples=x_samples)
        z = y = VP.add_param(n_lasso, samples=s0_samples)
        t = 1.0
        for k in range(1, K_lasso + 1):
            z, y, t, residual = ista_step(
                VP, D_lasso, eta_lasso, lambd_lasso, z, y, x_param, t,
                k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, 2)
        return np.sqrt(VP.solve())  # squared objective -> ||r^k||_2

    samples_ista = _simulate(momentum=False)
    samples_fista = _simulate(momentum=True)
    return samples_fista, samples_ista


@app.cell
def _(
    K_lasso,
    plot_sdp_vs_samples,
    samples_fista,
    samples_ista,
    sdp_fista,
    sdp_ista,
):
    plot_sdp_vs_samples(
        K_lasso,
        [
            ('ISTA', sdp_ista, samples_ista),
            ('FISTA', sdp_fista, samples_fista),
        ],
        'Sparse coding: SDP-VP for ISTA and FISTA',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Portfolio optimization

    OSQP-style ADMM applied to the mean-variance portfolio problem
    $\min\ (1/2)z^T\Sigma z - \mu^T z$ subject to $0 \le z \le u_{\max}$ and
    $\mathbf{1}^T z = 1$. The problem data is fixed, $z^0 = 0$, and the
    verifier searches over the initial dual iterate $v^0 \in [0, 1]^{n+1}$.
    See the [portfolio tutorial](portfolio.html) for the full setup.
    """)
    return


@app.cell
def _(csr_matrix, make_portfolio_data, np):
    n_port = 4
    Sigma_port, mu_port, A_port, l_port, u_port = make_portfolio_data(
        n=n_port, seed=0, position_limit=0.6
    )
    x0_port = np.zeros(n_port)
    v0_lb_port, v0_ub_port = np.zeros(n_port + 1), np.ones(n_port + 1)

    sigma_port, rho_port = 1e-4, 1.0
    lhs_port = csr_matrix(
        Sigma_port + sigma_port * np.eye(n_port) + rho_port * A_port.T @ A_port
    )
    K_port = 5
    return (
        A_port,
        K_port,
        l_port,
        lhs_port,
        mu_port,
        n_port,
        rho_port,
        sigma_port,
        u_port,
        v0_lb_port,
        v0_ub_port,
        x0_port,
    )


@app.cell
def _(
    A_port,
    Verifier,
    l_port,
    lhs_port,
    mu_port,
    n_port,
    np,
    portfolio_admm_step,
    rho_port,
    sigma_port,
    u_port,
    v0_lb_port,
    v0_ub_port,
    x0_port,
):
    def portfolio_sdp_verification(K, **verifier_kwargs):
        """Solve the portfolio ADMM verification problem as an SDP for horizons 1, ..., K.

        Returns ``(VP, residuals)``, with one upper bound on $\\|r^k\\|_2$ per
        horizon. Extra ``verifier_kwargs`` are forwarded to ``Verifier``.
        """
        VP = Verifier(method='sdp', **verifier_kwargs)
        x = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        v = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        x[0] = VP.add_iterate(n_port, lb=x0_port, ub=x0_port)
        v[0] = VP.add_iterate(n_port + 1, lb=v0_lb_port, ub=v0_ub_port)

        residuals = []
        for k in range(1, K + 1):
            x, v, residual = portfolio_admm_step(
                VP, A_port, l_port, u_port, mu_port, rho_port, sigma_port,
                x, v, lhs_port, k=k,
            )
            VP.set_objective_norm(residual, 2)
            residuals.append(np.sqrt(VP.solve()))  # squared objective -> ||r^k||_2
        return VP, residuals

    return (portfolio_sdp_verification,)


@app.cell
def _(K_port, portfolio_sdp_verification):
    _, sdp_port = portfolio_sdp_verification(
        K_port, solver='clarabel', obbt_solver='highs',
    )
    return (sdp_port,)


@app.cell
def _(
    A_port,
    K_port,
    Verifier,
    l_port,
    lhs_port,
    mu_port,
    n_port,
    np,
    portfolio_admm_step,
    rho_port,
    sigma_port,
    u_port,
    v0_lb_port,
    v0_ub_port,
    x0_port,
):
    def _simulate(n_samples=10000):
        """Simulate ADMM on uniform samples of v0; returns ||r^k||_2, shape (n_samples, K)."""
        rng = np.random.default_rng(0)
        v0_samples = rng.uniform(v0_lb_port, v0_ub_port, size=(n_samples, n_port + 1))

        VP = Verifier(method='simulator', obbt_solver=None)
        x = x0_port
        v = VP.add_param(n_port + 1, samples=v0_samples)
        lhs_dense = lhs_port.toarray()
        for _ in range(K_port):
            x, v, residual = portfolio_admm_step(
                VP, A_port, l_port, u_port, mu_port, rho_port, sigma_port,
                x, v, lhs_dense,
            )
            VP.set_objective_norm(residual, 2)
        return np.sqrt(VP.solve())  # squared objective -> ||r^k||_2

    samples_port = _simulate()
    return (samples_port,)


@app.cell
def _(K_port, plot_sdp_vs_samples, samples_port, sdp_port):
    plot_sdp_vs_samples(
        K_port,
        [('ADMM', sdp_port, samples_port)],
        'Portfolio optimization: SDP-VP for ADMM',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Network flow

    PDHG and its momentum variant applied to a capacitated min-cost flow linear
    program. The verifier searches over the demand vector $x$ and the initial
    primal-dual iterates $(z^0, v^0, w^0)$. See the
    [network flow tutorial](network.html) for the full setup.

    Two choices differ from the MIP notebook:

    - **Numeric problem data.** Capacity, cost, and supply are fixed in this
      experiment, so they are passed as numeric constants rather than
      zero-width parameters. Parameters would only add redundant variables to
      the lifted matrix.
    - **2-norm objective.** The network notebook reports the KKT residual in
      the infinity norm. An infinity-norm SDP does not finish in reasonable
      time here, so we use the squared Euclidean objective and simulate in the
      same norm. Since $\|r\|_\infty \le \|r\|_2$, the SDP curve is still a
      valid, more conservative, upper bound on the infinity-norm worst case.
    """)
    return


@app.cell
def _(make_network_data, np):
    # Bipartite graph with n_s supply and n_d demand nodes; each edge is added
    # with probability 0.5.
    n_s_net, n_d_net = 6, 3
    A_s_net, A_d_net, b_s_net, c_net, mu_net = make_network_data(
        n_supply=n_s_net,
        n_demand=n_d_net,
        edge_probability=0.5,
        graph_seed=1,
        data_seed=2,
    )
    n_e_net = A_s_net.shape[1]

    # Parameter set X = [x_lb, x_ub] of demands, and initial-iterate boxes
    x_lb_net, x_ub_net = -7.0 * np.ones(n_d_net), -6.0 * np.ones(n_d_net)
    z0_lb_net, z0_ub_net = np.zeros(n_e_net), c_net.copy()
    v0_lb_net, v0_ub_net = np.zeros(n_s_net), np.ones(n_s_net)
    w0_lb_net, w0_ub_net = -np.ones(n_d_net), np.ones(n_d_net)

    eta_net = 0.5 / np.linalg.norm(np.hstack([-A_s_net.T, A_d_net.T]), 2)
    K_net = 4
    return (
        A_d_net,
        A_s_net,
        K_net,
        b_s_net,
        c_net,
        eta_net,
        mu_net,
        n_d_net,
        n_e_net,
        n_s_net,
        v0_lb_net,
        v0_ub_net,
        w0_lb_net,
        w0_ub_net,
        x_lb_net,
        x_ub_net,
        z0_lb_net,
        z0_ub_net,
    )


@app.cell
def _(
    A_d_net,
    A_s_net,
    Verifier,
    b_s_net,
    c_net,
    eta_net,
    mu_net,
    n_d_net,
    n_e_net,
    n_s_net,
    np,
    pdhg_step,
    v0_lb_net,
    v0_ub_net,
    w0_lb_net,
    w0_ub_net,
    x_lb_net,
    x_ub_net,
    z0_lb_net,
    z0_ub_net,
):
    def pdhg_sdp_verification(K, momentum=False, **verifier_kwargs):
        """Solve the PDHG verification problem as an SDP for horizons 1, ..., K.

        Returns ``(VP, residuals)``, with one upper bound on $\\|r^k\\|_2$ per
        horizon. Extra ``verifier_kwargs`` are forwarded to ``Verifier``.
        """
        VP = Verifier(method='sdp', **verifier_kwargs)
        x_param = VP.add_param(n_d_net, lb=x_lb_net, ub=x_ub_net)

        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        v = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        w = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n_e_net, lb=z0_lb_net, ub=z0_ub_net)
        v[0] = VP.add_iterate(n_s_net, lb=v0_lb_net, ub=v0_ub_net)
        w[0] = VP.add_iterate(n_d_net, lb=w0_lb_net, ub=w0_ub_net)

        residuals = []
        for k in range(1, K + 1):
            z, v, w, residual = pdhg_step(
                VP, A_s_net, A_d_net, eta_net, z, v, w, x_param,
                c_net, mu_net, b_s_net, k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, 2)
            residuals.append(np.sqrt(VP.solve()))  # squared objective -> ||r^k||_2
        return VP, residuals

    return (pdhg_sdp_verification,)


@app.cell
def _(K_net, pdhg_sdp_verification):
    _, sdp_pdhg = pdhg_sdp_verification(
        K_net, solver='clarabel', obbt_solver='highs',
    )
    _, sdp_pdhg_momentum = pdhg_sdp_verification(
        K_net, momentum=True, solver='clarabel', obbt_solver='highs',
    )
    return sdp_pdhg, sdp_pdhg_momentum


@app.cell
def _(
    A_d_net,
    A_s_net,
    K_net,
    Verifier,
    b_s_net,
    c_net,
    eta_net,
    mu_net,
    n_d_net,
    n_e_net,
    n_s_net,
    np,
    pdhg_step,
    v0_lb_net,
    v0_ub_net,
    w0_lb_net,
    w0_ub_net,
    x_lb_net,
    x_ub_net,
    z0_lb_net,
    z0_ub_net,
):
    def _simulate(momentum, n_samples=10000):
        """Simulate PDHG on uniform samples; returns ||r^k||_2, shape (n_samples, K)."""
        rng = np.random.default_rng(0)
        x_samples = rng.uniform(x_lb_net, x_ub_net, size=(n_samples, n_d_net))
        z0_samples = rng.uniform(z0_lb_net, z0_ub_net, size=(n_samples, n_e_net))
        v0_samples = rng.uniform(v0_lb_net, v0_ub_net, size=(n_samples, n_s_net))
        w0_samples = rng.uniform(w0_lb_net, w0_ub_net, size=(n_samples, n_d_net))

        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=n_d_net, samples=x_samples)
        z = VP.add_param(ndim=n_e_net, samples=z0_samples)
        v = VP.add_param(ndim=n_s_net, samples=v0_samples)
        w = VP.add_param(ndim=n_d_net, samples=w0_samples)
        for k in range(1, K_net + 1):
            z, v, w, residual = pdhg_step(
                VP, A_s_net, A_d_net, eta_net, z, v, w, x_param,
                c_net, mu_net, b_s_net, k=k, momentum=momentum,
            )
            VP.set_objective_norm(residual, 2)
        return np.sqrt(VP.solve())  # squared objective -> ||r^k||_2

    samples_pdhg = _simulate(momentum=False)
    samples_pdhg_momentum = _simulate(momentum=True)
    return samples_pdhg, samples_pdhg_momentum


@app.cell
def _(
    K_net,
    plot_sdp_vs_samples,
    samples_pdhg,
    samples_pdhg_momentum,
    sdp_pdhg,
    sdp_pdhg_momentum,
):
    plot_sdp_vs_samples(
        K_net,
        [
            ('PDHG', sdp_pdhg, samples_pdhg),
            ('momentum PDHG', sdp_pdhg_momentum, samples_pdhg_momentum),
        ],
        'Network flow: SDP-VP for PDHG and momentum PDHG',
        ylabel=r'Worst-case KKT residual $\|r^k\|_2$',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal control

    OSQP-style ADMM applied to a finite-horizon constrained optimal control
    problem. The initial state $x$ is fixed at $x_\text{init}$ (set
    `fix_initial_x_ctrl = False` to let it vary over its box), and the
    verifier searches over the initial iterates $(z^0, v^0)$. The projection
    onto $[l(x), u(x)]$ is an `IndBox` whose bounds pin the state block to
    $x$. See the [optimal control tutorial](control.html) for the full setup.

    > **Runtime warning.** The SDP relaxation for this example is
    > substantially more expensive than the others. On the benchmark machine,
    > even the first two horizons took several minutes each. The solve cell
    > below is therefore disabled by default so that the notebook stays
    > practical to run end to end, and the figure shows only the simulation.
    > Uncomment the marked lines to compute the SDP curve.
    """)
    return


@app.cell
def _(make_control_dynamics, np, spa):
    # Problem data: state dimension n_s = 12, input dimension n_u = 4
    A_dyn_ctrl, B_dyn_ctrl = make_control_dynamics()
    n_s_ctrl, n_u_ctrl = B_dyn_ctrl.shape
    T_ctrl = 3  # finite horizon t = 1, 2, ..., T

    Q_ctrl = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    R_ctrl = 0.1 * spa.eye(n_u_ctrl)

    # Initial state x: fixed at x_init, or searched over [x_lb, x_ub]
    fix_initial_x_ctrl = True
    x_lb_ctrl = np.array([-np.pi/6, -np.pi/6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_ub_ctrl = np.array([ np.pi/6,  np.pi/6, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_init_ctrl = 2.0 * np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # Input constraints, stacked over the horizon
    u_stack_min_ctrl = np.tile(-0.99 * np.ones(n_u_ctrl), T_ctrl - 1)
    u_stack_max_ctrl = np.tile(2.41 * np.ones(n_u_ctrl), T_ctrl - 1)

    # ADMM parameters
    sigma_ctrl, rho_ctrl = 1e-4, 1.0
    K_ctrl = 7
    return (
        A_dyn_ctrl,
        B_dyn_ctrl,
        K_ctrl,
        Q_ctrl,
        R_ctrl,
        T_ctrl,
        fix_initial_x_ctrl,
        n_s_ctrl,
        n_u_ctrl,
        rho_ctrl,
        sigma_ctrl,
        u_stack_max_ctrl,
        u_stack_min_ctrl,
        x_init_ctrl,
        x_lb_ctrl,
        x_ub_ctrl,
    )


@app.cell
def _(
    A_dyn_ctrl,
    B_dyn_ctrl,
    Q_ctrl,
    R_ctrl,
    T_ctrl,
    build_admm_lhs,
    build_control_qp,
    np,
    rho_ctrl,
    sigma_ctrl,
):
    M_ctrl, P_ctrl, q_ctrl, n_xz_ctrl = build_control_qp(
        A_dyn_ctrl, B_dyn_ctrl, Q_ctrl, R_ctrl, T_ctrl
    )
    z0_lb_ctrl, z0_ub_ctrl = -np.ones(n_xz_ctrl), np.ones(n_xz_ctrl)
    v0_lb_ctrl, v0_ub_ctrl = np.zeros(n_xz_ctrl), np.ones(n_xz_ctrl)
    # OSQP linear-solve matrix P + sigma I + rho M^T M (sparse)
    lhs_ctrl = build_admm_lhs(M_ctrl, P_ctrl, rho_ctrl, sigma_ctrl)
    return (
        M_ctrl,
        lhs_ctrl,
        n_xz_ctrl,
        q_ctrl,
        v0_lb_ctrl,
        v0_ub_ctrl,
        z0_lb_ctrl,
        z0_ub_ctrl,
    )


@app.cell
def _(
    IndBox,
    M_ctrl,
    T_ctrl,
    Verifier,
    control_admm_step,
    fix_initial_x_ctrl,
    lhs_ctrl,
    n_s_ctrl,
    n_u_ctrl,
    n_xz_ctrl,
    np,
    q_ctrl,
    rho_ctrl,
    sigma_ctrl,
    u_stack_max_ctrl,
    u_stack_min_ctrl,
    v0_lb_ctrl,
    v0_ub_ctrl,
    x_init_ctrl,
    x_lb_ctrl,
    x_ub_ctrl,
    z0_lb_ctrl,
    z0_ub_ctrl,
):
    def control_sdp_verification(K, **verifier_kwargs):
        """Solve the optimal-control ADMM verification problem as an SDP for horizons 1, ..., K.

        Returns ``(VP, residuals)``, with one upper bound on $\\|r^k\\|_2$ per
        horizon. Extra ``verifier_kwargs`` are forwarded to ``Verifier``.
        """
        VP = Verifier(method='sdp', **verifier_kwargs)
        if fix_initial_x_ctrl:
            x_param = VP.add_param(n_s_ctrl, lb=x_init_ctrl, ub=x_init_ctrl)
        else:
            x_param = VP.add_param(n_s_ctrl, lb=x_lb_ctrl, ub=x_ub_ctrl)

        # l(x) <= M@z <= u(x): pin the state block to x, clip the input block.
        n_in = (T_ctrl - 1) * n_u_ctrl
        l_param = VP.add_param(n_in, lb=u_stack_min_ctrl, ub=u_stack_min_ctrl)
        u_param = VP.add_param(n_in, lb=u_stack_max_ctrl, ub=u_stack_max_ctrl)
        indbox = IndBox(
            l=VP.add_param_stack([x_param, l_param]),
            u=VP.add_param_stack([x_param, u_param]),
        )

        def project(VP, v_prev):
            return VP.prox(indbox, v_prev)

        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        v = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n_xz_ctrl, lb=z0_lb_ctrl, ub=z0_ub_ctrl)
        v[0] = VP.add_iterate(n_xz_ctrl, lb=v0_lb_ctrl, ub=v0_ub_ctrl)

        residuals = []
        for k in range(1, K + 1):
            z, v, residual = control_admm_step(
                VP, M_ctrl, q_ctrl, rho_ctrl, sigma_ctrl, z, v, project, lhs_ctrl, k=k
            )
            VP.set_objective_norm(residual, 2)
            residuals.append(np.sqrt(VP.solve()))  # squared objective -> ||r^k||_2
        return VP, residuals

    return (control_sdp_verification,)


@app.cell
def _(K_ctrl, control_sdp_verification):
    # WARNING: this solve can take a very long time, especially as K grows.
    # Uncomment the following lines to enable the SDP solve.
    # _, sdp_ctrl = control_sdp_verification(
    #     K_ctrl, solver='clarabel', obbt_solver='highs',
    # )
    sdp_ctrl = None
    return (sdp_ctrl,)


@app.cell
def _(
    K_ctrl,
    M_ctrl,
    Verifier,
    control_admm_step,
    fix_initial_x_ctrl,
    lhs_ctrl,
    n_s_ctrl,
    n_xz_ctrl,
    np,
    q_ctrl,
    rho_ctrl,
    sigma_ctrl,
    u_stack_max_ctrl,
    u_stack_min_ctrl,
    v0_lb_ctrl,
    v0_ub_ctrl,
    x_init_ctrl,
    x_lb_ctrl,
    x_ub_ctrl,
    z0_lb_ctrl,
    z0_ub_ctrl,
):
    def _simulate(n_samples=10000):
        """Simulate ADMM on uniform samples; returns ||r^k||_2, shape (n_samples, K)."""
        rng = np.random.default_rng(0)
        if fix_initial_x_ctrl:
            x_samples = np.repeat(x_init_ctrl[np.newaxis, :], n_samples, axis=0)
        else:
            x_samples = rng.uniform(x_lb_ctrl, x_ub_ctrl, size=(n_samples, n_s_ctrl))
        z0_samples = rng.uniform(z0_lb_ctrl, z0_ub_ctrl, size=(n_samples, n_xz_ctrl))
        v0_samples = rng.uniform(v0_lb_ctrl, v0_ub_ctrl, size=(n_samples, n_xz_ctrl))

        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=n_s_ctrl, samples=x_samples)
        z = VP.add_param(n_xz_ctrl, samples=z0_samples)
        v = VP.add_param(n_xz_ctrl, samples=v0_samples)

        # The simulator cannot project onto the zero-width state box, so it
        # clips the input block with satlin and stacks x on top instead.
        def project(VP, v_prev):
            w_control = VP.step.satlin(
                v_prev[n_s_ctrl:], l=u_stack_min_ctrl, u=u_stack_max_ctrl
            )
            return VP.add_param_stack([x_param, w_control])

        lhs_dense = lhs_ctrl.toarray()
        for _ in range(K_ctrl):
            z, v, residual = control_admm_step(
                VP, M_ctrl, q_ctrl, rho_ctrl, sigma_ctrl, z, v, project, lhs_dense
            )
            VP.set_objective_norm(residual, 2)
        return np.sqrt(VP.solve())  # squared objective -> ||r^k||_2

    samples_ctrl = _simulate()
    return (samples_ctrl,)


@app.cell
def _(K_ctrl, plot_sdp_vs_samples, samples_ctrl, sdp_ctrl):
    plot_sdp_vs_samples(
        K_ctrl,
        [('ADMM', sdp_ctrl, samples_ctrl)],
        'Optimal control: SDP-VP for ADMM',
    )
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


if __name__ == "__main__":
    app.run()
