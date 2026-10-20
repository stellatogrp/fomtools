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
    import scipy.sparse as spa

    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "fomtools").is_dir():
            repo_root = candidate
            break
    else:
        repo_root = start

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from docs.tutorials.helper_functions.control import (
        admm_step,
        build_admm_lhs,
        build_control_qp,
        estimate_control_pepit_radius,
        make_control_dynamics,
    )
    from fomtools.proxable_functions import IndBox

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
        IndBox,
        admm_step,
        build_admm_lhs,
        build_control_qp,
        estimate_control_pepit_radius,
        make_control_dynamics,
        mo,
        np,
        plt,
        solver_params,
        spa,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem setup: Constrained optimal control

    [Back to the verification tutorials](./)

    In this tutorial, we verify an OSQP-style ADMM method for a finite-horizon constrained optimal control problem. The system has state $s_t \in \mathbf{R}^{n_s}$, input $u_t \in \mathbf{R}^{n_u}$, and initial state parameter $x$.

    \begin{array}{lll}
        \text{minimize} & s_T^T Q_T s_T + \sum_{t=1}^{T-1} s_t^T Q s_t + u_t^T R u_t \\
        \text{subject to} & s_{t+1} = A s_t + B u_t, & t=1,\dots,T-1 \\
        & s_{\min} \le s_t \le s_{\max}, & t=1,\dots,T \\
        & u_{\min} \le u_t \le u_{\max}, & t=1,\dots,T-1 \\
        & s_1 = x,
    \end{array}

    with optimization variable $z = (u_1, \dots, u_{T-1}) \in \mathbf{R}^{(T-1) n_u}$ and parameter $x \in \mathbf{R}^{n_s}$.
    The verification problem searches over a bounded set of initial ADMM iterates and, depending on the experiment, either fixes $x$ or lets it vary over the prescribed box.
    """)
    return


@app.cell
def _(make_control_dynamics, np, spa):
    # problem data: state dimension n_s = 12, input dimension n_u = 4
    A_dym, B_dym = make_control_dynamics()
    (n_s, n_u) = B_dym.shape

    T = 3        # finite horizon t=1,2,...,T

    Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.]) # state cost matrix Q_t = Q_T = Q
    R = 0.1*spa.eye(n_u) # input cost matrix

    # State constraints are unconstrained in this tutorial.

    # initial state s_init as parameter x: the set X = [x_lb, x_ub]
    x_lb = np.array([-np.pi/6, -np.pi/6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_ub = np.array([ np.pi/6,  np.pi/6, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_init = 2.0*np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # initial x when we fix it

    # input constraint
    u_min = -0.99*np.ones(n_u)
    u_max =  2.41*np.ones(n_u)
    u_stack_min = np.tile(u_min, T - 1)
    u_stack_max = np.tile(u_max, T - 1)
    return (
        A_dym,
        B_dym,
        Q,
        R,
        T,
        n_s,
        n_u,
        u_stack_max,
        u_stack_min,
        x_init,
        x_lb,
        x_ub,
    )


@app.cell
def _():
    # ADMM parameters
    sigma = 1e-4
    rho = 1.  # 0.1, 1, 10
    return rho, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quadratic program form

    Now, consider the equivalent problem of the form:

    \begin{array}{ll}
        \text{minimize} & (1/2) z^T P z + q(x)^T z \\
        \text{subject to} & l(x) \le Mz \le u(x),
    \end{array}

    where $z = (u_1, \dots, u_{T-1})$. Note that for $t=0,\dots,T-1$,

    $$ s_{t+1} = A s_t + B u_t = A^t x + \sum_{m=1}^{t} A^{t-m} B u_m = A^t x + \begin{bmatrix} A^{t-1} B & \dots & B & 0 & \dots & 0 \end{bmatrix} z. $$

    In verification problem (VP) implementation, we use the optimization variable `xz_stack` defined as $(x, u_1, \dots, u_{T-1}) \in \mathbf{R}^{n_s + (T-1) n_u}$.
    This makes the initial state constraints and input box constraints part of one stacked box constraint.
    """)
    return


@app.cell
def _(A_dym, B_dym, Q, R, T, build_control_qp, np):
    M, P, q, n_xz = build_control_qp(A_dym, B_dym, Q, R, T)
    z0_lb, z0_ub = -np.ones(n_xz), np.ones(n_xz)
    v0_lb, v0_ub = np.zeros(n_xz), np.ones(n_xz)
    return M, P, n_xz, q, v0_lb, v0_ub, z0_lb, z0_ub


@app.cell
def _(M, build_admm_lhs, rho, sigma):
    def build_lhs(P):
        """OSQP linear-solve matrix ``P + sigma I + rho M^T M`` (sparse)."""
        return build_admm_lhs(M, P, rho, sigma)

    return (build_lhs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order method: OSQP-style ADMM

    We solve the boxed quadratic program with the splitting used by OSQP:

    \begin{array}{rcl}
        w^{k+1} &=& \Pi_{[l(x), u(x)]}(v^k), \\
        z^{k+1} &=& (P + \sigma I + \rho M^T M)^{-1}
            \left(\sigma z^k - q + \rho M^T(2w^{k+1} - v^k)\right), \\
        v^{k+1} &=& v^k + Mz^{k+1} - w^{k+1}.
    \end{array}

    To compare all verification methods in the same units, we measure the
    squared Euclidean fixed-point residual of the OSQP state:

    $$
        \left\| z^{k+1} - z^k \right\|_2^2 + \left\| v^{k+1} - v^k \right\|_2^2.
    $$

    The projection $\Pi_{[l(x), u(x)]}$ forces the state coordinates to equal
    $x$ and clips the input coordinates to $[u_{\min}, u_{\max}]$. Because the
    simulator cannot project onto the resulting zero-width state box, each mode
    passes the appropriate projection into the shared `admm_step`: the
    simulator clips the input block with `satlin` and stacks $x$ on top, while
    the MIP verification backend projects onto an `IndBox` whose
    bounds pin the state block to $x$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verification Problem

    We now describe how to use `fomtools` to verify the worst-case convergence rate of the OSQP-style ADMM method for constrained optimal control. We compare randomly sampled simulations, exact MIP verification, and a PEPit bound, up to $K$ iterations. The initial state parameter $x$ can either be fixed at $x_\text{init}$ or allowed to vary over its prescribed box.
    """)
    return


@app.cell
def _():
    K = 20  # Maximum iteration number
    fix_initial_x = True  # False lets x vary over [x_lb, x_ub].
    return K, fix_initial_x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulator class

    We first simulate the ADMM algorithm by sampling the initial iterates
    $(z^0,v^0)$ over their prescribed boxes. The parameter $x$ is either fixed
    at $x_\text{init}$ or sampled from $[x_\mathrm{lb},x_\mathrm{ub}]$,
    according to `fix_initial_x`; the MIP and radius computation use the same
    choice. The samples are drawn once below and reused by every simulation
    cell.
    """)
    return


@app.cell
def _(
    fix_initial_x,
    n_s,
    n_xz,
    np,
    v0_lb,
    v0_ub,
    x_init,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    n_samples = 10000
    _rng = np.random.default_rng(0)

    if fix_initial_x:
        x_samples = np.repeat(x_init[np.newaxis, :], n_samples, axis=0)
    else:
        x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, n_s))
    z0_samples = _rng.uniform(low=z0_lb, high=z0_ub, size=(n_samples, n_xz))
    v0_samples = _rng.uniform(low=v0_lb, high=v0_ub, size=(n_samples, n_xz))
    return v0_samples, x_samples, z0_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulation is run through the following function.
    """)
    return


@app.cell
def _(
    M,
    admm_step,
    build_lhs,
    n_s,
    n_xz,
    q,
    rho,
    sigma,
    u_stack_max,
    u_stack_min,
):
    from fomtools.verifier import Verifier

    def admm_simulation(x_samples, z0_samples, v0_samples, P, K, ord=2):
        """Simulate ADMM on sampled instances.

        Builds a simulator-mode :class:`Verifier`, registers the initial-state
        parameter ``x`` and the initial iterates ``(z0, v0)`` as sampled
        parameters, rolls out the recurrence with ``admm_step``, and simulates
        via ``VP.solve()``.
        The projection clips the input block with ``satlin`` and stacks ``x``
        on top. Each ``*_samples`` array has shape ``(n_samples, dim)`` (a
        single-row array replays one instance), and the returned ``residuals``
        has shape ``(n_samples, K)``. Summary statistics (``mean``, ``min``,
        ``max``, ...) are available on the returned instance ``VP`` via
        ``VP[...]``.
        """
        VP = Verifier(method='simulator', obbt_solver=None)
        x_param = VP.add_param(ndim=n_s, samples=x_samples)
        z = VP.add_param(n_xz, samples=z0_samples)
        v = VP.add_param(n_xz, samples=v0_samples)

        def project(VP, v_prev):
            w_control = VP.step.satlin(v_prev[n_s:], l=u_stack_min, u=u_stack_max)
            return VP.add_param_stack([x_param, w_control])

        lhs_mat = build_lhs(P).toarray()
        for _ in range(K):
            z, v, residual = admm_step(
                VP, M, q, rho, sigma, z, v, project, lhs_mat
            )
            VP.set_objective_norm(residual, ord)

        residuals = VP.solve()
        return VP, residuals

    return Verifier, admm_simulation


@app.cell
def _(K, P, admm_simulation, np, v0_samples, x_samples, z0_samples):
    _VP, _residuals = admm_simulation(x_samples, z0_samples, v0_samples, P, K, ord=2)

    print(
        f"Summary statistics:\n"
        f"mean: {_VP['mean']:.6e}\n"
        f"min: {_VP['min']:.6e}\n"
        f"max: {_VP['max']:.6e}\n"
        f"median: {_VP['median']:.6e}\n"
    )

    simulation_residuals_admm = np.mean(_residuals, axis=0)
    # Quantile 1.0 gives the in-sample worst case; indices are [90%, 50%, 10%, max].
    simulation_residual_quantiles_admm = np.quantile(_residuals, [0.90, 0.50, 0.10, 1.0], axis=0)
    return simulation_residual_quantiles_admm, simulation_residuals_admm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm verification through verification problem (VP)

    ### MIP-VP
    Maximizing a squared Euclidean norm would make the verification problem a
    nonconvex MIQP. Instead, the MIP maximizes the infinity norm of the same
    OSQP-state residual
    $$
        r^k=(z^{k+1}-z^k,\;v^{k+1}-v^k).
    $$
    Since $r^k\in\mathbf{R}^{2n_{xz}}$, the norm inequality
    $$
        \|r^k\|_2^2 \le 2n_{xz}\|r^k\|_\infty^2
    $$
    converts the exact MIP value into a valid upper bound in the squared
    Euclidean units used by PEP and the simulations. The verification below
    rolls out the same `admm_step`, projecting onto an `IndBox` whose bounds
    pin the state block to $x$.
    """)
    return


@app.cell
def _(
    IndBox,
    M,
    T,
    Verifier,
    admm_step,
    build_lhs,
    fix_initial_x,
    n_s,
    n_u,
    n_xz,
    np,
    q,
    rho,
    sigma,
    u_stack_max,
    u_stack_min,
    v0_lb,
    v0_ub,
    x_init,
    x_lb,
    x_ub,
    z0_lb,
    z0_ub,
):
    def admm_verification(K, method, P, ord=np.inf, **verifier_kwargs):
        """Solve the optimal-control ADMM verification problem for one backend.

        Builds a ``Verifier`` in ``method`` mode, registers the initial-state
        parameter (fixed at ``x_init`` or searched over ``[x_lb, x_ub]`` per
        ``fix_initial_x``) and the initial iterate boxes, and rolls out
        ``admm_step`` projecting onto the stacked ``IndBox``. Returns
        ``(VP, residuals)`` with one worst-case residual per iteration; the MIP
        uses the infinity-norm objective, so the caller converts it to squared
        Euclidean units.
        """
        VP = Verifier(method=method, **verifier_kwargs)

        if fix_initial_x:
            x_param = VP.add_param(n_s, lb=x_init, ub=x_init)
        else:
            x_param = VP.add_param(n_s, lb=x_lb, ub=x_ub)

        # l(x) <= M@z <= u(x): pin the state block to x, clip the input block.
        l_param = VP.add_param((T-1)*n_u, lb=u_stack_min, ub=u_stack_min)
        u_param = VP.add_param((T-1)*n_u, lb=u_stack_max, ub=u_stack_max)
        l_x = VP.add_param_stack([x_param, l_param])
        u_x = VP.add_param_stack([x_param, u_param])
        indbox = IndBox(l=l_x, u=u_x)

        def project(VP, v_prev):
            return VP.prox(indbox, v_prev)

        z = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        v = VP.iterate_trajectory(K + 1, collapse_expr='auto')
        z[0] = VP.add_iterate(n_xz, lb=z0_lb, ub=z0_ub)
        v[0] = VP.add_iterate(n_xz, lb=v0_lb, ub=v0_ub)
        lhs_mat = build_lhs(P)

        residuals = []
        for k in range(1, K + 1):
            z, v, residual = admm_step(
                VP, M, q, rho, sigma, z, v, project, lhs_mat, k=k
            )
            VP.set_objective_norm(residual, ord)
            residuals.append(VP.solve())
        return VP, residuals

    return (admm_verification,)


@app.cell
def _(K, P, admm_verification, n_xz, np, solver_params):
    # Solve VP for optimal control example (infinity-norm objective).
    _, all_residuals_inf = admm_verification(
        K, 'mip', P, ord=np.inf, solver='gurobi', params=solver_params,
    )

    # Convert the infinity-norm guarantee to squared Euclidean units.
    all_residuals = 2 * n_xz * np.square(all_residuals_inf)
    print(f"infinity-norm MIP bounds: {all_residuals_inf}")
    print(f"squared-Euclidean upper bounds: {all_residuals}")
    return (all_residuals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PEP-obtained worst-case convergence rate

    We compare the MIP result with a PEP that uses the same OSQP recurrence,
    including the projection-first ordering and the parameters $\sigma$ and
    $\rho$. For $M=I$, the linear solve is the proximal step
    $$
        z^{k+1}
        = \mathbf{prox}_{f/(\sigma+\rho)}
        \left(
            \frac{\sigma z^k+\rho(2w^{k+1}-v^k)}
                 {\sigma+\rho}
        \right).
    $$

    PEP still considers the larger class of all $L$-smooth,
    $\mu$-strongly-convex functions and all convex indicator sets:

    \begin{array}{ll}
        \text{minimize} & f(z) + g(z),
    \end{array}

    so it does not retain the exact matrix $P$, dimension, or box geometry. We
    show both the bound conditioned on the empirical radius above and a
    rate-shape curve calibrated to match the norm-converted MIP upper bound at
    $K=1$. The calibrated curve is not an independent absolute guarantee.

    For the radius-conditioned PEP comparison, we estimate the OSQP-state
    initial radius
    $$
        \|z^0-z^\star\|_2^2+\|v^0-v^\star\|_2^2 \le R^2
    $$
    from the same initial-state parameter set used by the MIP. Because the
    maximum below is taken over random initial iterates, it is an empirical
    radius rather than a certificate for every point in the initialization
    boxes.
    """)
    return


@app.cell(hide_code=True)
def _(
    K,
    M,
    P,
    all_residuals,
    build_lhs,
    estimate_control_pepit_radius,
    np,
    q,
    rho,
    sigma,
    u_stack_max,
    u_stack_min,
    v0_samples,
    x_samples,
    z0_samples,
):
    from docs.tutorials.helper_functions.pepit_runner import PEPitRunner

    _pepit_radius_empirical = estimate_control_pepit_radius(
        M,
        q,
        rho,
        sigma,
        build_lhs(P),
        u_stack_min,
        u_stack_max,
        z0_samples,
        v0_samples,
        x_samples,
    )
    _P_eigs = np.linalg.eigvalsh(P.toarray())
    _mu = float(_P_eigs.min())
    _L = float(_P_eigs.max())
    _runner = PEPitRunner()
    all_residuals_PEP_radius, _ = _runner.osqp_admm_residuals(
        K=K,
        mu=_mu,
        L=_L,
        sigma=sigma,
        rho=rho,
        R=_pepit_radius_empirical,
    )
    all_residuals_PEP, pepit_radius_calibrated = _runner.osqp_admm_residuals(
        K=K,
        mu=_mu,
        L=_L,
        sigma=sigma,
        rho=rho,
        calibrate_to=all_residuals[0],
    )
    return all_residuals_PEP, all_residuals_PEP_radius


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical Result

    We plot every curve in squared Euclidean units. The MIP curve uses the
    dimension-dependent norm conversion above; simulation and PEP use
    the squared Euclidean residual directly, and the simulation includes the
    sample worst-case instance.
    """)
    return


@app.cell
def _(
    K,
    all_residuals,
    all_residuals_PEP,
    all_residuals_PEP_radius,
    plt,
    simulation_residual_quantiles_admm,
    simulation_residuals_admm,
):
    plt.figure()
    _ks = range(1, K+1)
    plt.plot(_ks, all_residuals_PEP, label='ADMM (PEPit, matched at K=1)')
    plt.plot(
        _ks,
        all_residuals_PEP_radius,
        ':',
        label='ADMM (PEPit, empirical-radius conditioned)',
    )
    plt.plot(
        _ks,
        all_residuals,
        label=r'ADMM (MIP upper bound via $2n_{xz}\|\cdot\|_\infty^2$)',
    )
    _avg_line, = plt.plot(_ks, simulation_residuals_admm, '--', label='ADMM (in-sample avg)')
    plt.fill_between(_ks, simulation_residual_quantiles_admm[2], simulation_residual_quantiles_admm[0], alpha=.18, label='ADMM sample 10%-90% band')
    plt.plot(_ks, simulation_residual_quantiles_admm[3], ':', color=_avg_line.get_color(), label='ADMM sample worst-case')
    plt.suptitle('optimal control example')
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.ylabel(r'Worst-case squared fixed-point residual $\|r^k\|_2^2$')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
