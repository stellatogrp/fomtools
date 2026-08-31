"""Restarted accelerated PDHG, in NumPy.

Algorithm 1 of Lu and Yang, *A Practical and Optimal First-Order Method for
Large-Scale Convex Quadratic Programming* (arXiv:2311.07710), on the condensed
program in the inequality form of ``mpc.py``.  Writing the inner index ``k``,
which runs ``0 .. K-1`` and resets at every restart,

    xmd = (1 - 1/beta_k) xbar + (1/beta_k) x,
    y   = proj_+( y + tau_k ( G (theta_k (x - x_prev) + x) - h ) ),
    x   = x - eta_k ( Q xmd + c + G' y ),
    xbar, ybar = running averages with weight 1/beta_k,

and after K of those, ``(x, y) <- (xbar, ybar)``.

The point of this study is that no step of that is a linear solve.  The whole
iteration is matrix-vector products and vector arithmetic, so the generated
code has no factorization to spend its time in.

Two forms of the iteration live here and they are not redundant.
:func:`solve_as_written` is the nested loop above, transcribed from the paper.
:func:`solve` is the same thing as a single flat recurrence over a six-slot
state, which is what can be recorded for code generation: the restart becomes a
coefficient that is nonzero on every K-th iteration rather than a change of
control flow, and every coefficient becomes a table read at the loop counter.
``check_against_fomtools.py`` runs both and requires them to agree, so the flat
form is never taken on trust.

The flat form is what every timed arm runs, NumPy and generated C alike, so the
comparison is between implementations of one recurrence rather than between two
different arrangements of the algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from condensed.mpc import RapdhgProblem

#: Names of the schedule tables, in the order the recording declares them.
SCHEDULES = ("taux", "tth", "tau", "eta", "g1", "g2", "d1", "d2", "m1", "m2")

#: Slot widths of the flat state, in order: x, y, xbar, ybar, x_prev, xmd.
SLOT_ORDER = ("x", "y", "xbar", "ybar", "x_prev", "xmd")


@dataclass
class RapdhgResult:
    """Outcome of a fixed number of rAPDHG iterations."""

    x: np.ndarray
    y: np.ndarray
    xbar: np.ndarray
    ybar: np.ndarray
    x_prev: np.ndarray
    xmd: np.ndarray
    iterations: int
    #: Relative KKT error at the last iterate, or ``None`` when not requested.
    residual: float | None = None

    @property
    def state(self) -> np.ndarray:
        """The flat state vector, in the layout the generated solver carries."""
        return np.concatenate([self.x, self.y, self.xbar, self.ybar, self.x_prev, self.xmd])


def coefficients(problem: RapdhgProblem, k: int) -> tuple[float, float, float, float]:
    """The paper's ``(1/beta_k, theta_k, eta_k, tau_k)`` at inner index ``k``.

    Theorem 1 fixes the four, up to the primal weight ``w``, which multiplies the
    dual step and divides the primal one.  ``beta_0 = 1`` makes ``xmd = x`` and
    resets the running averages, and ``theta_0 = 0`` drops ``x_prev``; both are
    what lets a restart boundary need no special case.
    """
    K = problem.restart
    w = problem.primal_weight
    beta_inv = 2.0 / (k + 2.0)
    theta = k / (k + 1.0)
    eta = (k + 1.0) / (2.0 * (problem.norm_q + K * problem.norm_g)) / w
    tau = (k + 1.0) / (2.0 * K * problem.norm_g) * w
    return beta_inv, theta, eta, tau


def schedule_tables(problem: RapdhgProblem, n_iters: int) -> dict[str, np.ndarray]:
    """The ten coefficient tables, indexed by the 0-based global iteration.

    Every table is a function of the inner index ``k = t mod K``, so the restart
    is periodicity in the tables and nothing else.

    A coefficient block must be *affine* in the declared scalars, so any product
    of two of the paper's coefficients has to be a table in its own right: the
    fit in ``fomtools.codegen.extract_scheduled`` recovers one slope per
    declared name and rejects a block that needs two of them multiplied.  That
    is why there are ten tables here for four coefficients.
    """
    K = problem.restart

    def restart_flag(k: int) -> float:
        return 1.0 if k == K - 1 else 0.0

    rows = {name: np.empty(n_iters) for name in SCHEDULES}
    for t in range(n_iters):
        k = t % K
        kn = (t + 1) % K  # the next inner index, which xmd is built for
        binv, theta, eta, tau = coefficients(problem, k)
        binv_next, _, _, _ = coefficients(problem, kn)
        r = restart_flag(k)

        # Dual step: tau G (theta (x - x_prev) + x) splits into a coefficient on
        # x and one on x_prev.
        rows["taux"][t] = tau * (1.0 + theta)
        rows["tth"][t] = tau * theta
        rows["tau"][t] = tau
        rows["eta"][t] = eta

        # Restart mux.  At k = K-1 these become (1/beta, 1 - 1/beta), which is
        # exactly the running average, so x and xbar coincide and the next inner
        # loop starts from it.
        g1 = 1.0 - r + r * binv
        g2 = r - r * binv
        # Running averages.
        d1 = 1.0 - binv
        d2 = binv
        rows["g1"][t] = g1
        rows["g2"][t] = g2
        rows["d1"][t] = d1
        rows["d2"][t] = d2

        # xmd for the next inner index, over the two slots that carry it.
        rows["m1"][t] = (1.0 - binv_next) * d1 + binv_next * g2
        rows["m2"][t] = (1.0 - binv_next) * d2 + binv_next * g1
    return rows


def solve(
    problem: RapdhgProblem,
    q: np.ndarray,
    iterations: int,
    tables: dict[str, np.ndarray] | None = None,
    residual: bool = False,
    h: np.ndarray | None = None,
) -> RapdhgResult:
    """Run ``iterations`` rAPDHG steps as the flat six-slot recurrence.

    The initial iterate is the origin in every slot, matching the cold start the
    generated solver performs on every parameter update.  ``xmd`` starts at zero
    consistently: at ``k = 0`` it equals ``x``, which is also zero.  ``h`` is
    the right-hand side at this instant (``problem.rhs(state)``); it defaults
    to the problem's hover value.
    """
    n, m = problem.n, problem.m
    Q = problem.P
    h = problem.h if h is None else h
    apply_g, apply_gt = problem.apply_g, problem.apply_gt
    if tables is None:
        tables = schedule_tables(problem, iterations)

    x = np.zeros(n)
    y = np.zeros(m)
    xbar = np.zeros(n)
    ybar = np.zeros(m)
    x_prev = np.zeros(n)
    xmd = np.zeros(n)

    for t in range(iterations):
        taux = tables["taux"][t]
        tth = tables["tth"][t]
        tau = tables["tau"][t]
        eta = tables["eta"][t]
        g1, g2 = tables["g1"][t], tables["g2"][t]
        d1, d2 = tables["d1"][t], tables["d2"][t]
        m1, m2 = tables["m1"][t], tables["m2"][t]

        y_new = np.maximum(y + taux * apply_g(x) - tth * apply_g(x_prev) - tau * h, 0.0)
        x_new = x - eta * (Q @ xmd) - eta * q - eta * apply_gt(y_new)

        x, y, xbar, ybar, x_prev, xmd = (
            g1 * x_new + g2 * xbar,
            g1 * y_new + g2 * ybar,
            d1 * xbar + d2 * x_new,
            d1 * ybar + d2 * y_new,
            x,
            m1 * xbar + m2 * x_new,
        )

    return RapdhgResult(
        x=x,
        y=y,
        xbar=xbar,
        ybar=ybar,
        x_prev=x_prev,
        xmd=xmd,
        iterations=iterations,
        residual=relkkt(problem, q, x, y, h=h) if residual else None,
    )


def solve_as_written(
    problem: RapdhgProblem, q: np.ndarray, iterations: int, h: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 transcribed from the paper, as a nested loop.

    Returns the iterate the paper calls ``x^{n,k}`` after ``iterations`` inner
    steps, together with its dual.  Used only to check :func:`solve`; it is the
    reference for what the flat recurrence is supposed to be.
    """
    n, m = problem.n, problem.m
    Q = problem.P
    h = problem.h if h is None else h
    K = problem.restart

    x0 = np.zeros(n)
    y0 = np.zeros(m)
    done = 0
    xbar, ybar = x0, y0
    while done < iterations:
        xbar, ybar = x0.copy(), y0.copy()
        x, y, x_prev = x0.copy(), y0.copy(), x0.copy()
        for k in range(K):
            binv, theta, eta, tau = coefficients(problem, k)
            xmd = (1.0 - binv) * xbar + binv * x
            y_new = np.maximum(y + tau * (problem.apply_g(theta * (x - x_prev) + x) - h), 0.0)
            x_new = x - eta * (Q @ xmd + q + problem.apply_gt(y_new))
            xbar = (1.0 - binv) * xbar + binv * x_new
            ybar = (1.0 - binv) * ybar + binv * y_new
            x_prev, x, y = x, x_new, y_new
            done += 1
            if done >= iterations:
                # A restart lands on the running average; anywhere else the
                # current iterate is what the next step continues from.
                return (xbar, ybar) if k == K - 1 else (x, y)
        x0, y0 = xbar, ybar
    return xbar, ybar


def relkkt(
    problem: RapdhgProblem,
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray | None = None,
) -> float:
    """The relative KKT error of Section 5 of the paper.

    The maximum of the relative primal residual, dual residual, and duality gap.
    Unlike a fixed-point residual this is scale aware and comparable across
    problems, which is what the tuning sweep needs.
    """
    Qx = problem.P @ x
    Gx = problem.apply_g(x)
    Gty = problem.apply_gt(y)
    h = problem.h if h is None else h

    r_primal = np.abs(np.maximum(Gx - h, 0.0)).max() / (
        1.0 + max(np.abs(Gx).max(), np.abs(h).max())
    )
    r_dual = np.abs(Qx + q + Gty).max() / (
        1.0 + max(np.abs(Qx).max(), np.abs(Gty).max(), np.abs(q).max())
    )
    primal_obj = 0.5 * x @ Qx + q @ x
    dual_obj = -0.5 * x @ Qx - h @ y
    r_gap = abs(primal_obj - dual_obj) / (1.0 + max(abs(primal_obj), abs(dual_obj)))
    return float(max(r_primal, r_dual, r_gap))


def solve_exact(problem: RapdhgProblem, q: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """Solve the same program to high accuracy with Clarabel.

    Clarabel takes the one-sided form directly, so this checks the reformulation
    of ``mpc.py`` as well as the iteration.  Used only to check, never inside the
    closed loop.
    """
    import clarabel
    import scipy.sparse as sp

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1e-12
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    h = problem.h if h is None else h

    solution = clarabel.DefaultSolver(
        sp.triu(sp.csc_matrix(problem.P)).tocsc(),
        q,
        sp.csc_matrix(problem.G),
        h,
        [clarabel.NonnegativeConeT(problem.m)],
        settings,
    ).solve()
    x = np.array(solution.x)
    if x.size != problem.n or not np.all(np.isfinite(x)):
        raise RuntimeError(f"Clarabel failed on the reference solve: {solution.status}")
    return x
