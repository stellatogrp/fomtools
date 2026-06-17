import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.steps.quad_prog import QuadraticProgram, _is_linexpr


class LinearProgram(QuadraticProgram):
    r"""
    Step representing the solution of a linear program

    $$
        x^\star = \arg\min_x\; q^\top x
        \quad\text{s.t.}\quad A x = b,\; C x \le d.
    $$

    A :class:`QuadraticProgram` specialization with ``P = 0`` (stored as a sparse
    zero matrix). The KKT system drops the quadratic term: stationarity becomes
    the dual-only equality :math:`q + A^\top \nu + C^\top \lambda = 0`, so the
    primal ``x`` is pinned by primal feasibility + complementarity rather than by
    stationarity. Because stationarity no longer bounds ``x``, finite primal
    bounds must come from ``primal_bounds`` or OBBT over ``{Ax = b, Cx <= d}``.

    Parameters
    ----------
    q : np.ndarray or LinearExpression
        Linear cost; its dimension defines ``n``. Required (no ``-arg`` default
        makes sense for an LP cost). A scalar ``q`` is rejected (cannot size the LP).
    A, b, C, d, dual_ub, primal_bounds, nu_bounds, arg, relax_binary_vars
        See :class:`QuadraticProgram`.
    """

    def __init__(
        self,
        q,
        A=None,
        b=None,
        C=None,
        d=None,
        dual_ub=None,
        primal_bounds=None,
        nu_bounds=None,
        arg: LinearExpression = None,
        relax_binary_vars: bool = False,
    ):
        if _is_linexpr(q):
            n = q.ndim
        else:
            q_arr = np.asarray(q, dtype=float)
            if q_arr.ndim == 0:
                raise ValueError("LinearProgram requires a vector q (a scalar cannot size the LP)")
            n = q_arr.shape[0]

        P = spa.csr_matrix((n, n))  # P = 0
        super().__init__(
            P=P,
            q=q,
            A=A,
            b=b,
            C=C,
            d=d,
            dual_ub=dual_ub,
            primal_bounds=primal_bounds,
            nu_bounds=nu_bounds,
            arg=arg,
            relax_binary_vars=relax_binary_vars,
        )
        self.is_lp = True


class StandardFormLinearProgram(LinearProgram):
    r"""
    Step representing the solution of a **standard-form** linear program

    $$
        x^\star = \arg\min_x\; c^\top x
        \quad\text{s.t.}\quad A x = b,\; x \ge 0.
    $$

    A :class:`LinearProgram` with ``q = c``, equality ``(A, b)``, and the non-negativity
    constraint ``x >= 0`` written as the inequality ``-I x <= 0`` (``C = -I``, ``d = 0``).
    No new lowering logic — it reuses the entire ``LinearProgram`` / ``_qp_kkt`` machinery
    (the step decomposer resolves it to the ``LinearProgram`` decomposer via the class MRO).

    Parameters
    ----------
    c : np.ndarray or LinearExpression
        Linear cost; its dimension defines ``n``. A scalar is rejected (cannot size the LP).
    A, b : optional
        Equality matrix ``(p, n)`` and right-hand side (dim ``p``).
    dual_ub, primal_bounds, nu_bounds, arg, relax_binary_vars
        See :class:`QuadraticProgram`. ``dual_ub`` is required (the ``x >= 0`` inequality is
        always present, so the decomposer needs a finite bound on its multiplier).
    """

    def __init__(
        self,
        c,
        A=None,
        b=None,
        dual_ub=None,
        primal_bounds=None,
        nu_bounds=None,
        arg: LinearExpression = None,
        relax_binary_vars: bool = False,
    ):
        if _is_linexpr(c):
            n = c.ndim
        else:
            c_arr = np.asarray(c, dtype=float)
            if c_arr.ndim == 0:
                raise ValueError(
                    "StandardFormLinearProgram requires a vector c (a scalar cannot size the LP)"
                )
            n = c_arr.shape[0]

        super().__init__(
            q=c,
            A=A,
            b=b,
            C=-np.eye(n),  # x >= 0  <=>  -I x <= 0
            d=np.zeros(n),
            dual_ub=dual_ub,
            primal_bounds=primal_bounds,
            nu_bounds=nu_bounds,
            arg=arg,
            relax_binary_vars=relax_binary_vars,
        )
