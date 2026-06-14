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
        Linear cost; its dimension defines ``n``. Required (no ``-rhs`` default
        makes sense for an LP cost). A scalar ``q`` is rejected (cannot size the LP).
    A, b, C, d, dual_ub, primal_bounds, nu_bounds, rhs, relax_binary_vars
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
        rhs: LinearExpression = None,
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
            rhs=rhs,
            relax_binary_vars=relax_binary_vars,
        )
        self.is_lp = True
