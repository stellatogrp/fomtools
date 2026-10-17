"""Ruiz equilibration of the canonical QP.

The FOM iterates on an equilibrated copy of the canonical problem

    minimize    0.5 x'Px + q'x
    subject to  l <= Ax <= u

with positive diagonal scalings ``D`` (variables), ``E`` (constraint rows) and a
cost scaling ``c > 0``, as OSQP does::

    P~ = c D P D,   q~ = c D q,   A~ = E A D,   l~ = E l,   u~ = E u,

so that the scaled problem's solution maps back as ``x = D x~`` and
``y = E y~ / c``. ``D`` and ``E`` come from Ruiz passes over the KKT matrix
``[[P, A'], [A, 0]]``, driving every row and column inf-norm toward one.

Only the vectors ``q``, ``l``, ``u`` may depend on CVXPY parameters, and a
positive diagonal scaling maps each of their elementwise boxes to a box, so the
parameter bounds handed to the verifier scale exactly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import scipy.sparse as spa

from fomtools.cvxpygen.qpdata import CVXPYGEN_INF, QPData

#: Ruiz passes (OSQP's default ``scaling = 10``).
RUIZ_ITERS = 10
#: Norms are clipped to this range before scaling by them, as in OSQP, so an
#: empty or tiny row/column does not blow the scaling up.
_NORM_MIN, _NORM_MAX = 1e-4, 1e4


@dataclass(frozen=True)
class Scaling:
    """``x = D x~``, constraint rows scaled by ``E``, cost scaled by ``c``."""

    D: np.ndarray
    E: np.ndarray
    c: float


def _col_inf_norms(M: spa.spmatrix, size: int) -> np.ndarray:
    if M.shape[0] == 0 or M.nnz == 0:
        return np.zeros(size)
    return np.asarray(abs(M).max(axis=0).todense()).ravel()


def _clip(norms: np.ndarray) -> np.ndarray:
    return np.clip(norms, _NORM_MIN, _NORM_MAX)


def ruiz_kkt(
    P: spa.spmatrix, A: spa.spmatrix, q_abs: np.ndarray | None = None, iters: int = RUIZ_ITERS
) -> Scaling:
    """Ruiz equilibration of ``[[P, A'], [A, 0]]`` plus OSQP's cost scaling.

    ``q_abs`` is an elementwise bound on ``|q|`` over the values ``q`` can take
    (the cost scaling must not depend on the parameter values, since it is baked
    into the solver); ``None`` treats ``q`` as zero. A loose bound can let
    ``max |q|`` dominate the cost scaling and shrink ``c`` (down to the 1e-4
    clip), flattening ``P``: still correct, but possibly slower to converge.
    """
    P, A = spa.csc_matrix(P), spa.csc_matrix(A)
    n, m = P.shape[0], A.shape[0]
    q_abs = np.zeros(n) if q_abs is None else np.asarray(q_abs, dtype=float)
    D, E, c = np.ones(n), np.ones(m), 1.0
    # As OSQP's scale_data: each pass equilibrates the KKT matrix of the
    # *currently* scaled problem (the cost scaling included), then rescales the
    # cost, so the two interact pass by pass. The scaled matrices are updated in
    # place by each pass's increments rather than rebuilt from P and A.
    Ps, As = P.copy(), A.copy()
    for _ in range(iters):
        # KKT column norms: variable columns see P and A, row columns see A'.
        norm_x = np.maximum(_col_inf_norms(Ps, n), _col_inf_norms(As, n))
        norm_y = _col_inf_norms(As.T, m)
        dD, dE = 1.0 / np.sqrt(_clip(norm_x)), 1.0 / np.sqrt(_clip(norm_y))
        D, E = D * dD, E * dE
        Ps = (spa.diags(dD) @ Ps @ spa.diags(dD)).tocsc()
        As = (spa.diags(dE) @ As @ spa.diags(dD)).tocsc()

        mean_p = float(np.mean(_col_inf_norms(Ps, n))) if n else 0.0
        max_q = float(np.max(c * D * q_abs)) if n else 0.0
        scale = max(mean_p, max_q)
        if scale > 0:
            dc = 1.0 / float(np.clip(scale, _NORM_MIN, _NORM_MAX))
            c, Ps = c * dc, dc * Ps
    return Scaling(D=D, E=E, c=c)


def ruiz_for(qp: QPData, iters: int = RUIZ_ITERS) -> Scaling:
    """:func:`ruiz_kkt` for a :class:`QPData`, bounding ``|q|`` over its parameter box."""
    if "q" in qp.param_dependent:
        lb, ub = qp.bounds["q"]
        q_abs = np.maximum(np.abs(lb), np.abs(ub))
    else:
        q_abs = np.abs(qp.q)
    return ruiz_kkt(qp.P, qp.A, q_abs=q_abs, iters=iters)


def _scale_finite(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``scale * values``, keeping cvxpygen's +/-1e30 infinities infinite."""
    infinite = np.abs(values) >= CVXPYGEN_INF
    return np.where(infinite, values, scale * values)


def scale_qp(qp: QPData, s: Scaling) -> QPData:
    """The equilibrated copy of ``qp`` (data, and the bounds of its parametric pieces).

    The copy records ``(D, E, c)`` in ``unscale``, so the builders measure their
    residual objective in the original problem's units, as the bridge reports it.
    """
    n_eq = qp.n_eq
    E_eq, E_in = s.E[:n_eq], s.E[n_eq:]
    piece_scale = {"q": s.c * s.D, "b": E_eq, "u_ineq": E_in}
    bounds = {
        piece: (_scale_finite(lb, piece_scale[piece]), _scale_finite(ub, piece_scale[piece]))
        for piece, (lb, ub) in qp.bounds.items()
    }
    Dm, Em = spa.diags(s.D), spa.diags(s.E)
    return dataclasses.replace(
        qp,
        P=spa.csc_matrix(s.c * (Dm @ qp.P @ Dm)),
        q=s.c * s.D * qp.q,
        A=spa.csc_matrix(Em @ qp.A @ Dm),
        b=E_eq * qp.b,
        u_ineq=_scale_finite(qp.u_ineq, E_in),
        d=s.c * qp.d,
        bounds=bounds,
        unscale=(s.D, s.E, s.c),
    )
