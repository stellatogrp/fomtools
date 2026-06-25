"""First-order-method builders over the canonical QP form.

A *builder* takes a :class:`~fomtools.cvxpygen.qpdata.QPData` snapshot and
records one iteration of a first-order method in a fomtools
:class:`~fomtools.verifier.Verifier`, returning a :class:`FomSpec` that the
code generator and the C bridge consume. Built-in builders:

``admm``
    ADMM (KKT solve + projection). Default; handles any QP.
``pdhg``
    PDHG / Chambolle-Pock (matrix-vector products only, no linear solve).
    Requires a zero quadratic term (LP).

Users can pass their own builder callable ``f(qp, **fom_options) -> FomSpec``
to ``generate_solver(fom=...)`` — this is the "define your FOM in fomtools"
extension point.

Canonical structure exploited throughout: the first ``n_eq`` constraint rows
are equalities (projection onto the point ``b``) and the remaining rows are
one-sided inequalities ``Ax <= u`` (projection ``w -> w - relu(w - u)``), so
parameter-dependent projection bounds need only ``relu`` steps — the embedded
``satlin`` step requires numeric bounds and is never needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as spa

from fomtools.cvxpygen.qpdata import QPData
from fomtools.linearexpression import LinearExpression, stack
from fomtools.verifier import Verifier


@dataclass(frozen=True)
class FomSpec:
    """A recorded FOM iteration plus the layout info the C bridge needs.

    ``param_copy`` lists, in ``Verifier.add_param`` order (i.e. FomParams field
    order ``p0, p1, ...``), which canonical piece feeds each runtime parameter:
    ``'q'`` (length ``n``), ``'b'`` (first ``n_eq`` entries of canonical ``l``)
    or ``'u_ineq'`` (last ``n_ineq`` entries of canonical ``u``). The primal /
    dual slices locate the QP solution inside the stacked FOM state.

    ``state`` and ``params`` are the explicit leaf lists passed to
    ``extract_ir`` so that the state layout matches ``next_state`` (extraction
    would otherwise infer the ordering from leaf-discovery order).
    """

    verifier: Verifier
    next_state: LinearExpression
    name: str
    primal_offset: int
    primal_dim: int
    dual_offset: int
    dual_dim: int
    param_copy: tuple[str, ...]
    state: tuple[LinearExpression, ...]
    params: tuple[LinearExpression, ...]


#: The canonical vector pieces a FOM consumes, in FomParams field order.
PIECES = ("q", "b", "u_ineq")


def _add_params(v: Verifier, qp: QPData) -> tuple[dict, tuple[str, ...]]:
    """Add runtime parameters for the parametric pieces; bake constants.

    Returns ``(exprs, param_copy)`` where ``exprs`` maps each piece in
    :data:`PIECES` to a ``LinearExpression`` parameter or a constant numpy
    vector, and ``param_copy`` lists the parametric ones in add order.
    """
    values = {"q": qp.q, "b": qp.b, "u_ineq": qp.u_ineq}
    exprs: dict[str, LinearExpression | np.ndarray] = {}
    param_copy: list[str] = []
    for piece in PIECES:
        if piece in qp.param_dependent:
            lb, ub = qp.bounds[piece]
            exprs[piece] = v.add_param(values[piece].size, lb=lb, ub=ub)
            param_copy.append(piece)
        else:
            exprs[piece] = values[piece]
    return exprs, tuple(param_copy)


def _const_like(template: LinearExpression, value: np.ndarray) -> LinearExpression:
    """A constant vector as a LinearExpression (zero coefficients + offset)."""
    return 0.0 * template + value


def build_admm_qp(
    qp: QPData,
    *,
    rho: float = 0.1,
    sigma: float = 1e-6,
    alpha: float = 1.6,
    rho_eq_scale: float = 1e3,
) -> FomSpec:
    """One iteration of ADMM for the canonical QP (KKT solve + projection).

    State is ``(x, z, y)`` (primal, auxiliary, dual). The KKT matrix
    ``[[P + sigma I, A'], [A, -diag(1/rho_vec)]]`` is constant, so its
    factorization is computed at code-generation time; ``rho`` is boosted by
    ``rho_eq_scale`` on equality rows (the parameter defaults follow OSQP).
    The dual iterate ``y`` is exactly the dual of the canonical form, so
    CVXPY's duals are recovered from it directly.
    """
    n, m, n_eq = qp.n, qp.m, qp.n_eq
    if m == 0:
        raise ValueError("ADMM builder needs at least one constraint row")

    rho_vec = np.full(m, float(rho))
    rho_vec[:n_eq] *= float(rho_eq_scale)
    rho_inv = 1.0 / rho_vec
    Rinv = spa.diags(rho_inv)
    R = spa.diags(rho_vec)
    kkt = spa.bmat([[qp.P + sigma * spa.eye(n), qp.A.T], [qp.A, -Rinv]], format="csc")

    v = Verifier(method="simulator", obbt_solver=None)
    exprs, param_copy = _add_params(v, qp)
    q_e, b_e, u_e = exprs["q"], exprs["b"], exprs["u_ineq"]

    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    z = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
    y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

    rhs = stack([sigma * x - q_e, z - Rinv @ y])
    sol = v.step.implicit_linear(rhs, output_mat=kkt)
    x_tilde = sol[:n]
    nu = sol[n : n + m]

    z_tilde = z + Rinv @ (nu - y)
    proj_arg = alpha * z_tilde + (1.0 - alpha) * z + Rinv @ y

    # Projection onto the constraint set, row group by row group: equality rows
    # project onto the point b; inequality rows onto (-inf, u].
    parts: list[LinearExpression] = []
    if n_eq > 0:
        b_expr = b_e if isinstance(b_e, LinearExpression) else _const_like(proj_arg[:n_eq], b_e)
        parts.append(b_expr)
    if m - n_eq > 0:
        w = proj_arg[n_eq:m]
        parts.append(w - v.step.relu(w - u_e))
    z_next = stack(parts)

    x_next = alpha * x_tilde + (1.0 - alpha) * x
    y_next = y + R @ (alpha * z_tilde + (1.0 - alpha) * z - z_next)

    next_state = stack([x_next, z_next, y_next])
    return FomSpec(
        verifier=v,
        next_state=next_state,
        name="cvxpygen_admm",
        primal_offset=0,
        primal_dim=n,
        dual_offset=n + m,
        dual_dim=m,
        param_copy=param_copy,
        state=(x, z, y),
        params=tuple(exprs[piece] for piece in param_copy),
    )


def build_pdhg_lp(qp: QPData, *, step_scale: float = 0.9) -> FomSpec:
    """One iteration of PDHG (Chambolle-Pock) for the canonical LP.

    Requires ``P == 0``. Pure first-order: matrix-vector products and a relu
    projection, no linear-system solve — suited to large or factorization-free
    embedded targets. Step sizes ``tau = sigma_step = step_scale / ||A||_2``
    are computed offline.
    """
    if qp.P.nnz != 0:
        raise ValueError(
            "the PDHG builder requires a zero quadratic term (LP); use fom='admm' for QPs"
        )
    n, m, n_eq = qp.n, qp.m, qp.n_eq
    if m == 0:
        raise ValueError("PDHG builder needs at least one constraint row")

    norm_A = float(spa.linalg.norm(qp.A, 2)) if min(qp.A.shape) > 1 else float(abs(qp.A).max())
    if norm_A <= 0:
        raise ValueError("canonical A has zero norm")
    tau = sig = step_scale / norm_A

    v = Verifier(method="simulator", obbt_solver=None)
    exprs, param_copy = _add_params(v, qp)
    q_e, b_e, u_e = exprs["q"], exprs["b"], exprs["u_ineq"]

    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

    # Primal step: the prox of a linear objective is a shift.
    x_next = x - tau * (q_e + qp.A.T @ y)

    # Dual step via Moreau: y+ = v - sig * proj_[l,u](v / sig) with
    # v = y + sig * A (2 x+ - x), split into equality / inequality rows.
    v_arg = y + sig * (qp.A @ (2.0 * x_next - x))
    parts: list[LinearExpression] = []
    if n_eq > 0:
        parts.append(v_arg[:n_eq] - sig * b_e)
    if m - n_eq > 0:
        v_in = v_arg[n_eq:m]
        # proj onto (-inf, u]: w - relu(w - u)  =>  y+ = sig * relu(v/sig - u).
        parts.append(sig * v.step.relu((1.0 / sig) * v_in - u_e))
    y_next = stack(parts)

    next_state = stack([x_next, y_next])
    return FomSpec(
        verifier=v,
        next_state=next_state,
        name="cvxpygen_pdhg",
        primal_offset=0,
        primal_dim=n,
        dual_offset=n,
        dual_dim=m,
        param_copy=param_copy,
        state=(x, y),
        params=tuple(exprs[piece] for piece in param_copy),
    )


FomBuilder = Callable[..., FomSpec]

BUILDERS: dict[str, FomBuilder] = {
    "admm": build_admm_qp,
    "pdhg": build_pdhg_lp,
}


def resolve_builder(fom: str | FomBuilder) -> FomBuilder:
    """Resolve the ``fom=`` argument: a built-in name or a user callable."""
    if callable(fom):
        return fom
    try:
        return BUILDERS[fom]
    except KeyError:
        raise ValueError(
            f"unknown FOM {fom!r}; built-in choices: {sorted(BUILDERS)} "
            "(or pass a callable qp -> FomSpec)"
        ) from None
