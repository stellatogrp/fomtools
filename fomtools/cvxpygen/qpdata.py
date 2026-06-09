"""Canonical QP data extraction and parameter-bound propagation.

CVXPYgen canonicalizes a parametric CVXPY problem to the OSQP standard form

    minimize    0.5 x'Px + q'x + d
    subject to  l <= Ax <= u

where the first ``n_eq`` constraint rows are equalities (``l == u``) and the
remaining ``n_ineq`` rows are one-sided inequalities (``l == -inf``). Each
canonical quantity is an affine function of the flattened user parameters.

This module turns cvxpygen's :class:`Canon` object into a solver-agnostic
:class:`QPData` snapshot for the FOM builders, including interval bounds on the
parametric pieces (fomtools parameters need ``lb``/``ub`` for verification
sampling), propagated exactly through the affine parameter maps.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as spa

# cvxpygen's convention for clamping infinite bounds in generated C data.
CVXPYGEN_INF = 1e30


@dataclass(frozen=True)
class QPData:
    """Snapshot of a canonicalized parametric QP in OSQP standard form.

    ``q`` (linear cost), ``b`` (equality right-hand sides, the first ``n_eq``
    entries of canonical ``l``/``u``) and ``u_ineq`` (inequality upper bounds,
    the last ``n_ineq`` entries of canonical ``u``) may each be parametric;
    ``param_dependent`` lists which. ``bounds`` maps each parametric piece to
    elementwise ``(lb, ub)`` propagated from the user parameter bounds. The
    matrices ``P`` and ``A`` are always constant (enforced at extraction).
    """

    P: spa.csc_matrix
    q: np.ndarray
    A: spa.csc_matrix
    b: np.ndarray
    u_ineq: np.ndarray
    d: float
    n: int
    n_eq: int
    n_ineq: int
    param_dependent: frozenset[str]
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def m(self) -> int:
        """Total number of constraint rows."""
        return self.n_eq + self.n_ineq


def interval_matvec(
    mapping: spa.spmatrix, lb: np.ndarray, ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval propagation of ``mapping @ theta`` for ``theta in [lb, ub]``.

    ``mapping`` is one of cvxpygen's affine parameter maps: its columns
    correspond to the flattened user parameters *including* the trailing
    homogenization entry, which the caller pins via ``lb[-1] == ub[-1] == 1``.
    """
    if lb.shape != ub.shape or mapping.shape[1] != lb.size:
        raise ValueError(
            f"bound vectors of size {lb.size}/{ub.size} do not match affine map with "
            f"{mapping.shape[1]} columns"
        )
    M = spa.csr_matrix(mapping)
    pos = M.maximum(0)
    neg = M.minimum(0)
    lo = pos @ lb + neg @ ub
    hi = pos @ ub + neg @ lb
    return lo, hi


def user_bounds_vector(
    parameter_info, param_bounds: dict | None, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat lower/upper bounds over cvxpygen's user-parameter vector.

    ``param_bounds`` maps CVXPY ``Parameter`` objects (or their names) to
    ``(lb, ub)`` pairs; entries broadcast against the parameter shape and are
    flattened in Fortran order, matching cvxpygen. Parameters without supplied
    bounds fall back to their current value ``+/- margin`` with a warning. The
    trailing homogenization entry is pinned to 1.
    """
    flat = np.asarray(parameter_info.flat_usp, dtype=float)
    lb = flat - margin
    ub = flat + margin

    supplied: dict[str, tuple] = {}
    for key, value in (param_bounds or {}).items():
        name = key if isinstance(key, str) else key.name()
        supplied[name] = value

    name_to_col = {
        parameter_info.col_to_name_usp[col]: col for col in parameter_info.col_to_name_usp
    }
    for name in parameter_info.names:
        size = parameter_info.name_to_size_usp[name]
        col = name_to_col[name]
        if name in supplied:
            lo, hi = supplied.pop(name)
            shape = parameter_info.name_to_shape[name]
            lo = np.broadcast_to(np.asarray(lo, dtype=float), shape).flatten(order="F")
            hi = np.broadcast_to(np.asarray(hi, dtype=float), shape).flatten(order="F")
            if lo.size != size or hi.size != size:
                raise ValueError(
                    f"bounds for parameter {name!r} have size {lo.size}/{hi.size}, "
                    f"expected {size}"
                )
            if np.any(lo > hi):
                raise ValueError(f"bounds for parameter {name!r} have lb > ub")
            lb[col : col + size] = lo
            ub[col : col + size] = hi
        else:
            warnings.warn(
                f"no bounds supplied for CVXPY parameter {name!r}; using its current "
                f"value +/- {margin} (pass param_bounds={{{name!r}: (lb, ub)}} to "
                "control the verified parameter range)",
                stacklevel=2,
            )
    if supplied:
        raise ValueError(f"param_bounds given for unknown parameter(s): {sorted(supplied)}")

    # Trailing homogenization entry is the constant 1.
    lb[-1] = ub[-1] = 1.0
    return lb, ub


def _parametric_rows(mapping: spa.spmatrix) -> np.ndarray:
    """Boolean mask of rows that depend on user parameters (not just the constant)."""
    M = spa.csr_matrix(mapping)[:, :-1]
    return np.diff(M.indptr) > 0


def extract_qp_data(
    canon, param_bounds: dict | None = None, bound_margin: float = 1.0
) -> QPData:
    """Extract :class:`QPData` from a cvxpygen ``Canon`` (OSQP canonical form)."""
    pc = canon.parameter_canon
    pi = canon.parameter_info

    for p_id in ("P", "A"):
        if pc.p_id_to_changes.get(p_id):
            raise NotImplementedError(
                f"canonical matrix {p_id!r} depends on CVXPY parameters; fomtools bakes "
                "matrices into the generated solver at code-generation time, so only "
                "parametric vectors (q, l, u) are supported. Rebuild the model so that "
                "parameters enter only the objective's linear term and constraint "
                "right-hand sides."
            )

    P = spa.csc_matrix(pc.p["P"])
    A = spa.csc_matrix(pc.p["A"]) if pc.p["A"] is not None else spa.csc_matrix((0, P.shape[0]))
    q = np.asarray(pc.p["q"], dtype=float).ravel()
    l_full = np.asarray(pc.p["l"], dtype=float).ravel()
    u_full = np.asarray(pc.p["u"], dtype=float).ravel()
    d = float(np.asarray(pc.p.get("d", 0.0)).ravel()[0]) if pc.p.get("d") is not None else 0.0

    n = P.shape[0]
    m = A.shape[0]
    # Canonical l has size n_eq in the affine map; cvxpygen stores the full
    # augmented vector (ineq rows are -inf) in pc.p['l'].
    n_eq = int(pc.p_id_to_size["l"])
    n_ineq = m - n_eq

    if (P != P.T).nnz != 0:
        raise AssertionError("canonical P is not symmetric; cvxpygen convention changed?")
    if not np.allclose(l_full[:n_eq], u_full[:n_eq], equal_nan=True):
        raise AssertionError("canonical eq rows have l != u; cvxpygen convention changed?")

    b = u_full[:n_eq].copy()
    u_ineq = u_full[n_eq:].copy()
    if not np.all(np.isfinite(b)):
        raise AssertionError("equality right-hand sides contain infinities")
    # Match the generated C data (cvxpygen's replace_inf convention).
    u_ineq = np.clip(u_ineq, -CVXPYGEN_INF, CVXPYGEN_INF)

    # Which pieces depend on user parameters, row by row.
    param_dependent = set()
    if pc.p_id_to_changes.get("q"):
        param_dependent.add("q")
    map_u = pc.p_id_to_mapping.get("u")
    u_rows = _parametric_rows(map_u) if map_u is not None else np.zeros(m, dtype=bool)
    if pc.p_id_to_changes.get("l") or np.any(u_rows[:n_eq]):
        param_dependent.add("b")
    if np.any(u_rows[n_eq:]):
        param_dependent.add("u_ineq")

    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if param_dependent:
        user_lb, user_ub = user_bounds_vector(pi, param_bounds, bound_margin)
        if "q" in param_dependent:
            bounds["q"] = interval_matvec(pc.p_id_to_mapping["q"], user_lb, user_ub)
        if "b" in param_dependent:
            # The 'u' map covers all m rows; its first n_eq rows are the eq values.
            bounds["b"] = interval_matvec(spa.csr_matrix(map_u)[:n_eq], user_lb, user_ub)
        if "u_ineq" in param_dependent:
            lo, hi = interval_matvec(spa.csr_matrix(map_u)[n_eq:], user_lb, user_ub)
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                raise ValueError(
                    "parametric inequality upper bounds have infinite propagated bounds; "
                    "supply finite param_bounds for every parameter entering constraint "
                    "right-hand sides"
                )
            bounds["u_ineq"] = (lo, hi)
        for piece in ("q", "b"):
            if piece in bounds and not all(np.all(np.isfinite(v)) for v in bounds[piece]):
                raise ValueError(
                    f"parametric {piece!r} has infinite propagated bounds; supply finite "
                    "param_bounds for the parameters it depends on"
                )

    return QPData(
        P=P,
        q=q,
        A=A,
        b=b,
        u_ineq=u_ineq,
        d=d,
        n=n,
        n_eq=n_eq,
        n_ineq=n_ineq,
        param_dependent=frozenset(param_dependent),
        bounds=bounds,
    )
