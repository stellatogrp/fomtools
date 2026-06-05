"""Sparse direct factorizations for the affine (linear-solve) step.

Code generation must **never** form a matrix inverse. Instead, each affine
operator ``M w = rhs`` is factorized once, offline, and the generated solver
applies division-free triangular solves:

* symmetric ``M`` -> **LDL^T** (no pivoting) — but *only* when it factorizes
  accurately. This is guaranteed for SPD and **quasidefinite** systems (the
  ``[[H, A^T], [A, -G]]`` KKT systems with ``H, G`` positive-definite that arise
  in ADMM/OSQP), which are strongly factorizable. A general symmetric-indefinite
  matrix that is *not* quasidefinite can hit a zero pivot (detected) or a tiny
  pivot that silently destroys accuracy (caught by a backward-error check); in
  either case ``auto`` falls back to pivoted LU.
* otherwise   -> **LU** with partial pivoting (backward stable for any
  nonsingular matrix; a singular system is rejected with a clear error).

The (direct) factors are stored sparsely (CSC) and the diagonal is stored as
**reciprocals**, so the emitted forward/backward substitution uses only
multiply-add — no division — matching the embedded target's constraints.

For large-scale / GPU problems an *indirect* strategy is also available:
**preconditioned conjugate gradient** (``method="cg"``). It never factorizes and
never inverts — it stores the (symmetric positive-definite) system matrix
sparsely (CSC) plus a diagonal Jacobi preconditioner (held as reciprocals,
computed once offline) and reaches the solution through repeated sparse
matrix-vector products. CG needs a few scalar divisions per iteration, so it is
reserved for the large-scale targets (the embedded target stays division-free);
see :class:`fomtools.codegen.config.SolveMethod`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as spa
from scipy.sparse.csgraph import reverse_cuthill_mckee

#: Relative backward-error threshold above which a no-pivot LDL^T factorization is
#: rejected (and a pivoted LU used instead). SPD / quasidefinite systems factor to
#: ~machine precision; a much larger residual signals element growth from a small
#: pivot on a general symmetric-indefinite matrix.
_LDLT_RTOL = 1e-9


@dataclass(frozen=True)
class CSC:
    """A sparse matrix in compressed-sparse-column form."""

    n_rows: int
    n_cols: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    @classmethod
    def from_dense(cls, A: np.ndarray, tol: float = 1e-12) -> "CSC":
        A = np.asarray(A, dtype=float)
        sp = spa.csc_matrix(np.where(np.abs(A) < tol, 0.0, A))
        sp.eliminate_zeros()
        return cls(A.shape[0], A.shape[1], sp.indptr.copy(), sp.indices.copy(), sp.data.copy())

    def to_dense(self) -> np.ndarray:
        return spa.csc_matrix(
            (self.data, self.indices, self.indptr), shape=(self.n_rows, self.n_cols)
        ).toarray()

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])


@dataclass(frozen=True)
class Factorization:
    """A reusable solver for ``M x = b`` — direct (LDL^T / LU) or indirect (CG).

    A matrix inverse is never formed in any branch.

    Attributes
    ----------
    method:
        ``"ldlt"``, ``"lu"`` (direct factorizations) or ``"cg"`` (preconditioned
        conjugate gradient, indirect).
    dim:
        System dimension.
    perm:
        Row permutation applied to the right-hand side (``b' = b[perm]``). For
        natural-ordering LDL^T and for CG this is the identity.
    L:
        Unit lower-triangular factor (CSC) — direct methods only.
    dinv:
        Reciprocals of the LDL^T diagonal (``ldlt`` only).
    U:
        Upper-triangular factor (CSC) (``lu`` only).
    uinv:
        Reciprocals of ``diag(U)`` (``lu`` only).
    A:
        The symmetric positive-definite system matrix (CSC) — ``cg`` only;
        applied as repeated sparse matrix-vector products, never factorized.
    minv:
        Diagonal Jacobi preconditioner held as reciprocals of ``diag(M)``
        (computed once offline) — ``cg`` only. A diagonal scaling, not a matrix
        inverse.
    cg_tol:
        Relative residual tolerance for the CG stopping test (``cg`` only).
    cg_max_iter:
        Hard iteration cap for CG (``cg`` only).
    """

    method: str
    dim: int
    perm: np.ndarray
    L: CSC | None = None
    dinv: np.ndarray | None = None
    U: CSC | None = None
    uinv: np.ndarray | None = None
    A: CSC | None = None
    minv: np.ndarray | None = None
    cg_tol: float = 1e-12
    cg_max_iter: int = 0

    @property
    def nnz(self) -> int:
        if self.method == "cg":
            return self.A.nnz
        return self.L.nnz + (self.U.nnz if self.U is not None else 0)

    # -- reference solve (mirrors the emitted C) ---------------------------
    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float).reshape(-1)
        if self.method == "cg":
            return _pcg_solve(self.A.to_dense(), self.minv, b, self.cg_tol, self.cg_max_iter)
        rhs = b[self.perm]  # gather (symmetric perm for LDL^T, row perm for LU)
        Ld = self.L.to_dense()
        y = sla.solve_triangular(Ld, rhs, lower=True, unit_diagonal=True)
        if self.method == "ldlt":
            w = y * self.dinv
            sol = sla.solve_triangular(Ld.T, w, lower=False, unit_diagonal=True)
            x = np.empty_like(sol)
            x[self.perm] = sol  # symmetric scatter
            return x
        # lu: M = P L U  ->  x = U \ (L \ (P^T b)); column order preserved.
        Ud = self.U.to_dense()
        return sla.solve_triangular(Ud, y, lower=False, unit_diagonal=False)


def _pcg_solve(
    A: np.ndarray, minv: np.ndarray, b: np.ndarray, tol: float, max_iter: int
) -> np.ndarray:
    """Jacobi-preconditioned conjugate gradient for SPD ``A x = b``.

    Pure-NumPy reference that mirrors the emitted C **exactly** (same iteration,
    same stopping test ``r·r <= tol^2 (b·b)``) so the two agree to well within
    the verify tolerance. Starts from ``x = 0`` (hence the initial residual is
    ``b``). Never forms an inverse — only matrix-vector products and a diagonal
    scaling by the precomputed reciprocals ``minv``.
    """
    n = b.shape[0]
    x = np.zeros(n)
    bb = float(b @ b)
    if bb == 0.0:
        return x
    thresh = tol * tol * bb
    r = b.copy()
    z = minv * r
    p = z.copy()
    rz = float(r @ z)
    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp == 0.0:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr = float(r @ r)
        if rr <= thresh:
            break
        z = minv * r
        rz_new = float(r @ z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x


def _ldlt_no_pivot(M: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Up-looking LDL^T without pivoting. Returns (L unit-lower, d) or None if a
    (near-)zero pivot is hit (matrix not factorizable without pivoting)."""
    n = M.shape[0]
    L = np.eye(n)
    d = np.zeros(n)
    for j in range(n):
        dj = M[j, j] - (L[j, :j] ** 2 * d[:j]).sum()
        if abs(dj) < 1e-12:
            return None
        d[j] = dj
        if j + 1 < n:
            # L[i,j] = (M[i,j] - sum_{k<j} L[i,k] L[j,k] d_k) / dj
            acc = M[j + 1 :, j] - (L[j + 1 :, :j] * (L[j, :j] * d[:j])).sum(axis=1)
            L[j + 1 :, j] = acc / dj
    return L, d


def _ldlt_backward_error(L: np.ndarray, d: np.ndarray, M: np.ndarray) -> float:
    """Relative backward error ``||L diag(d) L^T - M|| / max(1, ||M||)``.

    No-pivot LDL^T is unconditionally stable only for SPD and *quasidefinite*
    matrices. For a general symmetric-indefinite matrix a small (but non-zero)
    pivot causes element growth that silently destroys accuracy; this measures
    it so the caller can reject the factorization and use a pivoted method.
    """
    if not (np.all(np.isfinite(L)) and np.all(np.isfinite(d))):
        return np.inf
    recon = (L * d) @ L.T  # L diag(d) L^T
    return float(np.linalg.norm(recon - M) / max(1.0, np.linalg.norm(M)))


def factorize(M, method: str = "auto", symmetry_tol: float = 1e-9) -> Factorization:
    """Factorize a square matrix ``M`` for repeated solves — never inverted.

    Parameters
    ----------
    M:
        Square matrix (dense ndarray or scipy sparse).
    method:
        ``"auto"`` (LDL^T if symmetric else LU), ``"ldlt"``, ``"lu"``, or
        ``"cg"`` (preconditioned conjugate gradient; requires a symmetric
        positive-definite ``M``, intended for large-scale / GPU targets).
    """
    Md = np.asarray(M.todense(), dtype=float) if spa.issparse(M) else np.asarray(M, dtype=float)
    n = Md.shape[0]
    if Md.shape[0] != Md.shape[1]:
        raise ValueError("factorize requires a square matrix")

    if method not in ("auto", "ldlt", "lu", "cg"):
        raise ValueError(f"unknown solve method {method!r}")

    is_sym = np.allclose(Md, Md.T, atol=symmetry_tol)

    if method == "cg":
        return _cg_factorization(Md, n, is_sym)

    chosen = method
    if method == "auto":
        chosen = "ldlt" if is_sym else "lu"

    if chosen == "ldlt":
        # Fill-reducing symmetric ordering (RCM) so the LDL^T factor stays sparse.
        pattern = spa.csr_matrix((np.abs(Md) > 1e-12).astype(int))
        perm = np.asarray(reverse_cuthill_mckee(pattern, symmetric_mode=True), dtype=int)
        Mp = Md[np.ix_(perm, perm)]
        res = _ldlt_no_pivot(Mp)
        # Keep LDL^T only when it actually factorizes Mp accurately. No-pivot
        # LDL^T is guaranteed stable for SPD / quasidefinite systems but can
        # silently lose accuracy on a general symmetric-indefinite matrix (small
        # pivot -> element growth); in that case fall back to pivoted LU rather
        # than emit a factorization that solves the wrong system.
        if res is not None and _ldlt_backward_error(res[0], res[1], Mp) <= _LDLT_RTOL:
            L, d = res
            return Factorization(
                method="ldlt",
                dim=n,
                perm=perm,
                L=CSC.from_dense(L),
                dinv=1.0 / d,
            )
        if method == "ldlt":
            # User forced LDL^T but it is not numerically viable here.
            raise ValueError(
                "LDL^T without pivoting is not numerically viable for this matrix "
                "(it is symmetric-indefinite but not quasidefinite). Use "
                "solve_method='auto' or 'lu' for a pivoted direct factorization."
            )
        chosen = "lu"  # auto: silently use the robust pivoted method

    # LU with partial pivoting: M = P L U (backward stable for any nonsingular M).
    P, L, U = sla.lu(Md)
    diag_u = np.diag(U)
    scale = max(1.0, float(np.max(np.abs(diag_u))) if diag_u.size else 1.0)
    if diag_u.size == 0 or np.min(np.abs(diag_u)) <= 1e-12 * scale:
        raise ValueError(
            "matrix is singular or numerically rank-deficient; cannot build a "
            "linear solve for it (no inverse is ever formed). Check the affine operator "
            "(e.g. add a small regularization term to make the system nonsingular)."
        )
    perm = np.argmax(P, axis=0)  # (P^T b)_i = b[perm_i]
    return Factorization(
        method="lu",
        dim=n,
        perm=perm,
        L=CSC.from_dense(L),
        U=CSC.from_dense(U),
        uinv=1.0 / diag_u,
    )


def _cg_factorization(Md: np.ndarray, n: int, is_sym: bool) -> Factorization:
    """Build a CG (indirect) solver for a symmetric positive-definite matrix.

    Stores the matrix sparsely for repeated matrix-vector products and a
    diagonal Jacobi preconditioner as reciprocals — never an inverse. CG only
    converges on SPD systems, so symmetry and positive-definiteness are checked
    up front with a clear error pointing at the direct fallbacks otherwise.
    """
    if not is_sym:
        raise ValueError(
            "CG requires a symmetric matrix; this system is nonsymmetric. "
            "Use solve_method='lu' (direct) instead."
        )
    try:
        np.linalg.cholesky(Md)  # cheap SPD certificate (offline only)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "CG requires a symmetric positive-definite matrix; this system is not "
            "positive-definite (e.g. a quasidefinite ADMM/OSQP KKT matrix). Use "
            "solve_method='ldlt' (direct) for symmetric-indefinite systems."
        ) from exc
    diag = np.diag(Md)
    return Factorization(
        method="cg",
        dim=n,
        perm=np.arange(n),
        A=CSC.from_dense(Md),
        minv=1.0 / diag,  # diagonal Jacobi preconditioner (reciprocals)
        cg_tol=1e-12,
        # SPD CG converges in <= n steps in exact arithmetic; allow headroom for
        # finite precision while the relative-residual test stops it much earlier.
        cg_max_iter=2 * n + 50,
    )
