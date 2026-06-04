"""Sparse direct factorizations for the affine (linear-solve) step.

Code generation must **never** form a matrix inverse. Instead, each affine
operator ``M w = rhs`` is factorized once, offline, and the generated solver
applies division-free triangular solves:

* symmetric ``M`` -> **LDL^T** (no pivoting; valid for the quasidefinite KKT
  systems that arise in ADMM/OSQP and for SPD systems),
* otherwise   -> **LU** with partial pivoting.

The factors are stored sparsely (CSC) and the diagonal is stored as
**reciprocals**, so the emitted forward/backward substitution uses only
multiply-add — no division — matching the embedded target's constraints.

Indirect methods (e.g. conjugate gradient) are a natural future strategy for
large/GPU problems; see :class:`fomtools.codegen.config.SolveMethod`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as spa
from scipy.sparse.csgraph import reverse_cuthill_mckee


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
    """A factorization of a square matrix, used to solve ``M x = b``.

    Attributes
    ----------
    method:
        ``"ldlt"`` or ``"lu"``.
    dim:
        System dimension.
    perm:
        Row permutation applied to the right-hand side (``b' = b[perm]``). For
        natural-ordering LDL^T this is the identity.
    L:
        Unit lower-triangular factor (CSC).
    dinv:
        Reciprocals of the LDL^T diagonal (``ldlt`` only).
    U:
        Upper-triangular factor (CSC) (``lu`` only).
    uinv:
        Reciprocals of ``diag(U)`` (``lu`` only).
    """

    method: str
    dim: int
    perm: np.ndarray
    L: CSC
    dinv: np.ndarray | None = None
    U: CSC | None = None
    uinv: np.ndarray | None = None

    @property
    def nnz(self) -> int:
        return self.L.nnz + (self.U.nnz if self.U is not None else 0)

    # -- reference solve (mirrors the emitted C) ---------------------------
    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float).reshape(-1)
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


def factorize(M, method: str = "auto", symmetry_tol: float = 1e-9) -> Factorization:
    """Factorize a square matrix ``M`` for repeated solves — never inverted.

    Parameters
    ----------
    M:
        Square matrix (dense ndarray or scipy sparse).
    method:
        ``"auto"`` (LDL^T if symmetric else LU), ``"ldlt"``, or ``"lu"``.
    """
    Md = np.asarray(M.todense(), dtype=float) if spa.issparse(M) else np.asarray(M, dtype=float)
    n = Md.shape[0]
    if Md.shape[0] != Md.shape[1]:
        raise ValueError("factorize requires a square matrix")

    if method == "cg":
        raise NotImplementedError(
            "CG (indirect) solve is reserved for a future backend; use 'auto', 'ldlt', or 'lu'."
        )
    if method not in ("auto", "ldlt", "lu"):
        raise ValueError(f"unknown solve method {method!r}")

    is_sym = np.allclose(Md, Md.T, atol=symmetry_tol)
    chosen = method
    if method == "auto":
        chosen = "ldlt" if is_sym else "lu"

    if chosen == "ldlt":
        # Fill-reducing symmetric ordering (RCM) so the LDL^T factor stays sparse.
        pattern = spa.csr_matrix((np.abs(Md) > 1e-12).astype(int))
        perm = np.asarray(reverse_cuthill_mckee(pattern, symmetric_mode=True), dtype=int)
        Mp = Md[np.ix_(perm, perm)]
        res = _ldlt_no_pivot(Mp)
        if res is not None:
            L, d = res
            return Factorization(
                method="ldlt",
                dim=n,
                perm=perm,
                L=CSC.from_dense(L),
                dinv=1.0 / d,
            )
        # Fall back to LU if LDL^T without pivoting is not viable.
        chosen = "lu"

    # LU with partial pivoting: M = P L U.
    P, L, U = sla.lu(Md)
    perm = np.argmax(P, axis=0)  # (P^T b)_i = b[perm_i]
    return Factorization(
        method="lu",
        dim=n,
        perm=perm,
        L=CSC.from_dense(L),
        U=CSC.from_dense(U),
        uinv=1.0 / np.diag(U),
    )
