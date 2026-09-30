"""Shared matrix helpers for symmetry / PSD validation (dense or sparse).

These back the ``P``/``Q`` validation in both :class:`fomtools.steps.QuadraticProgram`
and :class:`fomtools.proxable_functions.Quadratic`. They live in a neutral module so
``proxable_functions`` can reuse them without importing from ``steps`` (which would
create an import edge in the wrong direction).
"""

import numpy as np
import scipy.sparse as spa

# Tolerance for the symmetry and PSD checks. PSD allows a small negative eigenvalue
# to absorb floating-point noise (e.g. ``A.T @ A`` Hessians).
_PSD_TOL = 1e-8

# Below this dimension, sparse matrices are densified for the eigenvalue check
# (``eigsh`` is fragile / cannot return all-but-one eigenvalues on small matrices).
_DENSE_EIG_MAX_N = 200


def _min_eig(P) -> float:
    """Smallest eigenvalue of a symmetric matrix ``P`` (dense or sparse)."""
    if spa.issparse(P):
        if P.nnz == 0:  # all-zero (e.g. LinearProgram); trivially PSD, skip eigsh
            return 0.0
        n = P.shape[0]
        if n <= _DENSE_EIG_MAX_N:
            return float(np.linalg.eigvalsh(P.toarray()).min())
        try:
            return float(
                spa.linalg.eigsh(P.asfptype(), k=1, which="SA", return_eigenvectors=False)[0]
            )
        except Exception:  # pragma: no cover - convergence fallback
            return float(np.linalg.eigvalsh(P.toarray()).min())
    return float(np.linalg.eigvalsh(np.asarray(P, dtype=float)).min())


def _sym_error(P) -> float:
    """Max absolute asymmetry ``max|P - P^T|`` (dense or sparse)."""
    diff = P - P.T
    if spa.issparse(diff):
        diff = diff.tocoo()
        return 0.0 if diff.nnz == 0 else float(np.abs(diff.data).max())
    return float(np.abs(np.asarray(diff, dtype=float)).max())


def is_identity_matrix(P) -> bool:
    """True if ``P`` is ``None`` or (numerically) the identity (dense or sparse).

    ``P`` is expected to be 2-D; non-square (or 1-D) inputs return ``False``.
    """
    if P is None:
        return True
    n = P.shape[0]
    if P.shape != (n, n):
        return False
    if spa.issparse(P):
        diff = P - spa.eye(n, format="csr")
        return diff.nnz == 0 or float(np.abs(diff).max()) <= 1e-12
    return np.allclose(np.asarray(P, dtype=float), np.eye(n))
