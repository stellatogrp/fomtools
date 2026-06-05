"""Tests for the sparse factorization core (no inverses, ever)."""

import numpy as np
import pytest

from fomtools.codegen.factorize import CSC, factorize


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(n, n))
    return half @ half.T + 2.0 * np.eye(n)  # symmetric, eigenvalues >= 2


def _kkt(n, m, seed):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, n))
    H = H @ H.T + np.eye(n)
    A = rng.normal(size=(m, n))
    return np.block([[H + 1e-6 * np.eye(n), A.T], [A, -0.1 * np.eye(m)]])


def test_symmetric_uses_ldlt_and_solves():
    M = _kkt(8, 5, 0)
    f = factorize(M)
    assert f.method == "ldlt"
    rng = np.random.default_rng(1)
    for _ in range(5):
        b = rng.normal(size=M.shape[0])
        assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-9)


def test_ldlt_reconstructs_permuted_matrix():
    M = _kkt(6, 4, 2)
    f = factorize(M)
    L = f.L.to_dense()
    # L D L^T reconstructs the symmetrically permuted matrix (fill-reducing order).
    Mp = M[np.ix_(f.perm, f.perm)]
    assert np.allclose(L @ np.diag(1.0 / f.dinv) @ L.T, Mp, atol=1e-8)
    # L is unit lower triangular.
    assert np.allclose(np.diag(L), 1.0)
    assert np.allclose(np.triu(L, 1), 0.0)


def test_rcm_ordering_reduces_fill():
    # On a structured KKT, the fill-reducing order yields far fewer nonzeros
    # than the dense lower triangle.
    M = _kkt(20, 12, 9)
    f = factorize(M)
    assert f.method == "ldlt"
    dense_tri = M.shape[0] * (M.shape[0] + 1) // 2
    assert f.L.nnz < dense_tri


def test_nonsymmetric_uses_lu_and_solves():
    rng = np.random.default_rng(3)
    n = 10
    M = rng.normal(size=(n, n)) + n * np.eye(n)
    M[0, 1] += 5.0  # break symmetry
    f = factorize(M)
    assert f.method == "lu"
    for _ in range(5):
        b = rng.normal(size=n)
        assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-9)


def test_no_inverse_only_triangular_factors():
    # The factorization exposes triangular factors + reciprocal diagonals,
    # never a dense inverse.
    f = factorize(_kkt(5, 3, 4))
    assert f.dinv is not None and np.all(np.isfinite(f.dinv))
    assert f.U is None  # ldlt keeps a single triangle


def test_forced_lu_on_symmetric():
    M = _kkt(5, 3, 5)
    f = factorize(M, method="lu")
    assert f.method == "lu"
    b = np.arange(M.shape[0], dtype=float)
    assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-9)


def test_csc_roundtrip():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(5, 5))
    A[np.abs(A) < 0.3] = 0.0
    csc = CSC.from_dense(A)
    assert np.allclose(csc.to_dense(), A)


def test_sparse_input_accepted():
    import scipy.sparse as spa

    M = _kkt(6, 4, 7)
    f = factorize(spa.csc_matrix(M))
    b = np.ones(M.shape[0])
    assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-9)


def test_factor_is_sparser_than_dense_for_structured_kkt():
    # A banded/structured KKT should factor with fewer nonzeros than n^2.
    M = _kkt(20, 10, 8)
    f = factorize(M)
    assert f.nnz < M.shape[0] ** 2


# -- CG (indirect) -------------------------------------------------------------
def test_cg_solves_spd():
    M = _spd(12, 0)
    f = factorize(M, method="cg")
    assert f.method == "cg"
    rng = np.random.default_rng(1)
    for _ in range(5):
        b = rng.normal(size=M.shape[0])
        # CG converges to the exact solution on an SPD system.
        assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-8)


def test_cg_stores_matrix_not_factors_or_inverse():
    # CG keeps the system matrix + a diagonal preconditioner; no L/U/inverse.
    f = factorize(_spd(7, 2), method="cg")
    assert f.A is not None and f.L is None and f.U is None and f.dinv is None
    assert f.minv is not None and np.all(np.isfinite(f.minv))
    # The Jacobi preconditioner is exactly the reciprocal diagonal of M.
    M = f.A.to_dense()
    assert np.allclose(f.minv, 1.0 / np.diag(M))
    assert f.nnz == f.A.nnz


def test_cg_zero_rhs_returns_zero():
    f = factorize(_spd(5, 3), method="cg")
    assert np.allclose(f.solve(np.zeros(5)), 0.0)


def test_cg_rejects_nonsymmetric():
    rng = np.random.default_rng(4)
    M = rng.normal(size=(6, 6)) + 6 * np.eye(6)
    M[0, 1] += 3.0  # break symmetry
    with pytest.raises(ValueError, match="symmetric"):
        factorize(M, method="cg")


def test_cg_rejects_indefinite():
    # A symmetric-indefinite (quasidefinite) KKT is not SPD -> CG must refuse.
    M = _kkt(6, 4, 5)
    with pytest.raises(ValueError, match="positive-definite"):
        factorize(M, method="cg")


def test_cg_accepts_sparse_input():
    import scipy.sparse as spa

    M = _spd(8, 6)
    f = factorize(spa.csc_matrix(M), method="cg")
    b = np.ones(8)
    assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-8)


# -- non-quasidefinite robustness (LDL^T is not always right) ------------------
def test_zero_diagonal_indefinite_falls_back_to_lu():
    # Symmetric, indefinite, nonsingular, all-zero diagonal: no-pivot LDL^T hits a
    # zero pivot immediately, so AUTO must fall back to pivoted LU.
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    f = factorize(M)
    assert f.method == "lu"
    b = np.array([2.0, 3.0])
    assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-12)


def test_h_zero_kkt_not_quasidefinite_uses_lu():
    # KKT [[0, A^T],[A, 0]] with A square nonsingular: symmetric indefinite, not
    # quasidefinite -> LU, solved accurately.
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    K = np.block([[np.zeros((4, 4)), A.T], [A, np.zeros((4, 4))]])
    f = factorize(K)
    assert f.method == "lu"
    b = np.arange(1, 9, dtype=float)
    assert np.allclose(f.solve(b), np.linalg.solve(K, b), atol=1e-9)


def test_singular_matrix_raises():
    # A singular system has no solve; factorize must reject it cleanly rather
    # than silently produce an infinite reciprocal pivot.
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 3))  # 5x5 KKT with a 1-dim null space
    S = np.block([[np.zeros((3, 3)), A.T], [A, np.zeros((2, 2))]])
    with pytest.raises(ValueError, match="singular"):
        factorize(S)


def test_unstable_no_pivot_ldlt_falls_back_via_backward_error(monkeypatch):
    # Defeat the fill-reducing reordering (which would otherwise move the bad
    # pivot away) so the no-pivot LDL^T genuinely loses accuracy; the backward-
    # error gate must then reject it and AUTO must use pivoted LU.
    import fomtools.codegen.factorize as fz

    monkeypatch.setattr(fz, "reverse_cuthill_mckee", lambda pat, **k: np.arange(pat.shape[0]))
    M = np.array([[1e-11, 1.0, 0.5], [1.0, 1.0, 2.0], [0.5, 2.0, -1.0]])
    # no-pivot LDL^T on M proceeds but is inaccurate (backward error ~ 4e-6).
    L, d = fz._ldlt_no_pivot(M)
    assert fz._ldlt_backward_error(L, d, M) > 1e-7
    f = factorize(M)  # AUTO
    assert f.method == "lu"  # gate rejected the inaccurate LDL^T
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(f.solve(b), np.linalg.solve(M, b), atol=1e-9)


def test_forced_ldlt_on_non_quasidefinite_raises():
    # Forcing LDL^T on a zero-diagonal indefinite system is not viable.
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="not numerically viable|quasidefinite"):
        factorize(M, method="ldlt")
