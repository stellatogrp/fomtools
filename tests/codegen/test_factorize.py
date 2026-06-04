"""Tests for the sparse factorization core (no inverses, ever)."""

import numpy as np

from fomtools.codegen.factorize import CSC, factorize


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
