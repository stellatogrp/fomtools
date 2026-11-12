"""Tests for canonical QP data extraction from cvxpygen's Canon object."""

import warnings

import numpy as np
import pytest
import scipy.sparse as spa

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.cvxpygen.qpdata import (  # noqa: E402
    extract_qp_data,
    full_symmetric_P,
    is_upper_triangular,
)

from .conftest import cvxpygen_canonicalize  # noqa: E402


def _box_qp(n=3, m=2, seed=0):
    """QP with parametric linear cost and equality rhs, constant box rows."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    q = cp.Parameter(n, name="q")
    beq = cp.Parameter(m, name="beq")
    x = cp.Variable(n, name="x")
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
        [A @ x == beq, x <= 1.0, -1.0 <= x],
    )
    q.value = np.zeros(n)
    beq.value = np.zeros(m)
    return problem, q, beq, A


def test_extract_structure_and_values():
    problem, q, beq, A = _box_qp()
    canon, _ = cvxpygen_canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={q: (-1.0, 1.0), beq: (-0.5, 0.5)})

    assert qp.n == 3
    assert qp.n_eq == 2
    assert qp.n_ineq == 6  # two box constraints on 3 variables
    assert qp.m == 8
    np.testing.assert_allclose(qp.P.toarray(), np.eye(3))
    np.testing.assert_allclose(qp.A.toarray()[: qp.n_eq], A)
    np.testing.assert_allclose(qp.u_ineq, np.ones(6))
    assert qp.param_dependent == {"q", "b"}
    assert set(qp.bounds) == {"q", "b"}

    lo, hi = qp.bounds["q"]
    np.testing.assert_allclose(lo, -np.ones(3))
    np.testing.assert_allclose(hi, np.ones(3))
    lo, hi = qp.bounds["b"]
    np.testing.assert_allclose(lo, -0.5 * np.ones(2))
    np.testing.assert_allclose(hi, 0.5 * np.ones(2))


def test_extract_bounds_keyed_by_name_and_scalar_broadcast():
    problem, q, beq, _ = _box_qp()
    canon, _ = cvxpygen_canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={"q": (-2.0, 2.0), "beq": (-1.0, 1.0)})
    lo, hi = qp.bounds["q"]
    np.testing.assert_allclose(lo, -2 * np.ones(3))
    np.testing.assert_allclose(hi, 2 * np.ones(3))


def test_extract_missing_bounds_records_fallback_silently():
    # extract_qp_data records which params fell back to default bounds but does
    # not warn itself; generate_solver emits the user-facing warning so it
    # points at the user's call site rather than these internals.
    problem, q, beq, _ = _box_qp()
    q.value = 0.25 * np.ones(3)
    canon, _ = cvxpygen_canonicalize(problem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qp = extract_qp_data(canon, param_bounds={beq: (-0.5, 0.5)}, bound_margin=0.5)
    assert qp.unbounded_params == ("q",)
    lo, hi = qp.bounds["q"]
    np.testing.assert_allclose(lo, 0.25 - 0.5)
    np.testing.assert_allclose(hi, 0.25 + 0.5)


def test_extract_unknown_bound_key_raises():
    problem, q, beq, _ = _box_qp()
    canon, _ = cvxpygen_canonicalize(problem)
    with pytest.raises(ValueError, match="unknown parameter"):
        extract_qp_data(canon, param_bounds={q: (-1, 1), beq: (-1, 1), "ghost": (-1, 1)})


def test_extract_parametric_matrix_rejected():
    n = 3
    a_par = cp.Parameter(n, name="a_par")
    a_par.value = np.ones(n)
    x = cp.Variable(n)
    # Parameter multiplies the variable -> enters canonical A.
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [a_par @ x <= 1.0, x >= -1.0])
    canon, _ = cvxpygen_canonicalize(problem)
    with pytest.raises(NotImplementedError, match="canonical matrix 'A'"):
        extract_qp_data(canon, param_bounds={a_par: (0.5, 1.5)})


def test_extract_parametric_inequality_rhs():
    n = 2
    rng = np.random.default_rng(3)
    G = rng.standard_normal((3, n))
    h = cp.Parameter(3, name="h")
    h.value = np.ones(3)
    x = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1.0)), [G @ x <= h])
    canon, _ = cvxpygen_canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={h: (0.5, 2.0)})

    # cvxpy may introduce auxiliary variables with equality rows; the parametric
    # piece is the inequality right-hand side only.
    assert qp.n_ineq == 3
    assert qp.param_dependent == {"u_ineq"}
    lo, hi = qp.bounds["u_ineq"]
    np.testing.assert_allclose(lo, 0.5 * np.ones(3))
    np.testing.assert_allclose(hi, 2.0 * np.ones(3))


def _dense_P_qp(n=4, seed=0):
    """QP with a dense quadratic cost and a parametric linear cost."""
    W = np.random.default_rng(seed).standard_normal((n, n))
    P = W @ W.T + np.eye(n)
    q = cp.Parameter(n, name="q")
    x = cp.Variable(n, name="x")
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [cp.sum(x) == 1, x >= -1])
    q.value = np.zeros(n)
    return problem, q, P


def test_extract_dense_P_is_full_symmetric():
    # Holds for either cvxpygen convention: the full P (cvxpygen 1.0) or its
    # upper triangle (cvxpygen on CVXPY's canonical maps, cvxpy/cvxpygen#133).
    problem, q, P = _dense_P_qp()
    canon, _ = cvxpygen_canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={q: (-1.0, 1.0)})
    np.testing.assert_allclose(qp.P.toarray(), P, atol=1e-12)


def test_extract_either_P_convention_gives_the_same_qp():
    # Swap the stored P for the other convention than the installed cvxpygen
    # produces (full <-> upper triangle); the extracted QP must not change.
    problem, q, P = _dense_P_qp()
    canon, _ = cvxpygen_canonicalize(problem)
    for stored in (spa.csc_matrix(P), spa.csc_matrix(np.triu(P))):
        canon.parameter_canon.p["P"] = stored
        qp = extract_qp_data(canon, param_bounds={q: (-1.0, 1.0)})
        np.testing.assert_allclose(qp.P.toarray(), P, atol=1e-12)


def test_full_symmetric_P_conventions():
    P = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, -1.0], [0.0, -1.0, 4.0]])
    for stored in (P, np.triu(P)):
        np.testing.assert_array_equal(full_symmetric_P(spa.csc_matrix(stored)).toarray(), P)
    D = np.diag(np.diag(P))
    np.testing.assert_array_equal(full_symmetric_P(spa.csc_matrix(D)).toarray(), D)
    assert is_upper_triangular(spa.csc_matrix(np.triu(P)))
    assert is_upper_triangular(spa.csc_matrix(D))
    assert not is_upper_triangular(spa.csc_matrix(P))
