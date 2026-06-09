"""Unit tests for interval bound propagation in fomtools.cvxpygen.qpdata.

These need neither cvxpy nor cvxpygen: interval_matvec is pure numpy/scipy.
"""

import itertools

import numpy as np
import scipy.sparse as spa

from fomtools.cvxpygen.qpdata import interval_matvec


def test_interval_matvec_matches_corner_enumeration():
    rng = np.random.default_rng(0)
    rows, cols = 5, 4
    M = spa.random(rows, cols, density=0.6, random_state=0, data_rvs=rng.standard_normal)
    lb = rng.uniform(-2.0, 0.0, size=cols)
    ub = lb + rng.uniform(0.0, 3.0, size=cols)

    lo, hi = interval_matvec(M, lb, ub)

    # Affine map: extrema are attained at box corners.
    corners = np.array(
        [
            M @ np.where(mask, ub, lb)
            for mask in itertools.product([False, True], repeat=cols)
        ]
    )
    np.testing.assert_allclose(lo, corners.min(axis=0), atol=1e-12)
    np.testing.assert_allclose(hi, corners.max(axis=0), atol=1e-12)
    assert np.all(lo <= hi)


def test_interval_matvec_constant_column_convention():
    # cvxpygen affine maps carry the constant offset in the last column; the
    # caller pins it via lb[-1] == ub[-1] == 1.
    M = spa.csr_matrix(np.array([[2.0, -1.0, 5.0], [0.0, 0.0, -3.0]]))
    lb = np.array([-1.0, 0.0, 1.0])
    ub = np.array([1.0, 2.0, 1.0])
    lo, hi = interval_matvec(M, lb, ub)
    np.testing.assert_allclose(lo, [2 * -1 - 1 * 2 + 5, -3.0])
    np.testing.assert_allclose(hi, [2 * 1 - 1 * 0 + 5, -3.0])


def test_interval_matvec_degenerate_interval_is_exact():
    rng = np.random.default_rng(1)
    M = spa.csr_matrix(rng.standard_normal((3, 4)))
    theta = rng.standard_normal(4)
    lo, hi = interval_matvec(M, theta, theta)
    np.testing.assert_allclose(lo, M @ theta, atol=1e-12)
    np.testing.assert_allclose(hi, M @ theta, atol=1e-12)


def test_interval_matvec_size_mismatch_raises():
    M = spa.csr_matrix(np.ones((2, 3)))
    try:
        interval_matvec(M, np.zeros(2), np.zeros(2))
    except ValueError as e:
        assert "columns" in str(e)
    else:
        raise AssertionError("expected ValueError")
