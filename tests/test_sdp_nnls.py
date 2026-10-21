import numpy as np
import pytest

import fomtools as ft
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS


def test_nnls_sdp(vp_sdp):
    """Test SDP solver dispatch through Verifier.solve() API."""
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp_sdp.parameter(m, lb=0, ub=0.1)
    z0 = vp_sdp.iterate(n, lb=5, ub=5.1)

    z1 = vp_sdp.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp_sdp.relu(z1 - t * A.T @ (A @ z1 - x))

    vp_sdp.maximize(ft.sum_squares(z2 - z1))

    result = vp_sdp.solve()
    assert result > TEST_EPS


def test_sdp_direct_solver_calls():
    """Test direct solver method calls on canonicalizer."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver="highs")
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp.parameter(m, lb=0, ub=0.1)
    z0 = vp.iterate(n, lb=5, ub=5.1)

    z1 = vp.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp.relu(z1 - t * A.T @ (A @ z1 - x))

    vp.maximize(ft.sum_squares(z2 - z1))

    # Verify direct solver calls still work
    cvxpy_result = vp.canonicalizer.solve_via_cvxpy()
    assert cvxpy_result > TEST_EPS

    clarabel_result = vp.canonicalizer.solve_via_clarabel()
    assert clarabel_result > TEST_EPS

    try:
        cuclarabel_result = vp.canonicalizer.solve_via_cuclarabel()
        assert cuclarabel_result > TEST_EPS
    except ValueError as e:
        if "CUDA support" in str(e):
            pytest.skip("CUDA not available")
        raise


def test_sdp_infinity_norm_supported(vp_sdp):
    """Test that infinity-norm objective is now supported for SDP."""
    x = vp_sdp.parameter(2, lb=0, ub=1)

    # Should not raise - infinity-norm is now implemented
    vp_sdp.maximize(ft.norm([x], np.inf))
    result = vp_sdp.solve()

    # SDP provides an upper bound, should be >= 1 (the max abs bound)
    assert result >= 1.0 - TEST_EPS, f"Expected >= 1.0, got {result}"


def test_sdp_update_bounds(vp_sdp):
    """Test that update_bounds correctly modifies constraint matrices."""
    x = vp_sdp.parameter(2, lb=0, ub=1)
    _z = vp_sdp.iterate(2, lb=0, ub=1)

    # Tighten bounds
    vp_sdp.canonicalizer.update_bounds(x, lb=np.array([0.1, 0.2]), ub=np.array([0.8, 0.9]))

    # Verify constraint matrices were updated (not appended)
    # Number of inequality constraints should not change
    expected_num_constraints = 4  # 2 for x + 2 for z
    assert len(vp_sdp.canonicalizer.Aineq_matrices) == expected_num_constraints
    assert len(vp_sdp.canonicalizer.bineq_upperbounds) == expected_num_constraints

    # Verify the bound values were updated
    # For x[0]: l=0.1, u=0.8, so b = -u*l = -0.08
    # For x[1]: l=0.2, u=0.9, so b = -u*l = -0.18
    assert abs(vp_sdp.canonicalizer.bineq_upperbounds[0] - (-0.08)) < TEST_EPS
    assert abs(vp_sdp.canonicalizer.bineq_upperbounds[1] - (-0.18)) < TEST_EPS


def test_sdp_update_bounds_not_found(vp_sdp):
    """Test that update_bounds raises ValueError for unknown expression."""
    _x = vp_sdp.parameter(2, lb=0, ub=1)

    # Create a different expression that wasn't added
    vp2 = Verifier(method="sdp", solver="clarabel", obbt_solver="highs")
    y = vp2.parameter(2, lb=0, ub=1)

    with pytest.raises(ValueError, match="Expression not found"):
        vp_sdp.canonicalizer.update_bounds(y, lb=np.array([0.1, 0.2]), ub=np.array([0.8, 0.9]))
