import importlib.util

import numpy as np
import pytest

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS


def test_nnls_sdp():
    """Test SDP solver dispatch through Verifier.solve() API."""
    vp = Verifier(method='sdp', solver='clarabel', obbt_handler=None)
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp.add_param(m, lb=0, ub=0.1)
    z0 = vp.add_iterate(n, lb=5, ub=5.1)

    z1 = vp.step.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp.step.relu(z1 - t * A.T @ (A @ z1 - x))

    vp.set_two_norm_objective(z2 - z1)

    # Test standard API
    result = vp.solve()
    assert result > TEST_EPS

    # Also verify direct solver calls still work
    cvxpy_result = vp.canonicalizer.solve_via_cvxpy()
    assert cvxpy_result > TEST_EPS

    clarabel_result = vp.canonicalizer.solve_via_clarabel()
    assert clarabel_result > TEST_EPS

    if importlib.util.find_spec('cupy') is not None:
        cuclarabel_result = vp.canonicalizer.solve_via_cuclarabel()
        assert cuclarabel_result > TEST_EPS


def test_sdp_infinity_norm_not_supported():
    """Test that infinity-norm objective raises NotImplementedError for SDP."""
    vp = Verifier(method='sdp', solver='clarabel', obbt_handler=None)
    x = vp.add_param(2, lb=0, ub=1)

    with pytest.raises(NotImplementedError, match="Infinity-norm objective is not supported"):
        vp.set_infinity_norm_objective([x])


def test_sdp_update_bounds():
    """Test that update_bounds correctly modifies constraint matrices."""
    vp = Verifier(method='sdp', solver='clarabel', obbt_handler=None)

    x = vp.add_param(2, lb=0, ub=1)
    _z = vp.add_iterate(2, lb=0, ub=1)

    # Tighten bounds
    vp.canonicalizer.update_bounds(x, lb=np.array([0.1, 0.2]), ub=np.array([0.8, 0.9]))

    # Verify constraint matrices were updated (not appended)
    # Number of inequality constraints should not change
    expected_num_constraints = 4  # 2 for x + 2 for z
    assert len(vp.canonicalizer.Aineq_matrices) == expected_num_constraints
    assert len(vp.canonicalizer.bineq_upperbounds) == expected_num_constraints

    # Verify the bound values were updated
    # For x[0]: l=0.1, u=0.8, so b = -u*l = -0.08
    # For x[1]: l=0.2, u=0.9, so b = -u*l = -0.18
    assert abs(vp.canonicalizer.bineq_upperbounds[0] - (-0.08)) < TEST_EPS
    assert abs(vp.canonicalizer.bineq_upperbounds[1] - (-0.18)) < TEST_EPS


def test_sdp_update_bounds_not_found():
    """Test that update_bounds raises ValueError for unknown expression."""
    vp = Verifier(method='sdp', solver='clarabel', obbt_handler=None)
    _x = vp.add_param(2, lb=0, ub=1)

    # Create a different expression that wasn't added
    vp2 = Verifier(method='sdp', solver='clarabel', obbt_handler=None)
    y = vp2.add_param(2, lb=0, ub=1)

    with pytest.raises(ValueError, match="Expression not found"):
        vp.canonicalizer.update_bounds(y, lb=np.array([0.1, 0.2]), ub=np.array([0.8, 0.9]))
