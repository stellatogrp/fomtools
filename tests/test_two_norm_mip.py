import numpy as np
import pytest

import fomtools as ft
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS


def test_two_norm_mip_simple_gurobi():
    """Test two-norm objective with MIP using Gurobi."""
    n = 2
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None)

    x = vp.iterate(n, lb=-1, ub=1)
    vp.maximize(ft.sum_squares(x))
    result = vp.solve()

    # Maximum ||x||_2^2 when x in [-1, 1]^2 is 2 (at corners)
    assert np.isclose(result, 2.0, atol=TEST_EPS)


def test_two_norm_mip_highs_raises():
    """Test that two-norm objective with HiGHS raises an error."""
    n = 2
    vp = Verifier(method="mip", solver="highs", obbt_solver=None)

    x = vp.iterate(n, lb=-1, ub=1)
    vp.maximize(ft.sum_squares(x))

    with pytest.raises(ValueError, match="does not support MIQP"):
        vp.solve()


def test_two_norm_mip_asymmetric_bounds():
    """Test two-norm with asymmetric bounds."""
    n = 2
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None)

    x = vp.iterate(n, lb=np.array([0, -2]), ub=np.array([1, 1]))
    vp.maximize(ft.sum_squares(x))
    result = vp.solve()

    # Maximum ||x||_2^2 with x in [0,1] x [-2,1] is at (1, -2): 1 + 4 = 5
    assert np.isclose(result, 5.0, atol=TEST_EPS)


def test_two_norm_mip_after_relu():
    """Test two-norm on ReLU output."""
    n = 2
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None)

    x = vp.iterate(n, lb=-1, ub=1)
    y = vp.relu(x)
    vp.maximize(ft.sum_squares(y))
    result = vp.solve()

    # relu(x) for x in [-1,1] gives [0,1], max ||y||_2^2 at y=(1,1) is 2
    assert np.isclose(result, 2.0, atol=TEST_EPS)


def test_two_norm_mip_expression_difference():
    """Test two-norm on expression difference."""
    n = 2
    np.random.seed(42)
    A = np.random.randn(n, n)

    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None)

    x = vp.iterate(n, lb=0, ub=0.5)
    y = vp.relu(A @ x)
    vp.maximize(ft.sum_squares(y - x))
    result = vp.solve()

    assert result >= 0
    assert not np.isnan(result)


def test_two_norm_mip_vs_sdp():
    """Test that MIP two-norm result is close to SDP two-norm result."""
    n = 2
    np.random.seed(0)

    # MIP with Gurobi
    vp_mip = Verifier(method="mip", solver="gurobi", obbt_solver=None)
    x_mip = vp_mip.iterate(n, lb=-1, ub=1)
    vp_mip.maximize(ft.sum_squares(x_mip))
    result_mip = vp_mip.solve()

    # SDP
    vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_solver=None)
    x_sdp = vp_sdp.iterate(n, lb=-1, ub=1)
    vp_sdp.maximize(ft.sum_squares(x_sdp))
    result_sdp = vp_sdp.solve()

    # For simple bounded variables, MIP and SDP should give same result
    assert np.isclose(result_mip, result_sdp, atol=TEST_EPS)


def test_two_norm_mip_vs_sdp_after_relu():
    """Test MIP vs SDP two-norm after ReLU."""
    n = 2
    np.random.seed(42)

    # MIP with Gurobi
    vp_mip = Verifier(method="mip", solver="gurobi", obbt_solver=None)
    x_mip = vp_mip.iterate(n, lb=-1, ub=1)
    y_mip = vp_mip.relu(x_mip)
    vp_mip.maximize(ft.sum_squares(y_mip - x_mip))
    result_mip = vp_mip.solve()

    # SDP
    vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_solver=None)
    x_sdp = vp_sdp.iterate(n, lb=-1, ub=1)
    y_sdp = vp_sdp.relu(x_sdp)
    vp_sdp.maximize(ft.sum_squares(y_sdp - x_sdp))
    result_sdp = vp_sdp.solve()

    # MIP should be exact, SDP is relaxation
    # So MIP <= SDP
    assert result_mip <= result_sdp + TEST_EPS


def test_two_norm_mip_with_constraint():
    """Test two-norm with additional constraint."""
    n = 2
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None)

    x = vp.iterate(n, lb=-1, ub=1)
    # Constrain sum(x) <= 0
    vp.add_constraint(np.ones((1, n)) @ x <= 0)
    vp.maximize(ft.sum_squares(x))
    result = vp.solve()

    # Maximum ||x||_2^2 with sum(x) <= 0 and x in [-1,1]^2
    # Best is x = (1, -1) or (-1, 1), giving ||x||^2 = 2
    assert np.isclose(result, 2.0, atol=TEST_EPS)


def test_two_norm_mip_multiple_solves():
    """Test that multiple solves work correctly."""
    n = 2

    vp1 = Verifier(method="mip", solver="gurobi", obbt_solver=None)
    x1 = vp1.iterate(n, lb=-1, ub=1)
    vp1.maximize(ft.sum_squares(x1))
    result1 = vp1.solve()

    vp2 = Verifier(method="mip", solver="gurobi", obbt_solver=None)
    x2 = vp2.iterate(n, lb=-0.5, ub=0.5)
    vp2.maximize(ft.sum_squares(x2))
    result2 = vp2.solve()

    # Result with tighter bounds should be smaller
    assert result2 < result1


def test_two_norm_mip_linear_transform():
    """Test two-norm on linear transformation."""
    n = 2
    A = np.array([[2.0, 0.0], [0.0, 1.0]])

    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None)
    x = vp.iterate(n, lb=-1, ub=1)
    vp.maximize(ft.sum_squares(A @ x))
    result = vp.solve()

    # ||A @ x||_2^2 = ||[2x_1, x_2]||^2 = 4*x_1^2 + x_2^2
    # Max at x = (1, 1) or (-1, -1): 4 + 1 = 5
    assert np.isclose(result, 5.0, atol=TEST_EPS)
