import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# =============================================================================
# SDP Infinity-Norm Tests
# =============================================================================


def test_infinity_norm_sdp_simple(vp_sdp):
    """
    Test infinity-norm objective on a simple bounded variable with SDP.

    Maximizes ||x||_inf where x in [-5, 10]^3.
    Expected MIP result: 10 (max of abs bounds).
    SDP provides an upper bound >= 10.
    """
    n = 3
    x = vp_sdp.iterate(n, lb=-5, ub=10)

    vp_sdp.set_objective_norm(x, np.inf)
    result = vp_sdp.solve()

    # SDP should provide an upper bound >= the true maximum (10)
    assert result >= 10.0 - TEST_EPS, f"Expected >= 10.0, got {result}"


def test_infinity_norm_sdp_after_relu(vp_sdp):
    """
    Test infinity-norm objective on ReLU output with SDP.

    Maximizes ||relu(x)||_inf where x in [-5, 5]^3.
    Expected MIP result: 5 (relu(5) = 5).
    SDP provides an upper bound >= 5.
    """
    n = 3
    x = vp_sdp.iterate(n, lb=-5, ub=5)
    z = vp_sdp.relu(x)

    vp_sdp.set_objective_norm(z, np.inf)
    result = vp_sdp.solve()

    # SDP should provide an upper bound >= the true maximum (5)
    assert result >= 5.0 - TEST_EPS, f"Expected >= 5.0, got {result}"


def test_infinity_norm_sdp_negative_bounds(vp_sdp):
    """
    Test infinity-norm where optimal is at negative values using SDP.

    Maximizes ||x||_inf where x in [-10, -2]^2.
    Expected MIP result: 10 (|x_i| = 10 at x_i = -10).
    SDP provides an upper bound >= 10.
    """
    n = 2
    x = vp_sdp.iterate(n, lb=-10, ub=-2)

    vp_sdp.set_objective_norm(x, np.inf)
    result = vp_sdp.solve()

    # SDP should provide an upper bound >= the true maximum (10)
    assert result >= 10.0 - TEST_EPS, f"Expected >= 10.0, got {result}"


def test_infinity_norm_sdp_expression_difference(vp_sdp):
    """
    Test infinity-norm on a difference of iterates with SDP.

    Maximizes ||x - y||_inf where x, y in [0, 5]^2.
    Expected MIP result: 5 (x = [5,5], y = [0,0] or vice versa gives max diff of 5).
    SDP provides an upper bound >= 5.
    """
    n = 2
    x = vp_sdp.iterate(n, lb=0, ub=5)
    y = vp_sdp.iterate(n, lb=0, ub=5)

    vp_sdp.set_objective_norm(x - y, np.inf)
    result = vp_sdp.solve()

    # SDP should provide an upper bound >= the true maximum (5)
    assert result >= 5.0 - TEST_EPS, f"Expected >= 5.0, got {result}"


def test_infinity_norm_sdp_multiple_solves(vp_sdp):
    """
    Test that solving multiple times with SDP infinity-norm doesn't accumulate constraints.
    """
    n = 2
    x = vp_sdp.iterate(n, lb=-5, ub=5)

    vp_sdp.set_objective_norm(x, np.inf)
    result1 = vp_sdp.solve()
    result2 = vp_sdp.solve()

    assert abs(result1 - result2) < TEST_EPS, "Multiple solves should give same result"


def test_infinity_norm_sdp_vs_mip(vp, vp_sdp):
    """
    Test that SDP infinity-norm gives an upper bound on MIP infinity-norm.

    SDP is a relaxation, so SDP result >= MIP result.
    """
    n = 3

    # MIP
    x_mip = vp.iterate(n, lb=-5, ub=5)
    vp.set_objective_norm(x_mip, np.inf)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.iterate(n, lb=-5, ub=5)
    vp_sdp.set_objective_norm(x_sdp, np.inf)
    sdp_result = vp_sdp.solve()

    assert sdp_result >= mip_result - TEST_EPS, (
        f"SDP ({sdp_result}) should be >= MIP ({mip_result})"
    )


def test_infinity_norm_sdp_vs_one_norm(vp_sdp):
    """
    Test that one-norm >= infinity-norm for the same problem.

    For x in [0, 5]^3:
    - ||x||_1 = 15 at x = [5, 5, 5]
    - ||x||_inf = 5 at x = [5, 5, 5]

    One-norm should be >= infinity-norm.
    """
    sdp_solver = vp_sdp.canonicalizer.solver_name
    n = 3

    # One-norm
    vp1 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    x1 = vp1.iterate(n, lb=0, ub=5)
    vp1.set_objective_norm(x1, 1)
    one_norm_result = vp1.solve()

    # Infinity-norm
    vp2 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    x2 = vp2.iterate(n, lb=0, ub=5)
    vp2.set_objective_norm(x2, np.inf)
    inf_norm_result = vp2.solve()

    assert one_norm_result >= inf_norm_result - TEST_EPS, (
        f"One-norm ({one_norm_result}) should be >= infinity-norm ({inf_norm_result})"
    )


def test_infinity_norm_sdp_mip_after_relu(vp, vp_sdp):
    """
    Test SDP vs MIP for infinity-norm after ReLU.
    """
    n = 3

    # MIP
    x_mip = vp.iterate(n, lb=-5, ub=10)
    z_mip = vp.relu(x_mip)
    vp.set_objective_norm(z_mip, np.inf)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.iterate(n, lb=-5, ub=10)
    z_sdp = vp_sdp.relu(x_sdp)
    vp_sdp.set_objective_norm(z_sdp, np.inf)
    sdp_result = vp_sdp.solve()

    assert sdp_result >= mip_result - TEST_EPS, (
        f"SDP ({sdp_result}) should be >= MIP ({mip_result})"
    )


def test_infinity_norm_sdp_mip_expression_difference(vp, vp_sdp):
    """
    Test SDP vs MIP for infinity-norm on expression difference.
    """
    n = 2

    # MIP
    x_mip = vp.iterate(n, lb=0, ub=5)
    y_mip = vp.iterate(n, lb=0, ub=5)
    vp.set_objective_norm(x_mip - y_mip, np.inf)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.iterate(n, lb=0, ub=5)
    y_sdp = vp_sdp.iterate(n, lb=0, ub=5)
    vp_sdp.set_objective_norm(x_sdp - y_sdp, np.inf)
    sdp_result = vp_sdp.solve()

    assert sdp_result >= mip_result - TEST_EPS, (
        f"SDP ({sdp_result}) should be >= MIP ({mip_result})"
    )


def test_infinity_norm_sdp_scalar(vp_sdp):
    """
    Test infinity-norm on a scalar (1D) variable.

    For scalar x in [-3, 7], ||x||_inf = |x| should have maximum 7.
    """
    x = vp_sdp.iterate(1, lb=-3, ub=7)

    vp_sdp.set_objective_norm(x, np.inf)
    result = vp_sdp.solve()

    # SDP should provide an upper bound >= 7
    assert result >= 7.0 - TEST_EPS, f"Expected >= 7.0, got {result}"


def test_infinity_norm_sdp_symmetric_bounds(vp_sdp):
    """
    Test infinity-norm with symmetric bounds.

    Maximizes ||x||_inf where x in [-5, 5]^3.
    Expected MIP result: 5.
    """
    n = 3
    x = vp_sdp.iterate(n, lb=-5, ub=5)

    vp_sdp.set_objective_norm(x, np.inf)
    result = vp_sdp.solve()

    # SDP should provide an upper bound >= 5
    assert result >= 5.0 - TEST_EPS, f"Expected >= 5.0, got {result}"
