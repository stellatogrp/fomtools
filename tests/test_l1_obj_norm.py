import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}


def test_one_norm_simple(vp):
    """
    Test one-norm objective on a simple bounded variable.

    Maximizes ||x||_1 where x in [-5, 5]^3.
    Expected result: 15 (each |x_i| = 5).
    """
    n = 3
    x = vp.add_iterate(n, lb=-5, ub=5)

    vp.set_objective_norm(x, 1)
    result = vp.solve()

    x_val = vp.get_solution_value(x)
    manual_norm = np.sum(np.abs(x_val))

    assert abs(result - 15.0) < TEST_EPS, f"Expected 15.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_after_relu(vp):
    """
    Test one-norm objective on ReLU output.

    Maximizes ||relu(x)||_1 where x in [-5, 5]^3.
    Expected result: 15 (relu(5) = 5 for each component).
    """
    n = 3
    x = vp.add_iterate(n, lb=-5, ub=5)
    z = vp.step.relu(x)

    vp.set_objective_norm(z, 1)
    result = vp.solve()

    z_val = vp.get_solution_value(z)
    manual_norm = np.sum(np.abs(z_val))

    assert abs(result - 15.0) < TEST_EPS, f"Expected 15.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_asymmetric_bounds(vp):
    """
    Test one-norm with asymmetric bounds.

    Maximizes ||x||_1 where x in [-2, 10]^2.
    Expected result: 20 (each |x_i| = 10).
    """
    n = 2
    x = vp.add_iterate(n, lb=-2, ub=10)

    vp.set_objective_norm(x, 1)
    result = vp.solve()

    x_val = vp.get_solution_value(x)
    manual_norm = np.sum(np.abs(x_val))

    assert abs(result - 20.0) < TEST_EPS, f"Expected 20.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_negative_bounds(vp):
    """
    Test one-norm where optimal is at negative values.

    Maximizes ||x||_1 where x in [-10, -2]^2.
    Expected result: 20 (each |x_i| = 10 at x_i = -10).
    """
    n = 2
    x = vp.add_iterate(n, lb=-10, ub=-2)

    vp.set_objective_norm(x, 1)
    result = vp.solve()

    x_val = vp.get_solution_value(x)
    manual_norm = np.sum(np.abs(x_val))

    assert abs(result - 20.0) < TEST_EPS, f"Expected 20.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"
    assert np.allclose(x_val, [-10, -10], atol=TEST_EPS), "Solution should be at lower bound"


def test_one_norm_mixed_sign_optimal(vp):
    """
    Test one-norm where optimal has mixed signs.

    Maximizes ||x||_1 where x_0 in [2, 5], x_1 in [-8, -3].
    Expected: |5| + |-8| = 13.
    """
    mip_solver_name = MIP_SOLVER_NAMES[type(vp.canonicalizer.solver).__name__]

    vp_test = Verifier(method="mip", solver=mip_solver_name, obbt_solver="highs")
    x = vp_test.add_param(1, lb=2, ub=5)
    y = vp_test.add_param(1, lb=-8, ub=-3)

    vp_test.set_objective_norm([x, y], 1)
    result = vp_test.solve()

    x_val = vp_test.get_solution_value(x)
    y_val = vp_test.get_solution_value(y)
    manual_norm = np.abs(x_val[0]) + np.abs(y_val[0])

    assert abs(result - 13.0) < TEST_EPS, f"Expected 13.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_with_constraint(vp):
    """
    Test one-norm with an equality constraint.

    Maximizes ||x||_1 subject to x_0 = x_1, where x in [-5, 5]^2.
    Expected: 10 (both at 5 or both at -5).
    """
    n = 2
    x = vp.add_iterate(n, lb=-5, ub=5)
    vp.add_constraint(x[0] == x[1])

    vp.set_objective_norm(x, 1)
    result = vp.solve()

    x_val = vp.get_solution_value(x)
    manual_norm = np.sum(np.abs(x_val))

    assert abs(result - 10.0) < TEST_EPS, f"Expected 10.0, got {result}"
    assert abs(x_val[0] - x_val[1]) < TEST_EPS, "Constraint x_0 = x_1 should be satisfied"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_expression_difference(vp):
    """
    Test one-norm on a difference of iterates.

    Maximizes ||x - y||_1 where x, y in [0, 5]^2.
    Expected: 10 (x = [5,5], y = [0,0] or vice versa).
    """
    n = 2
    x = vp.add_iterate(n, lb=0, ub=5)
    y = vp.add_iterate(n, lb=0, ub=5)

    vp.set_objective_norm(x - y, 1)
    result = vp.solve()

    x_val = vp.get_solution_value(x)
    y_val = vp.get_solution_value(y)
    diff = x_val - y_val
    manual_norm = np.sum(np.abs(diff))

    assert abs(result - 10.0) < TEST_EPS, f"Expected 10.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_linear_transform(vp):
    """
    Test one-norm on a linear transformation.

    Maximizes ||Ax||_1 where A = [[1, 1], [1, -1]] and x in [-1, 1]^2.
    At x = [1, 1]: Ax = [2, 0], ||Ax||_1 = 2
    At x = [1, -1]: Ax = [0, 2], ||Ax||_1 = 2
    At x = [-1, 1]: Ax = [0, -2], ||Ax||_1 = 2
    At x = [-1, -1]: Ax = [-2, 0], ||Ax||_1 = 2
    Expected: 2
    """
    n = 2
    A = np.array([[1, 1], [1, -1]])
    x = vp.add_iterate(n, lb=-1, ub=1)

    vp.set_objective_norm(A @ x, 1)
    result = vp.solve()

    x_val = vp.get_solution_value(x)
    Ax_val = A @ x_val
    manual_norm = np.sum(np.abs(Ax_val))

    assert abs(result - 2.0) < TEST_EPS, f"Expected 2.0, got {result}"
    assert abs(result - manual_norm) < TEST_EPS, "Result should match manual computation"


def test_one_norm_multiple_solves(vp):
    """
    Test that solving multiple times with one-norm doesn't accumulate constraints.
    """
    n = 2
    x = vp.add_iterate(n, lb=-5, ub=5)

    vp.set_objective_norm(x, 1)
    result1 = vp.solve()
    result2 = vp.solve()

    assert abs(result1 - result2) < TEST_EPS, "Multiple solves should give same result"


def test_one_norm_vs_infinity_norm(vp):
    """
    Test that one-norm >= infinity-norm (for vectors with same sign components).

    For x in [0, 5]^3:
    - ||x||_1 = 15 at x = [5, 5, 5]
    - ||x||_inf = 5 at x = [5, *, *]

    One-norm should be >= infinity-norm.
    """
    mip_solver_name = MIP_SOLVER_NAMES[type(vp.canonicalizer.solver).__name__]
    n = 3

    # One-norm
    vp1 = Verifier(method="mip", solver=mip_solver_name, obbt_solver="highs")
    x1 = vp1.add_iterate(n, lb=0, ub=5)
    vp1.set_objective_norm(x1, 1)
    one_norm_result = vp1.solve()

    # Infinity-norm
    vp2 = Verifier(method="mip", solver=mip_solver_name, obbt_solver="highs")
    x2 = vp2.add_iterate(n, lb=0, ub=5)
    vp2.set_objective_norm(x2, np.inf)
    inf_norm_result = vp2.solve()

    assert one_norm_result >= inf_norm_result - TEST_EPS, (
        f"One-norm ({one_norm_result}) should be >= infinity-norm ({inf_norm_result})"
    )
    assert abs(one_norm_result - 15.0) < TEST_EPS
    assert abs(inf_norm_result - 5.0) < TEST_EPS


# =============================================================================
# SDP Tests
# =============================================================================


def test_one_norm_sdp_simple(vp_sdp):
    """
    Test one-norm objective on a simple bounded variable with SDP.

    Maximizes ||x||_1 where x in [-5, 5]^3.
    Expected result: 15 (each |x_i| = 5).
    """
    n = 3
    x = vp_sdp.add_iterate(n, lb=-5, ub=5)

    vp_sdp.set_objective_norm(x, 1)
    result = vp_sdp.solve()

    assert abs(result - 15.0) < TEST_EPS, f"Expected 15.0, got {result}"


def test_one_norm_sdp_after_relu(vp_sdp):
    """
    Test one-norm objective on ReLU output with SDP.

    Maximizes ||relu(x)||_1 where x in [-5, 5]^3.
    Expected result: 15 (relu(5) = 5 for each component).
    """
    n = 3
    x = vp_sdp.add_iterate(n, lb=-5, ub=5)
    z = vp_sdp.step.relu(x)

    vp_sdp.set_objective_norm(z, 1)
    result = vp_sdp.solve()

    assert abs(result - 15.0) < TEST_EPS, f"Expected 15.0, got {result}"


def test_one_norm_sdp_asymmetric_bounds(vp_sdp):
    """
    Test one-norm with asymmetric bounds using SDP.

    Maximizes ||x||_1 where x in [-2, 10]^2.
    Expected result: 20 (each |x_i| = 10).
    """
    n = 2
    x = vp_sdp.add_iterate(n, lb=-2, ub=10)

    vp_sdp.set_objective_norm(x, 1)
    result = vp_sdp.solve()

    assert abs(result - 20.0) < TEST_EPS, f"Expected 20.0, got {result}"


def test_one_norm_sdp_negative_bounds(vp_sdp):
    """
    Test one-norm where optimal is at negative values using SDP.

    Maximizes ||x||_1 where x in [-10, -2]^2.
    Expected result: 20 (each |x_i| = 10 at x_i = -10).
    """
    n = 2
    x = vp_sdp.add_iterate(n, lb=-10, ub=-2)

    vp_sdp.set_objective_norm(x, 1)
    result = vp_sdp.solve()

    assert abs(result - 20.0) < TEST_EPS, f"Expected 20.0, got {result}"


def test_one_norm_sdp_expression_difference(vp_sdp):
    """
    Test one-norm on a difference of iterates with SDP.

    Maximizes ||x - y||_1 where x, y in [0, 5]^2.
    Expected: 10 (x = [5,5], y = [0,0] or vice versa).
    """
    n = 2
    x = vp_sdp.add_iterate(n, lb=0, ub=5)
    y = vp_sdp.add_iterate(n, lb=0, ub=5)

    vp_sdp.set_objective_norm(x - y, 1)
    result = vp_sdp.solve()

    assert abs(result - 10.0) < TEST_EPS, f"Expected 10.0, got {result}"


def test_one_norm_sdp_multiple_solves(vp_sdp):
    """
    Test that solving multiple times with SDP one-norm doesn't accumulate constraints.
    """
    n = 2
    x = vp_sdp.add_iterate(n, lb=-5, ub=5)

    vp_sdp.set_objective_norm(x, 1)
    result1 = vp_sdp.solve()
    result2 = vp_sdp.solve()

    assert abs(result1 - result2) < TEST_EPS, "Multiple solves should give same result"


def test_one_norm_sdp_vs_mip(vp, vp_sdp):
    """
    Test that SDP one-norm matches MIP one-norm.

    Both should give the same result for a simple problem.
    """
    n = 3

    # MIP
    x_mip = vp.add_iterate(n, lb=-5, ub=5)
    vp.set_objective_norm(x_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-5, ub=5)
    vp_sdp.set_objective_norm(x_sdp, 1)
    sdp_result = vp_sdp.solve()

    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match"
    )


def test_one_norm_sdp_vs_two_norm(vp_sdp):
    """
    Test that one-norm >= two-norm for the same problem.

    For x in [0, 5]^3:
    - ||x||_1 = 15 at x = [5, 5, 5]
    - ||x||_2^2 = 75 at x = [5, 5, 5], so ||x||_2 = sqrt(75) ~ 8.66

    One-norm should be >= two-norm.
    """
    sdp_solver = vp_sdp.canonicalizer.solver_name
    n = 3

    # One-norm
    vp1 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    x1 = vp1.add_iterate(n, lb=0, ub=5)
    vp1.set_objective_norm(x1, 1)
    one_norm_result = vp1.solve()

    # Two-norm (returns squared norm)
    vp2 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    x2 = vp2.add_iterate(n, lb=0, ub=5)
    vp2.set_objective_norm(x2, 2)
    two_norm_sq_result = vp2.solve()
    two_norm_result = np.sqrt(two_norm_sq_result)

    assert one_norm_result >= two_norm_result - TEST_EPS, (
        f"One-norm ({one_norm_result}) should be >= two-norm ({two_norm_result})"
    )
    assert abs(one_norm_result - 15.0) < TEST_EPS
    assert abs(two_norm_sq_result - 75.0) < TEST_EPS


# =============================================================================
# Mixed SDP vs MIP Comparison Tests
# =============================================================================


def test_one_norm_sdp_mip_after_relu(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm on ReLU output.

    Maximizes ||relu(x)||_1 where x in [-5, 5]^3.
    Both methods should give 15.
    """
    n = 3

    # MIP
    x_mip = vp.add_iterate(n, lb=-5, ub=5)
    z_mip = vp.step.relu(x_mip)
    vp.set_objective_norm(z_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-5, ub=5)
    z_sdp = vp_sdp.step.relu(x_sdp)
    vp_sdp.set_objective_norm(z_sdp, 1)
    sdp_result = vp_sdp.solve()

    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match for ReLU"
    )
    assert abs(mip_result - 15.0) < TEST_EPS


def test_one_norm_sdp_mip_expression_difference(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm on difference of iterates.

    Maximizes ||x - y||_1 where x, y in [0, 5]^2.
    Both methods should give 10.
    """
    n = 2

    # MIP
    x_mip = vp.add_iterate(n, lb=0, ub=5)
    y_mip = vp.add_iterate(n, lb=0, ub=5)
    vp.set_objective_norm(x_mip - y_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=0, ub=5)
    y_sdp = vp_sdp.add_iterate(n, lb=0, ub=5)
    vp_sdp.set_objective_norm(x_sdp - y_sdp, 1)
    sdp_result = vp_sdp.solve()

    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match for expression difference"
    )
    assert abs(mip_result - 10.0) < TEST_EPS


def test_one_norm_sdp_mip_linear_transform(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm on linear transformation.

    Maximizes ||Ax||_1 where A = [[1, 1], [1, -1]] and x in [-1, 1]^2.
    MIP gives exact result 2. SDP is a relaxation so it upper bounds MIP.
    """
    n = 2
    A = np.array([[1, 1], [1, -1]])

    # MIP
    x_mip = vp.add_iterate(n, lb=-1, ub=1)
    vp.set_objective_norm(A @ x_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-1, ub=1)
    vp_sdp.set_objective_norm(A @ x_sdp, 1)
    sdp_result = vp_sdp.solve()

    # SDP is a relaxation, so it should upper bound the MIP result
    assert sdp_result >= mip_result - TEST_EPS, (
        f"SDP ({sdp_result}) should be >= MIP ({mip_result}) as SDP is a relaxation"
    )
    assert abs(mip_result - 2.0) < TEST_EPS


def test_one_norm_sdp_mip_negative_bounds(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm with all-negative bounds.

    Maximizes ||x||_1 where x in [-10, -2]^2.
    Both methods should give 20.
    """
    n = 2

    # MIP
    x_mip = vp.add_iterate(n, lb=-10, ub=-2)
    vp.set_objective_norm(x_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-10, ub=-2)
    vp_sdp.set_objective_norm(x_sdp, 1)
    sdp_result = vp_sdp.solve()

    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match for negative bounds"
    )
    assert abs(mip_result - 20.0) < TEST_EPS


def test_one_norm_sdp_mip_asymmetric_bounds(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm with asymmetric bounds.

    Maximizes ||x||_1 where x in [-2, 10]^2.
    Both methods should give 20.
    """
    n = 2

    # MIP
    x_mip = vp.add_iterate(n, lb=-2, ub=10)
    vp.set_objective_norm(x_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-2, ub=10)
    vp_sdp.set_objective_norm(x_sdp, 1)
    sdp_result = vp_sdp.solve()

    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match for asymmetric bounds"
    )
    assert abs(mip_result - 20.0) < TEST_EPS


def test_one_norm_sdp_mip_with_constraint(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm with equality constraint.

    Maximizes ||x||_1 subject to x_0 = x_1, where x in [-5, 5]^2.
    Both methods should give 10.
    """
    n = 2

    # MIP
    x_mip = vp.add_iterate(n, lb=-5, ub=5)
    vp.add_constraint(x_mip[0] == x_mip[1])
    vp.set_objective_norm(x_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-5, ub=5)
    vp_sdp.add_constraint(x_sdp[0] == x_sdp[1])
    vp_sdp.set_objective_norm(x_sdp, 1)
    sdp_result = vp_sdp.solve()

    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match with constraint"
    )
    assert abs(mip_result - 10.0) < TEST_EPS


def test_one_norm_sdp_mip_two_relu_steps(vp, vp_sdp):
    """
    Compare SDP and MIP one-norm after two ReLU steps.

    Maximizes ||relu(relu(x) - 2)||_1 where x in [-5, 5]^2.
    """
    n = 2

    # MIP
    x_mip = vp.add_iterate(n, lb=-5, ub=5)
    z1_mip = vp.step.relu(x_mip)
    z2_mip = vp.step.relu(z1_mip - 2)
    vp.set_objective_norm(z2_mip, 1)
    mip_result = vp.solve()

    # SDP
    x_sdp = vp_sdp.add_iterate(n, lb=-5, ub=5)
    z1_sdp = vp_sdp.step.relu(x_sdp)
    z2_sdp = vp_sdp.step.relu(z1_sdp - 2)
    vp_sdp.set_objective_norm(z2_sdp, 1)
    sdp_result = vp_sdp.solve()

    # relu(relu(5) - 2) = relu(5 - 2) = relu(3) = 3, so ||z2||_1 = 6
    assert abs(mip_result - sdp_result) < TEST_EPS, (
        f"MIP ({mip_result}) and SDP ({sdp_result}) should match for two ReLU steps"
    )
    assert abs(mip_result - 6.0) < TEST_EPS
