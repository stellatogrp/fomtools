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

    vp_test = Verifier(method='mip', solver=mip_solver_name, obbt_handler=None)
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
    vp1 = Verifier(method='mip', solver=mip_solver_name, obbt_handler=None)
    x1 = vp1.add_iterate(n, lb=0, ub=5)
    vp1.set_objective_norm(x1, 1)
    one_norm_result = vp1.solve()

    # Infinity-norm
    vp2 = Verifier(method='mip', solver=mip_solver_name, obbt_handler=None)
    x2 = vp2.add_iterate(n, lb=0, ub=5)
    vp2.set_objective_norm(x2, np.inf)
    inf_norm_result = vp2.solve()

    assert one_norm_result >= inf_norm_result - TEST_EPS, (
        f"One-norm ({one_norm_result}) should be >= infinity-norm ({inf_norm_result})"
    )
    assert abs(one_norm_result - 15.0) < TEST_EPS
    assert abs(inf_norm_result - 5.0) < TEST_EPS
