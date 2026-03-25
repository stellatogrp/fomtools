import numpy as np

from tests.conftest import TEST_EPS


def test_add_constraint_equality(vp):
    """
    Test that add_constraint works with equality constraints.

    This test verifies that:
    1. Adding an equality constraint doesn't break the solver
    2. The constraint is respected in the solution
    """
    n = 3

    # Create two iterates
    x = vp.add_iterate(n, lb=0, ub=10)
    y = vp.add_iterate(n, lb=0, ub=10)

    # Add equality constraint: x[0] == y[0]
    vp.add_constraint(x[0] == y[0])

    # Add a ReLU step to make the problem non-trivial
    z = vp.step.relu(x + y)

    # Set objective and solve
    vp.set_infinity_norm_objective([z])
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)

    # Verify the constraint is respected
    x_val = vp.get_solution_value(x)
    y_val = vp.get_solution_value(y)
    assert np.abs(x_val[0] - y_val[0]) < TEST_EPS, "Equality constraint not satisfied"


def test_add_constraint_inequality(vp):
    """
    Test that add_constraint works with inequality constraints.

    This test verifies that:
    1. Adding an inequality constraint doesn't break the solver
    2. The constraint is respected in the solution
    """
    n = 3

    # Create an iterate
    x = vp.add_iterate(n, lb=0, ub=10)

    # Add inequality constraint: x[0] >= 5
    vp.add_constraint(x[0] >= 5)

    # Add a ReLU step
    z = vp.step.relu(x - 3)

    # Set objective and solve
    vp.set_infinity_norm_objective([z])
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)

    # Verify the constraint is respected
    x_val = vp.get_solution_value(x)
    assert x_val[0] >= 5 - TEST_EPS, "Inequality constraint not satisfied"


def test_add_constraint_tightens_bounds(vp):
    """
    Test that adding a constraint can tighten bounds via OBBT.

    This test verifies that adding a constraint that restricts the
    feasible region results in bounds that are at most as large as
    without the constraint.
    """
    n = 2

    # Create a parameter and iterate
    p = vp.add_param(n, lb=0, ub=10)
    x = vp.add_iterate(n, lb=-100, ub=100)

    # Add constraint that restricts x: x == p (so x should be in [0, 10])
    vp.add_constraint(x == p)

    # Add a ReLU step
    z = vp.step.relu(x)

    # Set objective and solve
    vp.set_infinity_norm_objective([z])
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)

    # The result should be bounded by the parameter bounds
    # Since x == p and p in [0, 10], and ReLU(x) = max(x, 0),
    # the maximum should be at most 10
    assert result <= 10 + TEST_EPS, "Constraint should tighten bounds"


def test_skip_obbt_affine_step(vp):
    """
    Test that Affine steps correctly skip OBBT.

    This test verifies that the skip_obbt=True attribute on Affine
    steps works correctly and the problem is still solvable.
    """
    import scipy.sparse as spa

    n = 3

    # Create a simple invertible matrix
    M = np.eye(n) + 0.1 * np.ones((n, n))
    lhs = spa.csc_matrix(M)
    lhs_factored = spa.linalg.factorized(lhs)

    # Create parameter and iterate
    p = vp.add_param(n, lb=1, ub=2)
    z0 = vp.add_iterate(n, lb=0, ub=0)

    # Apply implicit_linear (Affine) step - this should skip OBBT
    z1 = vp.step.implicit_linear(z0 + p, lhs_mat=lhs.todense(), lhs_mat_factorization=lhs_factored)

    # Add another step
    z2 = vp.step.relu(z1)

    # Set objective and solve
    vp.set_infinity_norm_objective([z2])
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)
