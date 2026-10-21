import numpy as np

import fomtools as ft
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}


def test_add_constraint_equality(vp):
    """
    Test that add_constraint works with equality constraints.

    This test verifies that:
    1. Adding an equality constraint doesn't break the solver
    2. The constraint is respected in the solution
    """
    n = 3

    # Create two iterates
    x = vp.iterate(n, lb=0, ub=10)
    y = vp.iterate(n, lb=0, ub=10)

    # Add equality constraint: x[0] == y[0]
    vp.add_constraint(x[0] == y[0])

    # Add a ReLU step to make the problem non-trivial
    z = vp.relu(x + y)

    # Set objective and solve
    vp.maximize(ft.norm([z], np.inf))
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)

    # Verify the constraint is respected
    x_val = x.value
    y_val = y.value
    assert np.abs(x_val[0] - y_val[0]) < TEST_EPS, "Equality constraint not satisfied"


def test_add_constraint_inequality(vp):
    """
    Test that inequality constraints restrict the optimal solution.

    Maximizes inf_norm(relu(x - 3)) with x in [0, 10].
    Without constraint: optimal is 7 (x=10 gives relu(7)=7).
    With x <= 5: optimal is 2 (x=5 gives relu(2)=2).
    """
    n = 3
    mip_solver_name = MIP_SOLVER_NAMES[type(vp.canonicalizer.solver).__name__]

    # Unconstrained baseline
    vp_unc = Verifier(method="mip", solver=mip_solver_name, obbt_solver=None)
    x_unc = vp_unc.iterate(n, lb=0, ub=10)
    z_unc = vp_unc.relu(x_unc - 3)
    vp_unc.maximize(ft.norm([z_unc], np.inf))
    result_unc = vp_unc.solve()

    # Constrained: x[i] <= 5 for all i
    vp_con = Verifier(method="mip", solver=mip_solver_name, obbt_solver=None)
    x_con = vp_con.iterate(n, lb=0, ub=10)
    for i in range(n):
        vp_con.add_constraint(x_con[i] <= 5)
    z_con = vp_con.relu(x_con - 3)
    vp_con.maximize(ft.norm([z_con], np.inf))
    result_con = vp_con.solve()

    assert np.isfinite(result_unc)
    assert np.isfinite(result_con)
    assert result_con < result_unc, "Constraint should reduce objective"
    assert abs(result_con - 2.0) < TEST_EPS, f"Expected 2.0, got {result_con}"


def test_add_constraint_tightens_bounds(vp):
    """
    Test that adding a constraint can tighten bounds via OBBT.

    This test verifies that adding a constraint that restricts the
    feasible region results in bounds that are at most as large as
    without the constraint.
    """
    n = 2

    # Create a parameter and iterate
    p = vp.parameter(n, lb=0, ub=10)
    x = vp.iterate(n, lb=-100, ub=100)

    # Add constraint that restricts x: x == p (so x should be in [0, 10])
    vp.add_constraint(x == p)

    # Add a ReLU step
    z = vp.relu(x)

    # Set objective and solve
    vp.maximize(ft.norm([z], np.inf))
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)

    # The result should be bounded by the parameter bounds
    # Since x == p and p in [0, 10], and ReLU(x) = max(x, 0),
    # the maximum should be at most 10
    assert result <= 10 + TEST_EPS, "Constraint should tighten bounds"


def test_affine_step_solves(vp):
    """
    Test that an Affine step (``implicit_linear``) composed with a ReLU
    produces a solvable verification problem.
    """
    import scipy.sparse as spa

    n = 3

    # Create a simple invertible matrix
    M = np.eye(n) + 0.1 * np.ones((n, n))
    output_mat = spa.csc_matrix(M)

    # Create parameter and iterate
    p = vp.parameter(n, lb=1, ub=2)
    z0 = vp.iterate(n, lb=0, ub=0)

    # Apply implicit_linear (Affine) step
    z1 = vp.implicit_linear(z0 + p, output_mat=output_mat.todense())

    # Add another step
    z2 = vp.relu(z1)

    # Set objective and solve
    vp.maximize(ft.norm([z2], np.inf))
    result = vp.solve()

    # Verify the problem is solvable
    assert result is not None
    assert np.isfinite(result)


def test_add_constraint_equality_sdp_mip_comparison(vp, vp_sdp):
    """
    Test that SDP relaxation provides valid upper bound on MIP.

    Uses infinity-norm for MIP, two-norm for SDP.
    Since ||x||_inf <= ||x||_2, we have:
        mip_inf_norm <= sqrt(sdp_two_norm_squared)
    """
    n = 3

    # Get solver names for creating fresh verifiers
    mip_solver_name = MIP_SOLVER_NAMES[type(vp.canonicalizer.solver).__name__]
    sdp_solver_name = vp_sdp.canonicalizer.solver_name

    # MIP with infinity-norm objective
    vp_mip = Verifier(method="mip", solver=mip_solver_name, obbt_solver=None)

    x_mip = vp_mip.iterate(n, lb=0, ub=10)
    y_mip = vp_mip.iterate(n, lb=0, ub=10)

    # Add equality constraint: x[0] == y[0]
    vp_mip.add_constraint(x_mip[0] == y_mip[0])

    z_mip = vp_mip.relu(x_mip + y_mip)
    vp_mip.maximize(ft.norm([z_mip], np.inf))
    mip_inf_norm = vp_mip.solve()

    # SDP with two-norm objective
    vp_sdp_local = Verifier(method="sdp", solver=sdp_solver_name, obbt_solver=None)

    x_sdp = vp_sdp_local.iterate(n, lb=0, ub=10)
    y_sdp = vp_sdp_local.iterate(n, lb=0, ub=10)

    # Add equality constraint: x[0] == y[0]
    vp_sdp_local.add_constraint(x_sdp[0] == y_sdp[0])

    z_sdp = vp_sdp_local.relu(x_sdp + y_sdp)
    vp_sdp_local.maximize(ft.sum_squares(z_sdp))
    sdp_two_norm_sq = vp_sdp_local.solve()
    sdp_two_norm = np.sqrt(sdp_two_norm_sq)

    # MIP inf norm should be <= SDP two norm
    assert mip_inf_norm <= sdp_two_norm + TEST_EPS, (
        f"MIP inf norm ({mip_inf_norm}) should be <= SDP two norm ({sdp_two_norm})"
    )
