import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}

# Cut flags all disabled (base inequality constraint only)
CUTS_OFF = dict(
    ineq_product_cuts=False,
    ineq_box_cuts=False,
    ineq_square_cuts=False,
)


def test_sdp_inequality_no_longer_raises(vp_sdp):
    """
    Adding an inequality constraint to an SDP verifier no longer raises and solves.

    Previously ``_add_inequality_constraint_impl`` was ``NotImplementedError``.
    """
    x = vp_sdp.add_iterate(2, lb=-1, ub=1)
    vp_sdp.add_constraint(x[0] + x[1] <= 0.5)
    vp_sdp.set_objective_norm(x, 2)
    result = vp_sdp.solve()

    assert result is not None
    assert np.isfinite(result)
    # ||x||_2^2 over [-1,1]^2 with x0+x1<=0.5; (1,-1) is feasible -> max is 2.
    assert result <= 2.0 + TEST_EPS


def test_sdp_inequality_valid_bound_vs_mip(vp_sdp):
    """
    SDP relaxation with an inequality constraint is a valid bound on the MIP.

    Mirrors ``test_add_constraint_equality_sdp_mip_comparison`` but with an
    inequality constraint. Uses inf-norm for MIP, two-norm for SDP; since
    ``||x||_inf <= ||x||_2`` we require ``mip_inf <= sqrt(sdp_two_sq)``.
    """
    n = 3
    sdp_solver_name = vp_sdp.canonicalizer.solver_name

    # MIP with infinity-norm objective
    vp_mip = Verifier(method="mip", solver="gurobi", obbt_solver=None, params={"OutputFlag": 0})
    x_mip = vp_mip.add_iterate(n, lb=-1, ub=1)
    vp_mip.add_constraint(x_mip[0] + x_mip[1] <= 0.5)
    z_mip = vp_mip.step.relu(x_mip)
    vp_mip.set_objective_norm([z_mip], np.inf)
    mip_inf = vp_mip.solve()

    # SDP with two-norm objective (default cuts on)
    vp_sdp_local = Verifier(method="sdp", solver=sdp_solver_name, obbt_solver=None)
    x_sdp = vp_sdp_local.add_iterate(n, lb=-1, ub=1)
    vp_sdp_local.add_constraint(x_sdp[0] + x_sdp[1] <= 0.5)
    z_sdp = vp_sdp_local.step.relu(x_sdp)
    vp_sdp_local.set_objective_norm(z_sdp, 2)
    sdp_two = np.sqrt(vp_sdp_local.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS, (
        f"MIP inf-norm ({mip_inf}) should be <= SDP two-norm ({sdp_two})"
    )


def test_ineq_product_cut_tightens(vp_sdp):
    """
    The off-diagonal product cut tightens a loose relaxation to the true value.

    A single 2-row constraint ``|x0 + x1| <= 0.5`` (rows ``x0+x1<=0.5`` and
    ``-x0-x1<=0.5``) with objective ``max (x0+x1)^2``. The product of the two
    rows gives ``(x0+x1)^2 <= 0.25`` exactly. Without cuts the box-only
    relaxation allows up to 4.0.
    """
    sdp_solver_name = vp_sdp.canonicalizer.solver_name
    A = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b = np.array([0.5, 0.5])

    def run(flags):
        vp = Verifier(method="sdp", solver=sdp_solver_name, obbt_solver=None, **flags)
        x = vp.add_iterate(2, lb=-1, ub=1)
        vp.add_constraint(A @ x <= b)
        vp.set_objective_norm([x[0] + x[1]], 2)
        return vp.solve()

    obj_off = run(CUTS_OFF)
    obj_prod = run(dict(ineq_box_cuts=False, ineq_square_cuts=False))

    assert obj_off > 1.0, f"no-cut relaxation should be loose, got {obj_off}"
    assert obj_prod <= obj_off + TEST_EPS, "cuts must not loosen the relaxation"
    assert abs(obj_prod - 0.25) < 1e-4, f"product cut should give 0.25, got {obj_prod}"


def test_each_ineq_flag_adds_cuts(vp_sdp):
    """
    Each cut flag is individually wired: enabling it appends cuts to the block.

    Uses a single 2-row constraint so the product cut has an ``i<j`` pair.
    """
    sdp_solver_name = vp_sdp.canonicalizer.solver_name
    A = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b = np.array([0.5, 0.5])

    def ncuts(flags):
        vp = Verifier(method="sdp", solver=sdp_solver_name, obbt_solver=None, **flags)
        x = vp.add_iterate(2, lb=-1, ub=1)
        vp.add_constraint(A @ x <= b)
        vp.set_objective_norm([x[0] + x[1]], 2)
        vp.solve()
        return len(vp.canonicalizer._solved_copy._cut_ineq_matrices)

    assert ncuts(CUTS_OFF) == 0
    assert ncuts(dict(ineq_box_cuts=False, ineq_square_cuts=False)) > 0  # product
    assert ncuts(dict(ineq_product_cuts=False, ineq_square_cuts=False)) > 0  # box
    assert ncuts(dict(ineq_product_cuts=False, ineq_box_cuts=False)) > 0  # square
    assert ncuts({}) > ncuts(CUTS_OFF)  # default (all on) adds cuts


def test_cuts_track_bound_tightening(vp_sdp_obbt):
    """
    With OBBT enabled, an inequality constraint + cuts solve without stale-dim errors.

    The lazy cut rebuild must pick up tightened bounds (``_cuts_dirty`` is set on
    ``update_bounds``) and rebuild cut matrices at the current SDP dimension.
    """
    x = vp_sdp_obbt.add_iterate(3, lb=-1, ub=1)
    vp_sdp_obbt.add_constraint(x[0] + x[1] + x[2] <= -0.5)
    z = vp_sdp_obbt.step.relu(x)
    vp_sdp_obbt.set_objective_norm(z, 2)
    result = vp_sdp_obbt.solve()

    assert np.isfinite(result)
