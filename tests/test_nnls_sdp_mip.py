import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}


def test_nnls_sdp_mip_comparison(vp, vp_sdp):
    """
    Test that MIP with infinity norm lower bounds SDP with two norm.

    For the same NNLS problem, we verify:
        mip_inf_norm <= sqrt(sdp_two_norm_squared)

    This relationship holds because the infinity norm is always <= the two norm.
    """
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    # Get solver names for creating fresh verifiers
    mip_solver_name = MIP_SOLVER_NAMES[type(vp.canonicalizer.solver).__name__]
    sdp_solver_name = vp_sdp.canonicalizer.solver_name

    for K in [1, 2]:
        # SDP with two-norm objective
        vp_sdp_local = Verifier(
            method="sdp",
            solver=sdp_solver_name,
            obbt_solver="highs",
        )

        x_sdp = vp_sdp_local.add_param(m, lb=0, ub=0.1)
        z_sdp = [vp_sdp_local.add_iterate(n, lb=5, ub=5.1)]

        for k in range(K):
            z_sdp.append(vp_sdp_local.step.relu(z_sdp[k] - t * A.T @ (A @ z_sdp[k] - x_sdp)))

        vp_sdp_local.set_objective_norm(z_sdp[K] - z_sdp[K - 1], 2)
        sdp_two_norm_sq = vp_sdp_local.solve()

        # MIP with infinity-norm objective
        vp_mip = Verifier(
            method="mip",
            solver=mip_solver_name,
            obbt_solver="highs",
        )

        x_mip = vp_mip.add_param(m, lb=0, ub=0.1)
        z_mip = [vp_mip.add_iterate(n, lb=5, ub=5.1)]

        for k in range(K):
            z_mip.append(vp_mip.step.relu(z_mip[k] - t * A.T @ (A @ z_mip[k] - x_mip)))

        vp_mip.set_objective_norm([z_mip[K] - z_mip[K - 1]], np.inf)
        mip_inf_norm = vp_mip.solve()

        sdp_two_norm = np.sqrt(sdp_two_norm_sq)

        print(f"K={K}: MIP inf norm = {mip_inf_norm:.6f}, SDP two norm = {sdp_two_norm:.6f}")

        # MIP inf norm should lower bound SDP two norm
        assert mip_inf_norm <= sdp_two_norm + TEST_EPS, (
            f"K={K}: MIP inf norm ({mip_inf_norm}) should be <= SDP two norm ({sdp_two_norm})"
        )
