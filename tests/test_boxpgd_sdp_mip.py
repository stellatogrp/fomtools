import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}


def test_boxpgd_sdp_mip_comparison(vp, vp_sdp):
    """
    Test that MIP with infinity norm lower bounds SDP with two norm for Box PGD.

    For the same box-constrained projected gradient descent problem, we verify:
        mip_inf_norm <= sqrt(sdp_two_norm_squared)

    This relationship holds because the infinity norm is always <= the two norm.
    This test also validates that the SDP decomposition of SaturatedLinear into
    ReLU operations works correctly.
    """
    n = 2
    np.random.seed(0)

    # Simple quadratic: minimize 0.5 * x^T P x + q^T x subject to l <= x <= u
    P = np.eye(n)
    q = np.random.normal(size=(n,))

    # Box constraints
    l = np.array([-1.0, -0.5])
    u = np.array([0.5, 1.0])

    # Step size (should be < 2/L where L is max eigenvalue of P)
    t = 0.5

    # Get solver names for creating fresh verifiers
    mip_solver_name = MIP_SOLVER_NAMES[type(vp.canonicalizer.solver).__name__]
    sdp_solver_name = vp_sdp.canonicalizer.solver_name

    for K in [1, 2]:
        # SDP with two-norm objective
        vp_sdp_local = Verifier(
            method="sdp",
            solver=sdp_solver_name,
            obbt_handler='highs',
        )

        q_sdp = vp_sdp_local.add_param(n, lb=q - 0.1, ub=q + 0.1)
        z_sdp = [vp_sdp_local.add_iterate(n, lb=0, ub=0.1)]

        for k in range(K):
            # Box PGD step: z_{k+1} = proj(z_k - t * (P @ z_k + q), l, u)
            grad_step = z_sdp[k] - t * (P @ z_sdp[k] + q_sdp)
            z_sdp.append(vp_sdp_local.step.satlin(grad_step, l=l, u=u))

        vp_sdp_local.set_objective_norm(z_sdp[K] - z_sdp[K - 1], 2)
        sdp_two_norm_sq = vp_sdp_local.solve()

        # MIP with infinity-norm objective
        vp_mip = Verifier(
            method="mip",
            solver=mip_solver_name,
            obbt_handler='highs',
        )

        q_mip = vp_mip.add_param(n, lb=q - 0.1, ub=q + 0.1)
        z_mip = [vp_mip.add_iterate(n, lb=0, ub=0.1)]

        for k in range(K):
            grad_step = z_mip[k] - t * (P @ z_mip[k] + q_mip)
            z_mip.append(vp_mip.step.satlin(grad_step, l=l, u=u))

        vp_mip.set_objective_norm([z_mip[K] - z_mip[K - 1]], np.inf)
        mip_inf_norm = vp_mip.solve()

        sdp_two_norm = np.sqrt(sdp_two_norm_sq)

        print(f"K={K}: MIP inf norm = {mip_inf_norm:.6f}, SDP two norm = {sdp_two_norm:.6f}")

        # MIP inf norm should lower bound SDP two norm
        assert mip_inf_norm <= sdp_two_norm + TEST_EPS, (
            f"K={K}: MIP inf norm ({mip_inf_norm}) should be <= SDP two norm ({sdp_two_norm})"
        )


def test_boxpgd_sdp_satlin_vs_relus(vp_sdp):
    """
    Test manual saturated linear decomposition into ReLUs.

    Manually implements: satlin(x, l, u) = relu(x - l) - relu(x - u) + l
    and verifies that the SDP objective matches the automatic decomposition.
    """
    n = 2
    np.random.seed(0)

    P = np.eye(n)
    q = np.random.normal(size=(n,))

    # Box constraints
    l = np.array([-1.0, -0.5])
    u = np.array([0.5, 1.0])

    t = 0.5

    sdp_solver_name = vp_sdp.canonicalizer.solver_name

    # --- Manual decomposition: satlin(x, l, u) = relu(x - l) - relu(x - u) + l ---
    vp_manual = Verifier(
        method="sdp",
        solver=sdp_solver_name,
        obbt_handler='highs',
    )

    q_manual = vp_manual.add_param(n, lb=q - 0.1, ub=q + 0.1)
    z0_manual = vp_manual.add_iterate(n, lb=0, ub=0.1)

    # Compute the input to satlin: rhs = z0 - t * (P @ z0 + q)
    rhs = z0_manual - t * (P @ z0_manual + q_manual)

    # Manual ReLU decomposition
    relu_lower_lhs = vp_manual.step.relu(rhs - l)
    relu_upper_lhs = vp_manual.step.relu(rhs - u)

    # z1 = relu(x - l) - relu(x - u) + l (this is the satlin output)
    z1_manual = relu_lower_lhs - relu_upper_lhs + l

    # Set objective on z1 - z0
    vp_manual.set_objective_norm(z1_manual - z0_manual, 2)
    result_manual = vp_manual.solve()

    # --- Automatic decomposition using step.satlin ---
    vp_auto = Verifier(
        method="sdp",
        solver=sdp_solver_name,
        obbt_handler='highs',
    )

    q_auto = vp_auto.add_param(n, lb=q - 0.1, ub=q + 0.1)
    z0_auto = vp_auto.add_iterate(n, lb=0, ub=0.1)

    # Use automatic satlin decomposition
    rhs_auto = z0_auto - t * (P @ z0_auto + q_auto)
    z1_auto = vp_auto.step.satlin(rhs_auto, l=l, u=u)

    vp_auto.set_objective_norm(z1_auto - z0_auto, 2)
    result_auto = vp_auto.solve()

    print(f"Manual decomposition result: {result_manual}")
    print(f"Automatic decomposition result: {result_auto}")

    # Both should be valid
    assert not np.isnan(result_manual), "Manual decomposition should not return nan"
    assert not np.isnan(result_auto), "Automatic decomposition should not return nan"
    assert result_manual >= 0, "Two-norm squared should be non-negative"
    assert result_auto >= 0, "Two-norm squared should be non-negative"

    # Results should nearly match
    assert np.abs(result_manual - result_auto) <= TEST_EPS, (
        f"Manual ({result_manual}) and automatic ({result_auto}) decomposition "
        f"results should match within tolerance {TEST_EPS}"
    )
