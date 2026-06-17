import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}


def test_ista_sdp_mip_comparison(vp, vp_sdp):
    """
    Test that MIP with infinity norm lower bounds SDP with two norm for ISTA.

    For the same ISTA (soft-thresholding) problem, we verify:
        mip_inf_norm <= sqrt(sdp_two_norm_squared)

    This relationship holds because the infinity norm is always <= the two norm.
    This test also validates that the SDP decomposition of SoftThreshold into
    ReLU operations works correctly.
    """
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))

    # Compute step size based on Lipschitz constant
    lambd_max = np.max(np.real(np.linalg.eigvals(A.T @ A)))
    t = 1 / lambd_max
    lambd = 0.5  # Regularization parameter

    # Precompute ISTA matrices
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

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

        b_sdp = vp_sdp_local.add_param(m, lb=b - 0.1, ub=b + 0.1)
        z_sdp = [vp_sdp_local.add_iterate(n, lb=1, ub=1.1)]

        for k in range(K):
            z_sdp.append(
                vp_sdp_local.step.soft_threshold(At @ z_sdp[k] + Bt @ b_sdp, lambd=lambda_t)
            )

        vp_sdp_local.set_objective_norm(z_sdp[K] - z_sdp[K - 1], 2)
        sdp_two_norm_sq = vp_sdp_local.solve()

        # MIP with infinity-norm objective
        vp_mip = Verifier(
            method="mip",
            solver=mip_solver_name,
            obbt_solver="highs",
        )

        b_mip = vp_mip.add_param(m, lb=b - 0.1, ub=b + 0.1)
        z_mip = [vp_mip.add_iterate(n, lb=1, ub=1.1)]

        for k in range(K):
            z_mip.append(vp_mip.step.soft_threshold(At @ z_mip[k] + Bt @ b_mip, lambd=lambda_t))

        vp_mip.set_objective_norm([z_mip[K] - z_mip[K - 1]], np.inf)
        mip_inf_norm = vp_mip.solve()

        sdp_two_norm = np.sqrt(sdp_two_norm_sq)

        print(f"K={K}: MIP inf norm = {mip_inf_norm:.6f}, SDP two norm = {sdp_two_norm:.6f}")

        # MIP inf norm should lower bound SDP two norm
        assert mip_inf_norm <= sdp_two_norm + TEST_EPS, (
            f"K={K}: MIP inf norm ({mip_inf_norm}) should be <= SDP two norm ({sdp_two_norm})"
        )


def test_ista_sdp_softthresh_vs_relus(vp_sdp):
    """
    Test manual soft threshold decomposition into ReLUs.

    Manually implements: softthresh_λ(x) = relu(x - λ) - relu(-x - λ)
    and verifies that the SDP objective matches the automatic decomposition.
    """
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))

    # Compute step size based on Lipschitz constant
    lambd_max = np.max(np.real(np.linalg.eigvals(A.T @ A)))
    t = 1 / lambd_max
    lambd = 0.5  # Regularization parameter

    # Precompute ISTA matrices
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    sdp_solver_name = vp_sdp.canonicalizer.solver_name

    # --- Manual decomposition: softthresh_λ(x) = relu(x - λ) - relu(-x - λ) ---
    vp_manual = Verifier(
        method="sdp",
        solver=sdp_solver_name,
        obbt_solver=None,
    )

    b_manual = vp_manual.add_param(m, lb=b - 0.1, ub=b + 0.1)
    z0_manual = vp_manual.add_iterate(n, lb=1, ub=1.1)

    # Compute the input to soft threshold: arg = At @ z0 + Bt @ b_param
    arg = At @ z0_manual + Bt @ b_manual

    # Create lambda vector
    lambd_vec = np.full(n, lambda_t)

    # Manual ReLU decomposition
    relu_pos_output = vp_manual.step.relu(arg - lambd_vec)
    relu_neg_output = vp_manual.step.relu(-arg - lambd_vec)

    # z1 = relu_pos - relu_neg (this is the soft threshold output)
    z1_manual = relu_pos_output - relu_neg_output

    # Set objective on z1 - z0
    vp_manual.set_objective_norm(z1_manual - z0_manual, 2)
    result_manual = vp_manual.solve()

    # --- Automatic decomposition using step.soft_threshold ---
    vp_auto = Verifier(
        method="sdp",
        solver=sdp_solver_name,
        obbt_solver=None,
    )

    b_auto = vp_auto.add_param(m, lb=b - 0.1, ub=b + 0.1)
    z0_auto = vp_auto.add_iterate(n, lb=1, ub=1.1)

    # Use automatic soft threshold decomposition
    z1_auto = vp_auto.step.soft_threshold(At @ z0_auto + Bt @ b_auto, lambd=lambda_t)

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
