import numpy as np

from fomtools.verifier import Verifier


def test_nnls_sdp_mip_comparison():
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

    for K in [1, 2]:
        # SDP with two-norm objective
        vp_sdp = Verifier(method='sdp', solver='cvxpy_sdp', obbt_handler=None)

        x_sdp = vp_sdp.add_param(m, lb=0, ub=0.1)
        z_sdp = [vp_sdp.add_iterate(n, lb=5, ub=5.1)]

        for k in range(K):
            z_sdp.append(vp_sdp.step.relu(z_sdp[k] - t * A.T @ (A @ z_sdp[k] - x_sdp)))

        vp_sdp.set_two_norm_objective(z_sdp[K] - z_sdp[K - 1])
        sdp_two_norm_sq = vp_sdp.canonicalizer.solve_via_clarabel()

        # MIP with infinity-norm objective
        vp_mip = Verifier(method='mip', solver='highs', obbt_handler=None)

        x_mip = vp_mip.add_param(m, lb=0, ub=0.1)
        z_mip = [vp_mip.add_iterate(n, lb=5, ub=5.1)]

        for k in range(K):
            z_mip.append(vp_mip.step.relu(z_mip[k] - t * A.T @ (A @ z_mip[k] - x_mip)))

        vp_mip.set_infinity_norm_objective([z_mip[K] - z_mip[K - 1]])
        mip_inf_norm = vp_mip.solve()

        sdp_two_norm = np.sqrt(sdp_two_norm_sq)

        print(f"K={K}: MIP inf norm = {mip_inf_norm:.6f}, SDP two norm = {sdp_two_norm:.6f}")

        # MIP inf norm should lower bound SDP two norm
        assert mip_inf_norm <= sdp_two_norm + 1e-6, (
            f"K={K}: MIP inf norm ({mip_inf_norm}) should be <= SDP two norm ({sdp_two_norm})"
        )
