import numpy as np

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS, mip_params
from tests.cvxpy import satlin as satlin_cvxpy

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_satlin(vp_param_2):
    np.random.seed(8)

    n = 3
    P = np.eye(n)
    q = np.zeros(n)
    l = np.array([-1, 1, -0.1])
    u = np.array([-0.9, 1.01, 0.1])

    K = 100
    t = 0.01
    z0_init = (l + u) / 2
    zk = z0_init
    for _ in range(K):
        yk = zk - t * (P @ zk + q)
        zk = proj(yk, l, u)

    x, res = satlin_cvxpy(n, P, q, l, u)
    print("test solution from cvxpy:", x)
    print("solution from pgd:", zk)
    assert np.linalg.norm(x - zk) <= TEST_EPS

    q_offset = 1
    q_param = vp_param_2.parameter(n, lb=q - q_offset, ub=q + q_offset)

    z0 = vp_param_2.iterate(n, lb=z0_init, ub=z0_init)

    K = 12
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_res = []
    for k in range(1, K + 1):
        print(k)
        z[k] = vp_param_2.satlin(z[k - 1] - t * (P @ z[k - 1] + q_param), l=l, u=u)

        vp_param_2.set_objective_norm(z[k] - z[k - 1], np.inf)
        res = vp_param_2.solve()
        all_res.append(res)

    print(f"opt q_param at last K: {vp_param_2.get_solution_value(q_param)}")
    print(f"VP resids: {all_res}")

    print("extracting values from VP:")
    for k in range(K + 1):
        print(f"-K={k}-")
        print(vp_param_2.get_solution_value(z[k]))

    z_test = np.array([-0.91651, 1, 0])
    print(
        "test grad step from K-1:",
        z_test - t * (P @ z_test + vp_param_2.get_solution_value(q_param)),
    )
    print("bounds:", vp_param_2.iterates[-1].lb, vp_param_2.iterates[-1].ub)

    zk = z0_init
    q = vp_param_2.get_solution_value(q_param)
    PGD_resids = []

    for k in range(K):
        znew = proj(zk - t * (P @ zk + q), l, u)
        PGD_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print("PGD resids with last K:", PGD_resids)
    assert np.all(np.array(PGD_resids) <= np.array(all_res) + TEST_EPS)
    assert np.abs(PGD_resids[-1] - all_res[-1]) <= TEST_EPS


def test_satlin_param(vp_param_2):
    np.random.seed(8)

    n = 3
    P = np.eye(n)
    q = np.ones(n)

    l = np.array([-1, 1, -0.1])
    u = np.array([-0.9, 1.01, 0.1])

    x, res = satlin_cvxpy(n, P, q, l, u)

    print("test solution from cvxpy:", x)

    q_offset = 0.2
    q_param = vp_param_2.parameter(n, lb=q - q_offset, ub=q + q_offset)

    top_dim = 1
    top_param = vp_param_2.parameter(top_dim, lb=0.5, ub=0.6)
    l_rest = vp_param_2.parameter(n - 1, lb=np.array([-1.2, -0.35]), ub=np.array([-1.1, 0]))
    u_rest = vp_param_2.parameter(n - 1, lb=np.array([0, 0.1]), ub=np.array([0.1, 0.2]))

    l_param = vp_param_2.add_param_stack([top_param, l_rest])
    u_param = vp_param_2.add_param_stack([top_param, u_rest])

    t = 0.1
    K = 5

    z = [None for _ in range(K + 1)]
    z[0] = vp_param_2.iterate(n, lb=0, ub=0)

    all_res = []
    for k in range(1, K + 1):
        print(k)
        z[k] = vp_param_2.satlin(z[k - 1] - t * (P @ z[k - 1] + q_param), l=l_param, u=u_param)
        vp_param_2.set_objective_norm(z[k] - z[k - 1], np.inf)
        res = vp_param_2.solve()
        all_res.append(res)

        l_test = vp_param_2.get_solution_value(l_param)
        u_test = vp_param_2.get_solution_value(u_param)
        print(l_test, u_test)

        assert l_test[0] == u_test[0]

    print(all_res)


def test_satlin_sdp_mip(vp_param_2, vp_sdp):
    """Test that the SDP relaxation upper-bounds the MIP for constant saturation bounds."""
    np.random.seed(8)

    n = 3
    P = np.eye(n)
    q = np.zeros(n)
    l = np.array([-1, 1, -0.1])
    u = np.array([-0.9, 1.01, 0.1])

    q_offset = 1
    t = 0.01
    z0_init = (l + u) / 2
    K = 3

    sdp_solver_name = vp_sdp.canonicalizer.solver_name
    mip_solver_name = type(vp_param_2.canonicalizer.solver).__name__
    mip_solver_name = {"GurobiSolver": "gurobi", "HiGHSSolver": "highs"}[mip_solver_name]

    # --- SDP ---
    vp_s = Verifier(method="sdp", solver=sdp_solver_name, obbt_solver="highs")
    q_s = vp_s.parameter(n, lb=q - q_offset, ub=q + q_offset)
    z_s = [vp_s.iterate(n, lb=z0_init, ub=z0_init)]
    for k in range(K):
        z_s.append(vp_s.satlin(z_s[k] - t * (P @ z_s[k] + q_s), l=l, u=u))
    vp_s.set_objective_norm(z_s[K] - z_s[K - 1], 2)
    sdp_res = vp_s.solve()

    # --- MIP ---
    vp_m = Verifier(
        solver=mip_solver_name,
        obbt_solver=mip_solver_name,
        params=mip_params(mip_solver_name, limited=True),
    )
    q_m = vp_m.parameter(n, lb=q - q_offset, ub=q + q_offset)
    z_m = [vp_m.iterate(n, lb=z0_init, ub=z0_init)]
    for k in range(K):
        z_m.append(vp_m.satlin(z_m[k] - t * (P @ z_m[k] + q_m), l=l, u=u))
    vp_m.set_objective_norm(z_m[K] - z_m[K - 1], np.inf)
    mip_res = vp_m.solve()

    # SDP two-norm squared >= MIP inf-norm squared (relaxation upper bounds)
    assert sdp_res >= mip_res**2 - TEST_EPS


def test_satlin_param_sdp_mip(vp_param_2, vp_sdp):
    """Test SDP relaxation upper-bounds MIP for parameter saturation bounds with shared params."""
    np.random.seed(8)

    n = 3
    P = np.eye(n)
    q = np.ones(n)
    q_offset = 0.2
    t = 0.1
    K = 5

    sdp_solver_name = vp_sdp.canonicalizer.solver_name
    mip_solver_name = type(vp_param_2.canonicalizer.solver).__name__
    mip_solver_name = {"GurobiSolver": "gurobi", "HiGHSSolver": "highs"}[mip_solver_name]

    # --- SDP ---
    vp_s = Verifier(method="sdp", solver=sdp_solver_name, obbt_solver="highs")
    q_s = vp_s.parameter(n, lb=q - q_offset, ub=q + q_offset)
    top_s = vp_s.parameter(1, lb=0.5, ub=0.6)
    l_rest_s = vp_s.parameter(n - 1, lb=np.array([-1.2, -0.35]), ub=np.array([-1.1, 0]))
    u_rest_s = vp_s.parameter(n - 1, lb=np.array([0, 0.1]), ub=np.array([0.1, 0.2]))
    l_s = vp_s.add_param_stack([top_s, l_rest_s])
    u_s = vp_s.add_param_stack([top_s, u_rest_s])

    z_s = [vp_s.iterate(n, lb=0, ub=0)]
    for k in range(K):
        z_s.append(vp_s.satlin(z_s[k] - t * (P @ z_s[k] + q_s), l=l_s, u=u_s))
    vp_s.set_objective_norm(z_s[K] - z_s[K - 1], 2)
    sdp_res = vp_s.solve()

    # --- MIP ---
    vp_m = Verifier(
        solver=mip_solver_name,
        obbt_solver=mip_solver_name,
        params=mip_params(mip_solver_name, limited=True),
    )
    q_m = vp_m.parameter(n, lb=q - q_offset, ub=q + q_offset)
    top_m = vp_m.parameter(1, lb=0.5, ub=0.6)
    l_rest_m = vp_m.parameter(n - 1, lb=np.array([-1.2, -0.35]), ub=np.array([-1.1, 0]))
    u_rest_m = vp_m.parameter(n - 1, lb=np.array([0, 0.1]), ub=np.array([0.1, 0.2]))
    l_m = vp_m.add_param_stack([top_m, l_rest_m])
    u_m = vp_m.add_param_stack([top_m, u_rest_m])

    z_m = [vp_m.iterate(n, lb=0, ub=0)]
    for k in range(K):
        z_m.append(vp_m.satlin(z_m[k] - t * (P @ z_m[k] + q_m), l=l_m, u=u_m))
    vp_m.set_objective_norm(z_m[K] - z_m[K - 1], np.inf)
    mip_res = vp_m.solve()

    # SDP two-norm squared >= MIP inf-norm squared
    assert sdp_res >= mip_res**2 - TEST_EPS


def test_satlin_diamond(vp_param_2):
    """
    Diamond case: input bounds [lb, ub] strictly contain saturation region [l, u],
    with a verification objective whose worst case sits at a diamond corner that
    is NOT on the buggy upper-left diagonal.

    Setup: l=0, u=1 (asymmetric), z0=0, q in [-3, 3] so the satlin input arg = -q
    spans [-3, 3], clearly containing [0, 1]. The objective ||z1 - q||_inf equals
    |output + arg| with z0=0, and is maximized over the true diamond at the corner
    (arg, output) = (3, 1) -> value 4 (achieved at q=-3, z1=clip(3,0,1)=1).

    Under the previously buggy diamond cut (lines 645-647), feasibility collapsed
    onto the line through (lb, l)=(-3, 0) and (u, u)=(1, 1), so the (3, 1) corner
    was excluded and the MIP returned 3 instead of 4.
    """
    n = 1
    l = np.array([0.0])
    u = np.array([1.0])

    q_param = vp_param_2.parameter(n, lb=np.array([-3.0]), ub=np.array([3.0]))
    z0 = vp_param_2.iterate(n, lb=np.array([0.0]), ub=np.array([0.0]))
    z1 = vp_param_2.satlin(z0 - q_param, l=l, u=u)

    vp_param_2.set_objective_norm(z1 - q_param, np.inf)
    mip_res = vp_param_2.solve()

    true_worst_case = 4.0
    assert mip_res >= true_worst_case - TEST_EPS


def proj(v, l, u):
    return np.minimum(np.maximum(v, l), u)
