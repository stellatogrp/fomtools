import numpy as np
from tests.cvxpy import partial_relu as partial_relu_cvxpy

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation
np.set_printoptions(legacy="1.25")


def test_partial_nnqp(vp_gurobi):
    np.random.seed(0)
    n = 3
    proj_ranges = (1, 3)

    P = np.random.normal(size=(n, n))
    P = P @ P.T
    q = np.random.normal(size=(n,))

    K = 1000
    t = 0.01
    zk = -np.ones(n)
    for _ in range(K):
        yk = zk - t * (P @ zk + q)
        zk = partial_relu(yk, proj_ranges)

    print("solution from pgd:", zk)
    x = partial_relu_cvxpy(n, proj_ranges, P, q)
    assert np.linalg.norm(x - zk) <= 1e-7

    q_offset = 1
    q_param = vp_gurobi.add_param(n, lb=q - q_offset, ub=q + q_offset)

    z0 = vp_gurobi.initial_iterate(n, lb=-1, ub=-1)
    K = 10
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_res = []
    for k in range(1, K + 1):
        print(k)
        z[k] = vp_gurobi.relu_step(z[k - 1] - t * (P @ z[k - 1] + q_param), proj_ranges=proj_ranges)

        vp_gurobi.set_infinity_norm_objective(z[k] - z[k - 1])
        res = vp_gurobi.solve(huchette_cuts=True)
        all_res.append(res)
        vp_gurobi.post_process(z[k], z[0], all_res)

    print(f"opt q_param at last K: {vp_gurobi.extract_sol(q_param)}")

    print(f"vp_gurobi resids: {all_res}")

    zk = -np.ones(n)
    q = vp_gurobi.extract_sol(q_param)
    PGD_resids = []

    for k in range(K):
        znew = partial_relu(zk - t * (P @ zk + q), proj_ranges)
        PGD_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print(PGD_resids)

    assert np.all(np.array(PGD_resids) <= np.array(all_res) + 1e-7)
    assert np.abs(PGD_resids[-1] - all_res[-1]) <= 1e-8


def partial_relu(x, proj_ranges):
    out = x.copy()
    out[proj_ranges[0] : proj_ranges[1]] = np.maximum(x[proj_ranges[0] : proj_ranges[1]], 0)
    return out
