import numpy as np

from tests.conftest import TEST_EPS
from tests.cvxpy import ista

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_ista(vp_theory):
    m, n = 4, 2

    np.random.seed(0)

    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))
    lambd = 1

    x = ista(n, A, b, lambd)
    # t = 0.1
    lambd_max = np.max(np.real(np.linalg.eigvals(A.T @ A)))
    print(lambd_max)
    t = 1 / lambd_max

    print("testing ista")
    xk = np.ones(n)
    I = np.eye(n)
    K = 1000
    At = I - t * A.T @ A
    Bt = t * A.T

    for _ in range(K):
        xk = soft_threshold(At @ xk + Bt @ b, t * lambd)

    print(xk)
    assert np.linalg.norm(x - xk) <= TEST_EPS
    b_offset = 1.0
    b_param = vp_theory.add_param(m, lb=b - b_offset, ub=b + b_offset)

    z0 = vp_theory.add_iterate(n, lb=2, ub=2)

    K = 5
    z = [None for _ in range(K + 1)]

    z[0] = z0

    lambda_t = lambd * t
    all_res = []
    for k in range(1, K + 1):
        print(k)

        z[k] = vp_theory.step.soft_threshold(At @ z[k - 1] + Bt @ b_param, lambd=lambda_t)
        vp_theory.theory_bound(k, z[k], z[k - 1])

        vp_theory.set_infinity_norm_objective(z[k] - z[k - 1])
        res = vp_theory.solve(convex_hull_cuts=True)
        all_res.append(res)

    print(f"opt b_param at last K: {vp_theory.get_solution_value(b_param)}")
    print(f"VP resids: {all_res}")

    zk = 2 * np.ones(n)
    b = vp_theory.get_solution_value(b_param)
    ISTA_resids = []

    for k in range(K):
        znew = soft_threshold(At @ zk + Bt @ b, t * lambd)
        ISTA_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print(f"ISTA resids: {ISTA_resids}")

    assert np.all(np.array(ISTA_resids) <= np.array(all_res) + TEST_EPS)
    assert np.abs(ISTA_resids[-1] - all_res[-1]) <= TEST_EPS


def soft_threshold(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)
