import numpy as np
import scipy.sparse as spa

from tests.cvxpy import douglas_rachford as douglas_rachford_cvxpy

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_dr(vp):
    np.random.seed(5)

    n = 4
    m = 3

    Phalf = np.random.normal(size=(n, n))
    P = Phalf @ Phalf.T + 0.1 * np.eye(n)
    c = np.random.normal(size=n)

    print(np.linalg.eigvals(P))

    A = np.random.normal(size=(m, n))
    xtest = 1 / n * np.random.normal(size=n)
    b = A @ xtest

    douglas_rachford_cvxpy(n, A, b, m, P, c)

    print("testing with DR splitting:")
    M = np.block([[P, A.T], [-A, np.zeros((m, m))]])
    lhs = spa.csc_matrix(M) + spa.eye(m + n)
    lhs_factored = spa.linalg.factorized(lhs)
    q = np.hstack([c, b])

    zk = np.zeros(m + n)

    K = 1000
    for _ in range(K):
        u = spa.linalg.spsolve(lhs, zk - q)
        utilde = proj(2 * u - zk, n)
        zk = zk + utilde - u

    print("z from DR:", zk)

    c_offset = 0.1
    q_l = np.hstack([c - c_offset, b])
    q_u = np.hstack([c + c_offset, b])
    q_param = vp.add_param(n + m, lb=q_l, ub=q_u)

    z0_init = np.zeros(n + m)
    z0 = vp.add_iterate(m + n, lb=z0_init, ub=z0_init)
    K = 5
    u = [None for _ in range(K + 1)]
    utilde = [None for _ in range(K + 1)]
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_res = []

    for k in range(1, K + 1):
        print(f"-K = {k}-")
        u[k] = vp.step(z[k - 1] - q_param, lhs_mat=lhs.todense(),
                                       lhs_mat_factorization=lhs_factored, step='implicit_linear')
        utilde[k] = vp.step(2 * u[k] - z[k - 1], proj_ranges=(n, n + m), step="relu")
        z[k] = z[k - 1] + utilde[k] - u[k]

        vp.set_infinity_norm_objective(z[k] - z[k - 1])
        res = vp.solve()
        all_res.append(res)

    print(f"opt q_param at last K: {vp.extract_sol(q_param)}")
    print(f"vp_gurobi resids: {all_res}")

    zk = z0_init
    q = vp.extract_sol(q_param)
    PGD_resids = []

    for k in range(K):
        u = spa.linalg.spsolve(lhs, zk - q)
        utilde = proj(2 * u - zk, n)
        znew = zk + utilde - u
        PGD_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print("PGD resids with last K:", PGD_resids)


def proj(v, n):
    out = v.copy()
    out[n:] = np.maximum(out[n:], 0)
    return out
