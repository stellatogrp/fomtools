import numpy as np

from tests.conftest import TEST_EPS
from tests.cvxpy import verifier as verifier_cvxpy


def test_verifier(vp):
    m, n = 2, 3
    np.random.seed(1)

    x_test = np.random.uniform(size=(n,))
    A = np.random.normal(size=(m, n))
    c = np.random.uniform(size=(n,))
    b = A @ x_test
    verifier_cvxpy(n, c, A, b, x_test)

    print("testing with DR")

    xk = np.ones(n)
    yk = np.ones(m)

    def proj(v):
        return np.maximum(v, 0)

    t = 0.1
    K = 10000
    DR_resids = []
    DR_x = [xk]
    DR_y = [yk]
    for _ in range(K):
        xnew = proj(xk - t * (c - A.T @ yk))
        ynew = yk - t * (A @ (2 * xnew - xk) - b)

        z_resid = np.hstack([xnew - xk, ynew - yk])
        DR_resids.append(np.max(np.abs(z_resid)))

        DR_x.append(xnew)
        DR_y.append(ynew)

        xk = xnew
        yk = ynew

    print(f"x from DR: {xk}")

    b_offset = 0.1
    c_param = vp.parameter(n, lb=c, ub=c)
    b_param = vp.parameter(m, lb=b - b_offset, ub=b + b_offset)  # add boxes here when testing

    x0 = vp.iterate(n, lb=1, ub=1)
    y0 = vp.iterate(m, lb=1, ub=1)

    K = 10

    x = [None for _ in range(K + 1)]
    y = [None for _ in range(K + 1)]

    x[0] = x0
    y[0] = y0

    all_res = []
    for k in range(1, K + 1):
        print(k)
        x[k] = vp.relu(x[k - 1] - t * (c_param - A.T @ y[k - 1]))
        y[k] = y[k - 1] - t * (A @ (2 * x[k] - x[k - 1]) - b_param)

        vp.set_objective_norm([x[k] - x[k - 1], y[k] - y[k - 1]], np.inf)
        res = vp.solve()
        all_res.append(res)

    print(f"VP resids: {all_res}")
    print(f"DR resids: {DR_resids[:K]}")

    print(f"b_c: {b}")
    print(vp.get_solution_value(b_param))
    b_test = vp.get_solution_value(b_param)

    xK = vp.get_solution_value(x[K])
    xKminus1 = vp.get_solution_value(x[K - 1])
    yK = vp.get_solution_value(y[K])
    yKminus1 = vp.get_solution_value(y[K - 1])

    test_resid = np.max(np.abs(np.hstack([xK - xKminus1, yK - yKminus1])))
    print(test_resid)

    xk = np.ones(n)
    yk = np.ones(m)

    t = 0.1
    DR_resids = []
    DR_x = [xk]
    DR_y = [yk]
    for _ in range(K):
        xnew = proj(xk - t * (c - A.T @ yk))
        ynew = yk - t * (A @ (2 * xnew - xk) - b_test)

        z_resid = np.hstack([xnew - xk, ynew - yk])
        DR_resids.append(np.max(np.abs(z_resid)))

        DR_x.append(xnew)
        DR_y.append(ynew)

        xk = xnew
        yk = ynew

    print(DR_resids)

    assert np.all(np.array(DR_resids) <= np.array(all_res) + TEST_EPS)
    assert np.abs(DR_resids[-1] - test_resid) <= TEST_EPS
