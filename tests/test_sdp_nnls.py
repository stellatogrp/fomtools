import numpy as np

from algoverify.verifier import Verifier


def test_nnls_sdp():
    vp = Verifier(method='sdp', solver='cvxpy_sdp', obbt_handler=None)
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp.add_param(m, lb=0, ub=0.1)
    z0 = vp.add_iterate(n, lb=5, ub=5.1)

    z1 = vp.step.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp.step.relu(z1 - t * A.T @ (A @ z1 - x))

    vp.set_two_norm_objective(z2 - z1)

    # print(x.lb, x.ub)
    # print(z0.lb, z0.ub)
    # print(z1.lb, z1.ub)

    vp.canonicalizer.solve_via_cvxpy()
    # print(vp.canonicalizer.psd_cone_indices)
