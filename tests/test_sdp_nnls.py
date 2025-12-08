import numpy as np

from algoverify.verifier import Verifier


def test_nnls_sdp():
    vp = Verifier(method='sdp', solver='cvxpy_sdp', obbt_handler=None)
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp.add_param(m, lb=0, ub=0.1)
    z0 = vp.add_iterate(n, lb=10, ub=10.1)
    z0 - t * (A @ z0 - x)
    # TODO: z1 = z0 - t * (A @ z0 - x)
    # above should error out but instead it returns None? need to investigate

    z1 = vp.step.relu(z0 - t * A.T @ (A @ z0 - x))

    vp.set_two_norm_objective(z1 - z0)
