import importlib.util

import numpy as np

from fomtools.verifier import Verifier


def test_nnls_sdp():
    """Test SDP solver dispatch through Verifier.solve() API."""
    vp = Verifier(method='sdp', solver='clarabel', obbt_handler=None)
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp.add_param(m, lb=0, ub=0.1)
    z0 = vp.add_iterate(n, lb=5, ub=5.1)

    z1 = vp.step.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp.step.relu(z1 - t * A.T @ (A @ z1 - x))

    vp.set_two_norm_objective(z2 - z1)

    # Test standard API
    result = vp.solve()
    assert result is not None

    # Also verify direct solver calls still work
    vp.canonicalizer.solve_via_cvxpy()
    vp.canonicalizer.solve_via_clarabel()
    if importlib.util.find_spec('cupy') is not None:
        vp.canonicalizer.solve_via_cuclarabel()
