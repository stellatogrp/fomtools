import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from tests.conftest import TEST_EPS
from fomtools.verifier import Verifier


def test_sdp_relu_bound():
    """Test SDP solver bound propagation"""
    vp = Verifier(method='sdp', solver='clarabel')
    m, n = 3, 2

    np.random.seed(0)
    A = np.random.normal(size=(m, n))

    z0 = vp.add_iterate(n, lb=5.0, ub=5.1)
    z1 = vp.step.relu(A@z0) - vp.step.relu(A@z0 - 10*np.ones(m))

    assert np.all(np.array(z1.ub) - np.array(z1.lb) >= -TEST_EPS)
