"""Regression tests for unbounded leaves (issue #123, item 3).

Bound propagation must return +/-inf intervals instead of NaN, the OBBT ReLU
relaxation must not write NaN coefficients, and the exact MIP must refuse a
sign-changing ReLU input with an infinite bound instead of passing NaN or inf
to the solver.
"""

import warnings

import numpy as np
import pytest

from fomtools.canonicalizers.obbt_handler import _relu_upper_envelope
from fomtools.linearexpression import LinearExpression
from fomtools.verifier import Verifier


def _issue_123_iterate(solver):
    vp = Verifier(solver=solver)
    x = vp.add_iterate(ndim=2, lb=np.array([-np.inf, 0.0]), ub=np.array([np.inf, 1.0]))
    return vp, x


@pytest.mark.parametrize("solver", ["highs"])
def test_issue_123_bound_propagation(solver):
    _, x = _issue_123_iterate(solver)
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lb, ub = (A @ x).bound_propagation()
    np.testing.assert_array_equal(lb, [0.0, -np.inf])
    np.testing.assert_array_equal(ub, [1.0, np.inf])


@pytest.mark.parametrize("solver", ["highs"])
def test_issue_123_relu_on_unbounded_input_fails_clearly(solver):
    vp, x = _issue_123_iterate(solver)
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="finite bounds"):
        vp.step.relu(A @ x)


def test_relu_upper_envelope_limits():
    out = LinearExpression.from_int(1)
    arg = LinearExpression.from_int(1)
    assert len(_relu_upper_envelope(out, arg, -1.0, 2.0)) == 1
    assert len(_relu_upper_envelope(out, arg, -1.0, np.inf)) == 1  # output <= arg - lb
    assert len(_relu_upper_envelope(out, arg, -np.inf, 2.0)) == 1  # output <= ub
    assert _relu_upper_envelope(out, arg, -np.inf, np.inf) == []
    for c in _relu_upper_envelope(out, arg, -1.0, np.inf) + _relu_upper_envelope(
        out, arg, -np.inf, 2.0
    ):
        assert np.all(np.isfinite(c.lhs.const))
        for M in c.lhs.decomposition_dict.values():
            M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
            assert np.all(np.isfinite(M))
