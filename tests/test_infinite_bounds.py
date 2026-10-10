"""Regression tests for unbounded leaves (issue #123, item 3).

Bound propagation must return +/-inf intervals instead of NaN, the OBBT ReLU
relaxation must not write NaN coefficients, and the exact MIP must refuse a
sign-changing ReLU input with an infinite bound instead of passing NaN or inf
to the solver. The same holds for soft-threshold and saturated linear, and the
MIP convex-hull cut callback ignores infinite bounds on leaves outside the row.
"""

import warnings

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.utils.convex_hull_cuts import separate_relu
from fomtools.utils.envelopes import relu_upper_envelope, softthreshold_envelope
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
    assert len(relu_upper_envelope(out, arg, -1.0, 2.0)) == 1
    assert len(relu_upper_envelope(out, arg, -1.0, np.inf)) == 1  # output <= arg - lb
    assert len(relu_upper_envelope(out, arg, -np.inf, 2.0)) == 1  # output <= ub
    assert relu_upper_envelope(out, arg, -np.inf, np.inf) == []
    for c in relu_upper_envelope(out, arg, -1.0, np.inf) + relu_upper_envelope(
        out, arg, -np.inf, 2.0
    ):
        assert np.all(np.isfinite(c.lhs.const))
        for M in c.lhs.decomposition_dict.values():
            M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
            assert np.all(np.isfinite(M))


# --- Soft-threshold, saturated linear, SDP and cut callback (follow-up to the ReLU fix) ---

try:
    import gurobipy  # noqa: F401

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

# Gurobi rejects NaN coefficients when a row is added, which is the original failure.
SOLVERS = [
    "highs",
    pytest.param("gurobi", marks=pytest.mark.skipif(not _HAS_GUROBI, reason="gurobipy")),
]
LAMBD = 0.1
# Mixed soft-threshold input ranges with an infinite bound: two-sided (both, ub, lb),
# right-or-middle with ub = inf, left-or-middle with lb = -inf.
SOFT_LB = np.array([-np.inf, -1.0, -np.inf, 0.0, -np.inf])
SOFT_UB = np.array([np.inf, np.inf, 1.0, np.inf, 0.0])
# Mixed satlin ranges for [l, u] = [-0.5, 0.5]: lb < l < u < ub (both, ub, lb), lower-or-middle,
# upper-or-middle.
SATLIN_LB = np.array([-np.inf, -1.0, -np.inf, -np.inf, 0.0])
SATLIN_UB = np.array([np.inf, np.inf, 1.0, 0.0, np.inf])


def _assert_finite_rows(constraints):
    for c in constraints:
        assert np.all(np.isfinite(c.lhs.const))
        for M in c.lhs.decomposition_dict.values():
            M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
            assert np.all(np.isfinite(M))


@pytest.fixture
def recorded_rows(monkeypatch):
    """Every linear row sent to the MIP model or the OBBT LP."""
    from fomtools.canonicalizers.mip import MIP
    from fomtools.canonicalizers.obbt_handler import OBBTHandler

    rows = []
    for cls in (MIP, OBBTHandler):
        original = cls.add_constraints

        def spy(self, constraints, _original=original):
            constraints = list(constraints)
            rows.extend(c for c in constraints if hasattr(c, "lhs"))
            return _original(self, constraints)

        monkeypatch.setattr(cls, "add_constraints", spy)
    return rows


def _row_value(c, values):
    total = np.asarray(c.lhs.const, dtype=float).reshape(-1).copy()
    for leaf, M in c.lhs.decomposition_dict.items():
        M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        total = total + M @ np.atleast_1d(values[leaf])
    return total


def _soft(a):
    return np.sign(a) * np.maximum(np.abs(a) - LAMBD, 0.0)


@pytest.mark.parametrize("lb, ub", list(zip(SOFT_LB, SOFT_UB)))
def test_softthreshold_envelope_limits_are_finite_and_valid(lb, ub):
    out = LinearExpression.from_int(1)
    arg = LinearExpression.from_int(1)
    rows = softthreshold_envelope(out, arg, lb, ub, _soft(lb), _soft(ub), LAMBD)
    assert len(rows) == 3 if np.isfinite(lb) or np.isfinite(ub) else len(rows) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assert_finite_rows(rows)
    # Each row holds on the graph (a, T(a)) over the input range.
    for a in np.linspace(max(lb, -50.0), min(ub, 50.0), 201):
        values = {out: _soft(a), arg: a}
        for c in rows:
            v = _row_value(c, values)
            if c.operator == "<=":
                assert np.all(v <= 1e-9)
            elif c.operator == ">=":
                assert np.all(v >= -1e-9)
            else:
                assert np.allclose(v, 0.0)


def test_softthreshold_envelope_finite_unchanged():
    out = LinearExpression.from_int(1)
    arg = LinearExpression.from_int(1)
    for lb, ub in [(-1.0, 2.0), (0.0, 2.0), (-2.0, 0.0)]:
        assert len(softthreshold_envelope(out, arg, lb, ub, _soft(lb), _soft(ub), LAMBD)) in (3, 4)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("i", range(len(SOFT_LB)))
def test_softthreshold_exact_mip_refuses_unbounded_input(solver, i):
    vp = Verifier(solver=solver)
    x = vp.add_iterate(ndim=1, lb=SOFT_LB[i : i + 1], ub=SOFT_UB[i : i + 1])
    with pytest.raises(ValueError, match="finite bounds"):
        vp.step.soft_threshold(x, lambd=LAMBD)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("i", range(len(SATLIN_LB)))
def test_satlin_exact_mip_refuses_unbounded_input(solver, i):
    vp = Verifier(solver=solver)
    x = vp.add_iterate(ndim=1, lb=SATLIN_LB[i : i + 1], ub=SATLIN_UB[i : i + 1])
    with pytest.raises(ValueError, match="finite bounds"):
        vp.step.satlin(x, l=-0.5 * np.ones(1), u=0.5 * np.ones(1))


@pytest.mark.parametrize("solver", SOLVERS)
def test_softthreshold_relaxed_builds_without_nan(solver, recorded_rows):
    """Relaxed MIP and the OBBT LP get the limit rows, never NaN."""
    vp = Verifier(solver=solver)
    x = vp.add_iterate(ndim=len(SOFT_LB), lb=SOFT_LB, ub=SOFT_UB)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vp.step.soft_threshold(x, lambd=LAMBD, relax_binary_vars=True)
    assert recorded_rows
    _assert_finite_rows(recorded_rows)


@pytest.mark.parametrize("solver", SOLVERS)
def test_satlin_relaxed_builds_without_nan(solver, recorded_rows):
    """Satlin chord slopes go to 0 with an infinite bound: finite rows in the MIP and OBBT LP."""
    vp = Verifier(solver=solver)
    x = vp.add_iterate(ndim=len(SATLIN_LB), lb=SATLIN_LB, ub=SATLIN_UB)
    n = len(SATLIN_LB)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vp.step.satlin(x, l=-0.5 * np.ones(n), u=0.5 * np.ones(n), relax_binary_vars=True)
    assert recorded_rows
    _assert_finite_rows(recorded_rows)


def test_conv_cuts_ignore_infinite_leaf_with_zero_coefficient():
    a = np.array([1.0, -2.0])
    y = np.array([0.0, 0.0])
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    # w = 2 is above the triangle chord (1.5) at y = 0, so some pair cut is violated.
    expected = separate_relu(a, 0.0, lo, hi, y, 2.0)
    assert expected is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = separate_relu(
            np.append(a, 0.0),
            0.0,
            np.append(lo, -np.inf),
            np.append(hi, np.inf),
            np.append(y, 3.0),
            2.0,
        )
    assert got is not None
    assert np.all(np.isfinite(got.alpha)) and np.isfinite(got.beta)
    np.testing.assert_allclose(got.alpha[:-1], expected.alpha)
    assert got.alpha[-1] == 0.0
    assert got.beta == pytest.approx(expected.beta)


def test_nan_bounds_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        LinearExpression.from_int(2, lb=np.nan)
    x = LinearExpression.from_int(2, lb=-1.0, ub=1.0)
    with pytest.raises(ValueError, match="NaN"):
        x.update_bounds(np.array([np.nan, 0.0]), np.ones(2))
    x.lb = np.array([np.nan, 0.0])  # bypasses the setter; propagation still catches it
    with pytest.raises(ValueError, match="NaN"):
        (2 * x).bound_propagation()
