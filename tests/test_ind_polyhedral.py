"""Tests for IndPolyhedral: numeric prox and the MIP KKT decomposer wiring."""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import IndPolyhedral
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    if method == "mip":
        return Verifier(method="mip", solver="gurobi", obbt_solver=obbt, params={"OutputFlag": 0})
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_solver=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_solver=None)
    raise ValueError(f"unknown method: {method}")


def _box(n, lb, ub):
    """Return (A, b) describing the box {lb <= x <= ub} as A x <= b."""
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([ub, -lb])
    return A, b


# --- numeric prox -----------------------------------------------------------


def test_prox_matches_clip_for_box():
    """Projection onto a box equals elementwise clip."""
    n = 4
    lb, ub = -np.ones(n), 2.0 * np.ones(n)
    A, b = _box(n, lb, ub)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(scale=3.0, size=n)
        np.testing.assert_allclose(IndPolyhedral(A, b).prox(x), np.clip(x, lb, ub), atol=1e-6)


def test_prox_halfspace_closed_form():
    """Projection onto a single half-space matches the closed form."""
    a = np.array([1.0, 2.0, -1.0])
    b0 = 0.5
    A, b = a.reshape(1, -1), np.array([b0])
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.normal(size=3)
        expected = x - max(0.0, (a @ x - b0)) / (a @ a) * a
        np.testing.assert_allclose(IndPolyhedral(A, b).prox(x), expected, atol=1e-6)


def test_constructor_dim_mismatch():
    """b dimension must match the number of rows of A."""
    with pytest.raises(ValueError):
        IndPolyhedral(np.ones((3, 2)), np.ones(2))


# --- decomposer wiring ------------------------------------------------------


def test_returns_linear_expression_on_mip():
    """prox_indpolyhedral returns the primal u as a LinearExpression of dim n."""
    v = _make_verifier("mip", obbt="gurobi")
    rhs = v.add_param(2, lb=-1.0, ub=1.0)
    A = np.array([[1.0, 1.0]])
    out = v.prox(IndPolyhedral(A, np.array([0.5]), dual_ub=10.0), rhs)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 2


@pytest.mark.parametrize(
    "A, b, x0, dual_ub",
    [
        (np.array([[1.0, 1.0]]), np.array([0.5]), np.array([1.0, 1.0]), 10.0),  # half-space
        (*_box(2, -np.ones(2), np.ones(2)), np.array([2.0, -3.0]), 10.0),  # box
        (
            np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
            np.array([1.0, 1.0, -0.5]),
            np.array([3.0, 3.0]),
            20.0,
        ),  # triangle
    ],
)
def test_mip_projection_matches_numeric(A, b, x0, dual_ub):
    """The MIP KKT encoding reproduces the true projection at a fixed rhs."""
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(len(x0), lb=x0, ub=x0)  # fixed rhs
    u = v.prox(IndPolyhedral(A, b, dual_ub=dual_ub), x)
    v.set_objective_norm([u], np.inf)
    v.solve()
    got = v.get_solution_value(u)
    np.testing.assert_allclose(got, IndPolyhedral(A, b).prox(x0), atol=1e-4)


def test_mip_without_dual_ub_raises():
    """Without a finite dual bound the big-M complementarity cannot be built."""
    A, b = _box(2, -np.ones(2), np.ones(2))
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(2, lb=np.array([2.0, -3.0]), ub=np.array([2.0, -3.0]))
    with pytest.raises(ValueError, match="finite upper bounds"):
        v.prox(IndPolyhedral(A, b), x)  # dual_ub=None -> lambda unbounded


def test_simulator_not_supported():
    """IndPolyhedral is an implicit-iterate prox; the simulator defers it."""
    v = _make_verifier("simulator")
    rhs = v.add_param(2, lb=-1.0, ub=1.0)
    with pytest.raises(NotImplementedError):
        v.prox(IndPolyhedral(np.array([[1.0, 1.0]]), np.array([0.5]), dual_ub=10.0), rhs)


def test_sdp_complementarity_not_yet_supported():
    """SDP complementarity is planned for step 5; surface a clear error for now."""
    v = _make_verifier("sdp")
    rhs = v.add_param(2, lb=-1.0, ub=1.0)
    with pytest.raises(NotImplementedError, match="SDP"):
        v.prox(IndPolyhedral(np.array([[1.0, 1.0]]), np.array([0.5]), dual_ub=10.0), rhs)


def test_complementarity_composite_slack_bound():
    """Regression: complementarity big-M reads bound_propagation, so a composite
    second argument (b - A@u) yields finite M and does not spuriously raise."""
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(2, lb=np.array([0.0, 0.0]), ub=np.array([0.0, 0.0]))  # rhs=0, inside box
    A, b = _box(2, -np.ones(2), np.ones(2))
    u = v.prox(IndPolyhedral(A, b, dual_ub=5.0), x)
    v.set_objective_norm([u], np.inf)
    v.solve()
    # rhs=0 is interior; projection is the identity.
    np.testing.assert_allclose(v.get_solution_value(u), np.zeros(2), atol=TEST_EPS)
