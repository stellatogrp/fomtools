"""Tests for IndBallL1: numeric prox and the lifted-KKT decomposer (MIP + SDP)."""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import IndBallL1
from fomtools.verifier import Verifier
from tests.conftest import PROX_ATOL, TEST_EPS


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    if method == "mip":
        return Verifier(method="mip", solver="gurobi", obbt_solver=obbt, params={"OutputFlag": 0})
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_solver=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_solver=None)
    raise ValueError(f"unknown method: {method}")


def _proj_l1_ref(x, r):
    """Reference ℓ1-ball projection via bisection on the soft-threshold level."""
    x = np.asarray(x, dtype=float)
    if np.sum(np.abs(x)) <= r:
        return x.copy()
    lo, hi = 0.0, float(np.max(np.abs(x)))
    for _ in range(200):
        th = 0.5 * (lo + hi)
        if np.sum(np.maximum(np.abs(x) - th, 0.0)) > r:
            lo = th
        else:
            hi = th
    th = 0.5 * (lo + hi)
    return np.sign(x) * np.maximum(np.abs(x) - th, 0.0)


# --- numeric prox -----------------------------------------------------------


def test_prox_inside_ball_is_identity():
    x = np.array([0.2, -0.1, 0.05])
    np.testing.assert_allclose(IndBallL1(1.0).prox(x), x, atol=PROX_ATOL)


def test_prox_matches_soft_threshold():
    rng = np.random.default_rng(0)
    r = 1.3
    for _ in range(8):
        x = rng.normal(scale=2.0, size=5)
        p = IndBallL1(r).prox(x)
        np.testing.assert_allclose(p, _proj_l1_ref(x, r), atol=PROX_ATOL)
        assert abs(np.sum(np.abs(p)) - r) < PROX_ATOL  # projection lands on the boundary


def test_constructor_rejects_negative_r():
    with pytest.raises(ValueError):
        IndBallL1(-1.0)


# --- decomposer / verification ----------------------------------------------


def test_returns_linear_expression_on_mip():
    v = _make_verifier("mip", obbt="gurobi")
    arg = v.parameter(3, lb=-2.0, ub=2.0)
    out = v.prox(IndBallL1(1.0), arg)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


@pytest.mark.parametrize(
    "x0, r",
    [
        (np.array([3.0, 0.0, 0.0]), 1.0),  # outside, sparse
        (np.array([0.2, -0.1, 0.05]), 1.0),  # inside -> identity
        (np.array([1.0, 1.0, 1.0, -1.0]), 1.5),  # outside, spread
    ],
)
def test_mip_projection_matches_numeric(x0, r):
    v = _make_verifier("mip", obbt="gurobi")
    x = v.parameter(len(x0), lb=x0, ub=x0)  # fixed arg
    u = v.prox(IndBallL1(r), x)  # dual_ub auto-derived
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), IndBallL1(r).prox(x0), atol=PROX_ATOL)


def test_param_radius():
    """r as a dim-1 LinearExpression parameter lowers correctly."""
    x0 = np.array([3.0, 0.0, 0.0])
    r0 = 1.0
    v = _make_verifier("mip", obbt="gurobi")
    x = v.parameter(3, lb=x0, ub=x0)
    r_param = v.parameter(1, lb=np.array([r0]), ub=np.array([r0]))
    u = v.prox(IndBallL1(r_param), x)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), IndBallL1(r0).prox(x0), atol=PROX_ATOL)


@pytest.mark.parametrize(
    "x0, r",
    [(np.array([3.0, 0.0, 0.0]), 1.0), (np.array([1.0, 1.0, 1.0, -1.0]), 1.5)],
)
def test_sdp_tight_at_fixed_point(x0, r):
    """At a fixed arg, max ||u||^2 over the relaxed KKT equals ||proj(x0)||^2."""
    v = _make_verifier("sdp")
    x = v.parameter(len(x0), lb=x0, ub=x0)
    u = v.prox(IndBallL1(r), x)
    v.set_objective_norm(u, 2)
    obj = v.solve()
    proj = IndBallL1(r).prox(x0)
    np.testing.assert_allclose(obj, float(proj @ proj), atol=1e-3)


def test_sdp_bound_vs_mip_over_box():
    """SDP relaxation is a valid bound on the MIP over a box of arg."""
    arg_lb, arg_ub = -2.0 * np.ones(3), 2.0 * np.ones(3)
    r = 1.0

    vm = _make_verifier("mip", obbt="gurobi")
    xm = vm.parameter(3, lb=arg_lb, ub=arg_ub)
    um = vm.prox(IndBallL1(r), xm)
    vm.set_objective_norm([um], np.inf)
    mip_inf = vm.solve()

    vs = _make_verifier("sdp")
    xs = vs.parameter(3, lb=arg_lb, ub=arg_ub)
    us = vs.prox(IndBallL1(r), xs)
    vs.set_objective_norm(us, 2)
    sdp_two = np.sqrt(vs.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS, f"MIP inf {mip_inf} should be <= SDP two {sdp_two}"


def test_simulator_supported():
    """Simulator now evaluates the ℓ1-ball projection with a batched solver-free
    algorithm, returning a LinearExpression of the input dimension."""
    v = _make_verifier("simulator")
    arg = v.parameter(3, lb=-1.0, ub=1.0)
    out = v.prox(IndBallL1(1.0), arg)
    assert out.ndim == 3
