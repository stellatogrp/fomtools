"""Tests for IndSimplex: numeric prox and the mixed-KKT decomposer (MIP + SDP)."""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import IndSimplex
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


def _proj_simplex_ref(x, a):
    """Reference simplex projection via bisection on the relu threshold τ."""
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x) - a), float(np.max(x))
    for _ in range(200):
        t = 0.5 * (lo + hi)
        if np.sum(np.maximum(x - t, 0.0)) > a:
            lo = t
        else:
            hi = t
    return np.maximum(x - 0.5 * (lo + hi), 0.0)


# --- numeric prox -----------------------------------------------------------


def test_constructor_rejects_nonpositive_a():
    with pytest.raises(ValueError):
        IndSimplex(0.0)
    with pytest.raises(ValueError):
        IndSimplex(-1.0)


def test_prox_on_simplex_is_identity():
    n, a = 4, 1.0
    x = np.full(n, a / n)
    np.testing.assert_allclose(IndSimplex(a).prox(x), x, atol=PROX_ATOL)


def test_prox_matches_relu_threshold():
    rng = np.random.default_rng(0)
    for _ in range(8):
        x = rng.normal(scale=2.0, size=5)
        a = rng.uniform(0.5, 2.0)
        p = IndSimplex(a).prox(x)
        np.testing.assert_allclose(p, _proj_simplex_ref(x, a), atol=PROX_ATOL)
        assert np.all(p >= -PROX_ATOL) and abs(np.sum(p) - a) < PROX_ATOL


# --- decomposer / verification ----------------------------------------------


def test_returns_linear_expression_on_mip():
    v = _make_verifier("mip", obbt="gurobi")
    rhs = v.add_param(3, lb=-2.0, ub=2.0)
    out = v.prox(IndSimplex(1.0), rhs)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


@pytest.mark.parametrize(
    "x0, a",
    [
        (np.array([3.0, 0.0, 0.0]), 1.0),  # outside, sparse projection
        (np.array([0.5, 0.3, 0.2]), 1.0),  # already on the simplex
        (np.array([1.0, 1.0, 1.0, -1.0]), 2.0),  # spread, a != 1
    ],
)
def test_mip_projection_matches_numeric(x0, a):
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(len(x0), lb=x0, ub=x0)  # fixed rhs
    u = v.prox(IndSimplex(a), x)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), IndSimplex(a).prox(x0), atol=PROX_ATOL)


def test_param_a():
    """a as a dim-1 LinearExpression parameter lowers correctly."""
    x0 = np.array([3.0, 0.0, 0.0])
    a0 = 1.0
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(3, lb=x0, ub=x0)
    a_param = v.add_param(1, lb=np.array([a0]), ub=np.array([a0]))
    u = v.prox(IndSimplex(a_param), x)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), IndSimplex(a0).prox(x0), atol=PROX_ATOL)


@pytest.mark.parametrize(
    "x0, a", [(np.array([3.0, 0.0, 0.0]), 1.0), (np.array([1.0, 1.0, 1.0, -1.0]), 2.0)]
)
def test_sdp_tight_at_fixed_point(x0, a):
    """At a fixed rhs, max ||u||^2 over the relaxed KKT equals ||proj(x0)||^2."""
    v = _make_verifier("sdp")
    x = v.add_param(len(x0), lb=x0, ub=x0)
    u = v.prox(IndSimplex(a), x)
    v.set_objective_norm(u, 2)
    obj = v.solve()
    proj = IndSimplex(a).prox(x0)
    np.testing.assert_allclose(obj, float(proj @ proj), atol=1e-3)


def test_sdp_bound_vs_mip_over_box():
    """SDP relaxation is a valid bound on the MIP over a box of rhs."""
    rhs_lb, rhs_ub = -2.0 * np.ones(3), 2.0 * np.ones(3)
    a = 1.0

    vm = _make_verifier("mip", obbt="gurobi")
    xm = vm.add_param(3, lb=rhs_lb, ub=rhs_ub)
    um = vm.prox(IndSimplex(a), xm)
    vm.set_objective_norm([um], np.inf)
    mip_inf = vm.solve()

    vs = _make_verifier("sdp")
    xs = vs.add_param(3, lb=rhs_lb, ub=rhs_ub)
    us = vs.prox(IndSimplex(a), xs)
    vs.set_objective_norm(us, 2)
    sdp_two = np.sqrt(vs.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS


def test_lam_ub_per_component():
    """The dual bound is per-component (hi - rhs_lb), tighter than the global hi - lo."""
    rhs_lb = np.array([-1.0, 0.0, 2.0])
    rhs_ub = np.array([1.0, 3.0, 5.0])
    v = _make_verifier("mip", obbt=None)
    x = v.add_param(3, lb=rhs_lb, ub=rhs_ub)
    v.prox(IndSimplex(1.0), x)

    hi, lo = float(np.max(rhs_ub)), float(np.min(rhs_lb))
    expected = hi - rhs_lb  # [6, 5, 3]
    # The lambda iterate is the dim-3 nonnegative dual registered before u; match by ub.
    lam_ubs = [it.ub for it in v.iterates if it.ndim == 3 and np.allclose(it.lb, 0.0)]
    assert any(np.allclose(ub, expected) for ub in lam_ubs), lam_ubs
    assert np.all(expected <= hi - lo + 1e-9)  # tighter than (or equal to) the old global bound


def test_simulator_not_supported():
    v = _make_verifier("simulator")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    with pytest.raises(NotImplementedError):
        v.prox(IndSimplex(1.0), rhs)
