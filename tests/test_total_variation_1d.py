"""Tests for TotalVariation1D: numeric prox (fused lasso) and the KKT decomposer."""

import cvxpy as cp
import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import TotalVariation1D
from fomtools.proxable_functions.total_variation_1d import diff_matrix
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


def _tv_prox_dual(v, tau):
    """Independent reference: TV prox via the dual box-QP prox = v - Dᵀu*."""
    n = len(v)
    D = diff_matrix(n)
    u = cp.Variable(n - 1)
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(v - D.T @ u)), [cp.norm_inf(u) <= tau]).solve()
    return v - D.T @ u.value


# --- numeric prox -----------------------------------------------------------


def test_constructor_rejects_negative_lambd():
    with pytest.raises(ValueError):
        TotalVariation1D(-1.0)


def test_eval():
    assert abs(TotalVariation1D(2.0).eval(np.array([0.0, 1.0, 1.0, 3.0])) - 2.0 * 3.0) < 1e-9


def test_prox_constant_signal_identity():
    v = np.full(5, 0.7)
    np.testing.assert_allclose(TotalVariation1D(0.8).prox(v), v, atol=PROX_ATOL)


def test_prox_scalar_identity():
    """Length < 2 -> zero TV -> identity, with a UserWarning."""
    with pytest.warns(UserWarning, match="total variation is identically"):
        out = TotalVariation1D(1.0).prox(np.array([2.5]))
    np.testing.assert_allclose(out, np.array([2.5]))


def test_decomposer_warns_on_short_signal():
    """The verification lowering also warns (and is the identity) for length < 2."""
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(1, lb=np.array([2.5]), ub=np.array([2.5]))
    with pytest.warns(UserWarning, match="total variation is identically"):
        out = v.prox(TotalVariation1D(1.0), x)
    v.set_objective_norm([out], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(out), np.array([2.5]), atol=PROX_ATOL)


def test_prox_matches_dual_box_qp():
    """Primal cvxpy prox == v - Dᵀu* (independent box-QP); also TV(prox) <= TV(v)."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        v = rng.normal(scale=2.0, size=6)
        lam = rng.uniform(0.2, 1.5)
        g = rng.uniform(0.3, 2.0)
        p = TotalVariation1D(lam).prox(v, gamma=g)
        np.testing.assert_allclose(p, _tv_prox_dual(v, g * lam), atol=PROX_ATOL)
        assert np.sum(np.abs(np.diff(p))) <= np.sum(np.abs(np.diff(v))) + PROX_ATOL


# --- decomposer / verification ----------------------------------------------


def test_returns_linear_expression_on_mip():
    v = _make_verifier("mip", obbt="gurobi")
    rhs = v.add_param(4, lb=-2.0, ub=2.0)
    out = v.prox(TotalVariation1D(0.5), rhs)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 4


@pytest.mark.parametrize(
    "x0, lam, g",
    [
        (np.array([0.0, 0.1, 0.9, 1.0]), 0.5, 1.0),  # noisy step -> 2-level
        (np.array([1.0, -2.0, 0.5, 1.5, 0.0]), 0.3, 1.4),
        (np.full(4, 0.7), 0.8, 1.0),  # constant -> identity
    ],
)
def test_mip_matches_numeric(x0, lam, g):
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(len(x0), lb=x0, ub=x0)  # fixed rhs
    out = v.prox(TotalVariation1D(lam), x, gamma=g)
    v.set_objective_norm([out], np.inf)
    v.solve()
    np.testing.assert_allclose(
        v.get_solution_value(out), TotalVariation1D(lam).prox(x0, gamma=g), atol=PROX_ATOL
    )


def test_param_lambd():
    """lambd as a dim-1 LinearExpression parameter lowers correctly."""
    x0 = np.array([0.0, 0.1, 0.9, 1.0])
    lam0 = 0.5
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(4, lb=x0, ub=x0)
    lam_param = v.add_param(1, lb=np.array([lam0]), ub=np.array([lam0]))
    out = v.prox(TotalVariation1D(lam_param), x)
    v.set_objective_norm([out], np.inf)
    v.solve()
    np.testing.assert_allclose(
        v.get_solution_value(out), TotalVariation1D(lam0).prox(x0), atol=PROX_ATOL
    )


@pytest.mark.parametrize(
    "x0, lam", [(np.array([0.0, 0.1, 0.9, 1.0]), 0.5), (np.array([1.0, -2.0, 0.5, 1.5, 0.0]), 0.3)]
)
def test_sdp_valid_bound_at_fixed_point(x0, lam):
    """SDP gives a valid (upper) bound on ||prox(x0)||^2 (relaxation not always tight)."""
    v = _make_verifier("sdp")
    x = v.add_param(len(x0), lb=x0, ub=x0)
    out = v.prox(TotalVariation1D(lam), x)
    v.set_objective_norm(out, 2)
    obj = v.solve()
    proj = TotalVariation1D(lam).prox(x0)
    assert obj >= float(proj @ proj) - TEST_EPS


def test_sdp_bound_vs_mip_over_box():
    """SDP relaxation is a valid bound on the MIP over a box of rhs."""
    rhs_lb, rhs_ub = -2.0 * np.ones(4), 2.0 * np.ones(4)
    lam = 0.5

    vm = _make_verifier("mip", obbt="gurobi")
    xm = vm.add_param(4, lb=rhs_lb, ub=rhs_ub)
    om = vm.prox(TotalVariation1D(lam), xm)
    vm.set_objective_norm([om], np.inf)
    mip_inf = vm.solve()

    vs = _make_verifier("sdp")
    xs = vs.add_param(4, lb=rhs_lb, ub=rhs_ub)
    os_ = vs.prox(TotalVariation1D(lam), xs)
    vs.set_objective_norm(os_, 2)
    sdp_two = np.sqrt(vs.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS


def test_simulator_not_supported():
    v = _make_verifier("simulator")
    rhs = v.add_param(4, lb=-1.0, ub=1.0)
    with pytest.raises(NotImplementedError):
        v.prox(TotalVariation1D(0.5), rhs)
