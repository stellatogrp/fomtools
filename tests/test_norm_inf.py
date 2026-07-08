"""Tests for NormInf: numeric prox (Moreau on IndBallL1) and the decomposer (MIP + SDP)."""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import IndBallL1, NormInf
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


# --- numeric prox -----------------------------------------------------------


def test_constructor_rejects_negative_lambd():
    with pytest.raises(ValueError):
        NormInf(-1.0)


def test_eval():
    assert abs(NormInf(2.0).eval(np.array([1.0, -3.0, 2.0])) - 6.0) < 1e-9


def test_prox_moreau_identity():
    """prox_{gamma*lambda*||.||_inf}(v) == v - proj_{l1 <= gamma*lambda}(v)."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        v = rng.normal(scale=2.0, size=5)
        lam = rng.uniform(0.2, 1.5)
        g = rng.uniform(0.3, 2.0)
        # Both sides are independent cvxpy solves, so the error budget is ~2x a single
        # solve; add a relative tolerance (cvxpy accuracy is relative) for large entries.
        np.testing.assert_allclose(
            NormInf(lam).prox(v, gamma=g),
            v - IndBallL1(g * lam).prox(v),
            rtol=PROX_ATOL,
            atol=PROX_ATOL,
        )


def test_prox_extremes():
    v = np.array([3.0, 0.0, 0.0])
    np.testing.assert_allclose(NormInf(100.0).prox(v), np.zeros(3), atol=PROX_ATOL)  # huge -> 0
    np.testing.assert_allclose(NormInf(0.0).prox(v), v, atol=PROX_ATOL)  # lambda 0 -> identity


# --- decomposer / verification ----------------------------------------------


def test_returns_linear_expression_on_mip():
    v = _make_verifier("mip", obbt="gurobi")
    arg = v.add_param(3, lb=-2.0, ub=2.0)
    out = v.prox(NormInf(1.0), arg)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


@pytest.mark.parametrize(
    "x0, lam, g",
    [
        (np.array([3.0, 0.0, 0.0]), 1.0, 1.0),
        (np.array([1.0, -2.0, 0.5, 1.5]), 0.7, 1.3),
    ],
)
def test_mip_matches_numeric(x0, lam, g):
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(len(x0), lb=x0, ub=x0)  # fixed arg
    out = v.prox(NormInf(lam), x, gamma=g)
    v.set_objective_norm([out], np.inf)
    v.solve()
    np.testing.assert_allclose(
        v.get_solution_value(out), NormInf(lam).prox(x0, gamma=g), atol=PROX_ATOL
    )


def test_param_lambd():
    """lambd as a dim-1 LinearExpression parameter lowers correctly."""
    x0 = np.array([3.0, 0.0, 0.0])
    lam0 = 1.0
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(3, lb=x0, ub=x0)
    lam_param = v.add_param(1, lb=np.array([lam0]), ub=np.array([lam0]))
    out = v.prox(NormInf(lam_param), x)
    v.set_objective_norm([out], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(out), NormInf(lam0).prox(x0), atol=PROX_ATOL)


@pytest.mark.parametrize(
    "x0, lam", [(np.array([3.0, 0.0, 0.0]), 1.0), (np.array([1.0, -2.0, 0.5, 1.5]), 0.7)]
)
def test_sdp_valid_bound_at_fixed_point(x0, lam):
    """SDP gives a valid (upper) bound on ||prox(x0)||^2.

    Unlike IndBallL1 (objective on u), maximizing ||arg - u||^2 leaves the Shor relaxation
    non-tight here, so we only assert validity: sdp_obj >= ||prox(x0)||^2.
    """
    v = _make_verifier("sdp")
    x = v.add_param(len(x0), lb=x0, ub=x0)
    out = v.prox(NormInf(lam), x)
    v.set_objective_norm(out, 2)
    obj = v.solve()
    proj = NormInf(lam).prox(x0)
    assert obj >= float(proj @ proj) - TEST_EPS


def test_sdp_bound_vs_mip_over_box():
    """SDP relaxation is a valid bound on the MIP over a box of arg."""
    arg_lb, arg_ub = -2.0 * np.ones(3), 2.0 * np.ones(3)
    lam = 1.0

    vm = _make_verifier("mip", obbt="gurobi")
    xm = vm.add_param(3, lb=arg_lb, ub=arg_ub)
    om = vm.prox(NormInf(lam), xm)
    vm.set_objective_norm([om], np.inf)
    mip_inf = vm.solve()

    vs = _make_verifier("sdp")
    xs = vs.add_param(3, lb=arg_lb, ub=arg_ub)
    os_ = vs.prox(NormInf(lam), xs)
    vs.set_objective_norm(os_, 2)
    sdp_two = np.sqrt(vs.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS


def test_dual_ub_stored_and_forwarded():
    """NormInf accepts dual_ub, stores it, and the lowering still matches the numeric prox.

    dual_ub overrides the auto-derived bound of the underlying ℓ1-ball projection.
    """
    assert NormInf(1.0, dual_ub=7.5).dual_ub == 7.5

    x0 = np.array([3.0, 0.0, 0.0])
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(3, lb=x0, ub=x0)
    out = v.prox(NormInf(1.0, dual_ub=10.0), x)  # explicit (valid) override
    v.set_objective_norm([out], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(out), NormInf(1.0).prox(x0), atol=PROX_ATOL)


def test_simulator_supported():
    """Simulator now evaluates the ℓ∞-norm prox via the batched Moreau / ℓ1-ball
    projection, returning a LinearExpression of the input dimension."""
    v = _make_verifier("simulator")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    out = v.prox(NormInf(1.0), arg)
    assert out.ndim == 3
