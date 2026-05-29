"""
Integration tests for the proxable-function decomposer layer.

Covers the contract between ``Verifier.prox`` and
``StepDecomposer.process_prox`` / ``_PROX_DECOMPOSERS``:

- cross-backend inheritance of base-class registrations,
- recursive interaction with the step decomposer registry, and
- the ``NotImplementedError`` path for proxes not registered on a
  given backend.

Numerical correctness of individual prox math (e.g. ``NormL1.prox``
returning the right soft-threshold output) is the concern of
``tests/test_prox_eval.py``, not this file.
"""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import (
    ElasticNet,
    IndBallLinf,
    IndBox,
    IndNonNegative,
    IndNonPositive,
    IndPoint,
    NormL1,
    ProxableFunction,
)
from fomtools.steps import Relu, SaturatedLinear, SoftThreshold
from fomtools.verifier import Verifier


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    """Construct a Verifier for the given backend with sensible defaults."""
    if method == "mip":
        return Verifier(
            method="mip",
            solver="gurobi",
            obbt_solver=obbt,
            params={"OutputFlag": 0},
        )
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_solver=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_solver=None)
    raise ValueError(f"unknown method: {method}")


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_norml1_returns_linear_expression(method):
    """NormL1 prox is registered on the base class; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(NormL1(lambd=0.3), rhs, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_norml1_emits_softthreshold_on_mip():
    """
    On MIP, NormL1 lowers to a single SoftThreshold primitive (no recursive
    decomposition — MIP encodes SoftThreshold natively).
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(NormL1(lambd=0.3), rhs, gamma=0.5)

    n_soft = sum(
        1 for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)
    )
    assert n_soft == 1
    # SoftThreshold's lambd should be the scaled value lambd * gamma = 0.15
    soft_step = next(
        s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)
    )
    assert np.isclose(soft_step.lambd, 0.15)


def test_norml1_decomposes_to_two_relus_on_sdp():
    """
    On SDP, the SoftThreshold emitted by prox_norml1 recursively decomposes
    into two ReLUs via the existing _DECOMPOSERS registry — confirms that
    the two registry layers (prox + step) compose correctly.
    """
    v = _make_verifier("sdp")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(NormL1(lambd=0.3), rhs, gamma=0.5)

    # SDP canonicalizer tracks Relus it encoded
    assert len(v.canonicalizer._relu_steps) == 2
    assert all(isinstance(s, Relu) for s in v.canonicalizer._relu_steps)


# --- ElasticNet ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_elasticnet_returns_linear_expression(method):
    """ElasticNet prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(ElasticNet(alpha=0.3, beta=0.5), rhs, gamma=0.4)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_elasticnet_emits_softthreshold_on_mip():
    """
    ElasticNet lowers to a single SoftThreshold step on MIP, with the
    threshold scaled by alpha*gamma (the post-softthreshold scalar
    shrinkage by 1/(1+2*beta*gamma) is an analytic LinearExpression op,
    not a separate step).
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(ElasticNet(alpha=0.3, beta=0.5), rhs, gamma=0.4)

    soft_steps = [
        s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)
    ]
    assert len(soft_steps) == 1
    # SoftThreshold's lambd should be alpha * gamma = 0.12
    assert np.isclose(soft_steps[0].lambd, 0.12)


def test_elasticnet_decomposes_to_two_relus_on_sdp():
    """
    On SDP, the SoftThreshold from prox_elasticnet recursively decomposes
    into two ReLUs via _DECOMPOSERS — same composition as NormL1.
    """
    v = _make_verifier("sdp")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(ElasticNet(alpha=0.3, beta=0.5), rhs, gamma=0.4)

    assert len(v.canonicalizer._relu_steps) == 2
    assert all(isinstance(s, Relu) for s in v.canonicalizer._relu_steps)


def test_elasticnet_reduces_to_norml1_when_beta_zero():
    """
    With beta = 0 the ridge-shrinkage scalar collapses to 1, so
    ElasticNet(alpha, 0) produces a SoftThreshold step identical to
    NormL1(alpha) at the same gamma.
    """
    alpha, gamma = 0.3, 0.4

    v_en = _make_verifier("mip")
    rhs_en = v_en.add_param(3, lb=-1.0, ub=1.0)
    v_en.prox(ElasticNet(alpha=alpha, beta=0.0), rhs_en, gamma=gamma)

    v_l1 = _make_verifier("mip")
    rhs_l1 = v_l1.add_param(3, lb=-1.0, ub=1.0)
    v_l1.prox(NormL1(lambd=alpha), rhs_l1, gamma=gamma)

    soft_en = [
        s for s in v_en.canonicalizer.steps if isinstance(s, SoftThreshold)
    ][0]
    soft_l1 = [
        s for s in v_l1.canonicalizer.steps if isinstance(s, SoftThreshold)
    ][0]
    assert np.isclose(soft_en.lambd, soft_l1.lambd)


def test_elasticnet_ridge_only_when_alpha_zero():
    """
    With alpha = 0 the prox is pure ridge shrinkage: rhs / (1 + 2*beta*gamma).
    Verified numerically on the math wrapper (the lowering still emits a
    SoftThreshold with lambd=0, which is the identity-shaped envelope).
    """
    beta, gamma = 0.5, 0.4
    rhs = np.array([-2.0, 0.05, 1.5])
    expected = rhs / (1.0 + 2.0 * beta * gamma)
    np.testing.assert_allclose(
        ElasticNet(alpha=0.0, beta=beta).prox(rhs, gamma), expected
    )


def test_elasticnet_prox_matches_closed_form():
    """
    Numerical math-layer sanity: ElasticNet.prox matches the closed-form
    S_{alpha*gamma}(rhs) / (1 + 2*beta*gamma) on a random rhs.
    """
    rng = np.random.default_rng(7)
    alpha, beta, gamma = 0.3, 0.5, 0.4
    rhs = rng.normal(size=8)
    thresh = alpha * gamma
    expected = (
        np.sign(rhs) * np.maximum(np.abs(rhs) - thresh, 0.0)
        / (1.0 + 2.0 * beta * gamma)
    )
    np.testing.assert_allclose(
        ElasticNet(alpha=alpha, beta=beta).prox(rhs, gamma), expected
    )


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indnonnegative_returns_linear_expression(method):
    """IndNonNegative prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndNonNegative(), rhs)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indnonnegative_emits_relu_on_mip():
    """On MIP, IndNonNegative lowers to a single Relu primitive."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndNonNegative(), rhs)

    n_relu = sum(1 for s in v.canonicalizer.steps if isinstance(s, Relu))
    assert n_relu == 1


def test_indnonnegative_gamma_is_ignored():
    """Indicator prox is gamma-independent; lowering should still produce one Relu."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndNonNegative(), rhs, gamma=42.0)

    n_relu = sum(1 for s in v.canonicalizer.steps if isinstance(s, Relu))
    assert n_relu == 1


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indnonpositive_returns_linear_expression(method):
    """IndNonPositive prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndNonPositive(), rhs)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indnonpositive_emits_one_relu_on_mip():
    """
    IndNonPositive uses the -Relu(-rhs) reduction, so MIP sees exactly one Relu.
    Its input bounds are the negation/swap of the original rhs bounds.
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-2.0, ub=1.0)
    v.prox(IndNonPositive(), rhs)

    relus = [s for s in v.canonicalizer.steps if isinstance(s, Relu)]
    assert len(relus) == 1
    # Bounds on -rhs are [-rhs.ub, -rhs.lb] = [-1, 2]
    np.testing.assert_allclose(relus[0].lb, [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(relus[0].ub, [2.0, 2.0, 2.0])


# --- IndBallLinf ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indballlinf_returns_linear_expression(method):
    """IndBallLinf prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-2.0, ub=2.0)

    out = v.prox(IndBallLinf(r=0.7), rhs)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indballlinf_emits_satlin_with_symmetric_bounds():
    """IndBallLinf lowers to SaturatedLinear with l=-r, u=r broadcast to rhs.ndim."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-2.0, ub=2.0)
    v.prox(IndBallLinf(r=0.7), rhs)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, [-0.7, -0.7, -0.7])
    np.testing.assert_allclose(satlins[0].u, [0.7, 0.7, 0.7])


# --- IndBox ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indbox_returns_linear_expression(method):
    """IndBox prox lowering is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-2.0, ub=2.0)

    out = v.prox(IndBox(l=-0.5, u=0.5), rhs)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indbox_scalar_bounds_broadcast():
    """Scalar l/u get broadcast to rhs.ndim before SaturatedLinear sees them."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-2.0, ub=2.0)
    v.prox(IndBox(l=-0.5, u=0.5), rhs)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(satlins[0].u, [0.5, 0.5, 0.5])


def test_indbox_array_bounds_forwarded_unchanged():
    """Array l/u pass through to SaturatedLinear unchanged."""
    l = np.array([-1.0, 0.0, -0.5])
    u = np.array([0.5, 1.0, 1.0])
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-2.0, ub=2.0)
    v.prox(IndBox(l=l, u=u), rhs)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, l)
    np.testing.assert_allclose(satlins[0].u, u)


# --- IndPoint ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indpoint_returns_linear_expression(method):
    """
    IndPoint lowers to a degenerate SaturatedLinear with l == u == p.
    All three backends should accept this.
    """
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndPoint(p=np.array([0.1, 0.2, 0.3])), rhs)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indpoint_scalar_broadcast():
    """Scalar p is broadcast to rhs.ndim; SaturatedLinear sees l == u arrays."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndPoint(p=0.5), rhs)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(satlins[0].u, [0.5, 0.5, 0.5])


def test_indpoint_lhs_bounds_collapse_to_p():
    """The SaturatedLinear lhs bounds collapse to [p, p] for IndPoint."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    p = np.array([0.1, 0.2, 0.3])
    v.prox(IndPoint(p=p), rhs)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].lhs.lb, p)
    np.testing.assert_allclose(satlins[0].lhs.ub, p)


def test_unsupported_prox_raises_on_every_backend():
    """
    A ProxableFunction subclass not registered in any _PROX_DECOMPOSERS dict
    raises NotImplementedError with a backend-named message on every backend.
    """

    class Unregistered(ProxableFunction):
        pass

    for method in ("mip", "sdp", "simulator"):
        v = _make_verifier(method)
        rhs = v.add_param(3, lb=-1.0, ub=1.0)

        with pytest.raises(NotImplementedError, match="Unregistered"):
            v.prox(Unregistered(), rhs)


def test_unsupported_prox_error_names_the_backend():
    """The error message identifies which decomposer rejected the prox."""

    class Unregistered(ProxableFunction):
        pass

    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    with pytest.raises(NotImplementedError, match="MIPStepDecomposer"):
        v.prox(Unregistered(), rhs)
