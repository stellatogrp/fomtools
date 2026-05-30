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
    HuberLoss,
    IndBallLinf,
    IndBox,
    IndHalfSpace,
    IndNonNegative,
    IndNonPositive,
    IndPoint,
    Linear,
    NormL1,
    ProxableFunction,
    SqL2Norm,
    SumNegative,
    SumPositive,
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


# --- HuberLoss ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_huberloss_returns_linear_expression(method):
    """HuberLoss prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(HuberLoss(delta=0.5), rhs, gamma=0.4)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_huberloss_emits_softthreshold_on_mip():
    """
    HuberLoss lowers to a single SoftThreshold step on MIP, with the
    threshold scaled to delta * (1 + gamma).
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(HuberLoss(delta=0.5), rhs, gamma=0.4)

    soft_steps = [
        s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)
    ]
    assert len(soft_steps) == 1
    # lambd should be delta * (1 + gamma) = 0.5 * 1.4 = 0.7
    assert np.isclose(soft_steps[0].lambd, 0.7)


def test_huberloss_decomposes_to_two_relus_on_sdp():
    """
    On SDP, the SoftThreshold from prox_huberloss recursively decomposes
    into two ReLUs via _DECOMPOSERS — same composition shape as the user's
    explicit two-ReLU form.
    """
    v = _make_verifier("sdp")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(HuberLoss(delta=0.5), rhs, gamma=0.4)

    assert len(v.canonicalizer._relu_steps) == 2
    assert all(isinstance(s, Relu) for s in v.canonicalizer._relu_steps)


def test_huberloss_prox_middle_branch():
    """
    For rhs entirely inside the quadratic region (|rhs_i| <= delta*(1+gamma)),
    the prox is pure linear shrinkage by 1/(1+gamma).
    """
    delta, gamma = 1.0, 1.0
    # theta = 2.0; all |rhs_i| < 2.0
    rhs = np.array([-1.5, 0.5, 1.9, -0.1])
    expected = rhs / (1.0 + gamma)
    np.testing.assert_allclose(
        HuberLoss(delta=delta).prox(rhs, gamma), expected
    )


def test_huberloss_prox_tail_branch():
    """
    For rhs entirely outside the quadratic region (|rhs_i| > delta*(1+gamma)),
    the prox subtracts a constant gamma*delta toward zero.
    """
    delta, gamma = 1.0, 1.0
    # theta = 2.0; all |rhs_i| > 2.0
    rhs = np.array([-5.0, 3.5, 2.5, -4.2])
    expected = rhs - gamma * delta * np.sign(rhs)
    np.testing.assert_allclose(
        HuberLoss(delta=delta).prox(rhs, gamma), expected
    )


def test_huberloss_prox_matches_closed_form():
    """
    Numerical math-layer sanity: HuberLoss.prox matches the closed-form
    piecewise formula on a random rhs mixing both regimes.
    """
    rng = np.random.default_rng(11)
    delta, gamma = 0.7, 0.6
    rhs = rng.normal(scale=2.0, size=12)  # mixes the quadratic and linear regions
    theta = delta * (1.0 + gamma)
    expected = np.where(
        np.abs(rhs) <= theta,
        rhs / (1.0 + gamma),
        rhs - gamma * delta * np.sign(rhs),
    )
    np.testing.assert_allclose(
        HuberLoss(delta=delta).prox(rhs, gamma), expected
    )


def test_huberloss_delta_zero_identity():
    """
    delta = 0 makes h(x) identically zero; the prox is the identity for
    any rhs and any gamma.
    """
    rhs = np.array([-3.7, 0.0, 0.42, 2.0])
    for gamma in (0.1, 1.0, 10.0):
        np.testing.assert_allclose(HuberLoss(delta=0.0).prox(rhs, gamma), rhs)


# --- IndNonNegative ---


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


# --- IndHalfSpace ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indhalfspace_returns_linear_expression(method):
    """
    IndHalfSpace lowers to one scalar Relu plus a rank-one lift back to
    rhs.ndim. All three backends accept this.
    """
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndHalfSpace(a=np.array([1.0, 1.0, 0.0]), b=0.5), rhs)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indhalfspace_emits_one_scalar_relu_on_mip():
    """Exactly one Relu, applied to a 1-D signed-distance LinearExpression."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndHalfSpace(a=np.array([1.0, 1.0, 0.0]), b=0.5), rhs)

    relus = [s for s in v.canonicalizer.steps if isinstance(s, Relu)]
    assert len(relus) == 1
    assert relus[0].rhs.ndim == 1


def test_indhalfspace_gamma_is_ignored():
    """
    Indicator prox is independent of gamma — the emitted step structure
    is identical regardless of the gamma argument.
    """
    a, b = np.array([1.0, -1.0, 0.5]), 0.3

    v1 = _make_verifier("mip")
    rhs1 = v1.add_param(3, lb=-1.0, ub=1.0)
    v1.prox(IndHalfSpace(a=a, b=b), rhs1, gamma=1.0)

    v2 = _make_verifier("mip")
    rhs2 = v2.add_param(3, lb=-1.0, ub=1.0)
    v2.prox(IndHalfSpace(a=a, b=b), rhs2, gamma=10.0)

    relus1 = [s for s in v1.canonicalizer.steps if isinstance(s, Relu)]
    relus2 = [s for s in v2.canonicalizer.steps if isinstance(s, Relu)]
    assert len(relus1) == len(relus2) == 1
    assert relus1[0].rhs.ndim == relus2[0].rhs.ndim == 1


def test_indhalfspace_prox_matches_closed_form():
    """
    Numpy-level: cover the three regions — strictly inside, on the
    boundary (prox is identity), and outside (boundary projection).
    """
    a = np.array([1.0, 2.0, -1.0])
    b = 0.5
    norm_a_sq = float(np.dot(a, a))
    f = IndHalfSpace(a=a, b=b)

    # strictly inside (a @ v = -1.5 < 0.5): prox == v
    v_in = np.array([0.0, -0.5, 0.5])
    np.testing.assert_allclose(f.prox(v_in), v_in)

    # boundary (a @ v == b exactly): prox == v
    v_bd = np.array([0.5, 0.0, 0.0])
    assert np.isclose(float(a @ v_bd), b)
    np.testing.assert_allclose(f.prox(v_bd), v_bd)

    # outside: prox lands on the boundary
    v_out = np.array([1.0, 1.0, 0.0])
    expected = v_out - ((float(a @ v_out) - b) / norm_a_sq) * a
    np.testing.assert_allclose(f.prox(v_out), expected)
    assert np.isclose(float(a @ f.prox(v_out)), b)


def test_indhalfspace_eval():
    """eval returns 0.0 inside the halfspace, +inf outside."""
    f = IndHalfSpace(a=np.array([1.0, 0.0]), b=1.0)
    assert f.eval(np.array([0.5, 100.0])) == 0.0     # inside
    assert f.eval(np.array([1.0, 100.0])) == 0.0     # boundary
    assert f.eval(np.array([1.5, 100.0])) == np.inf  # outside


def test_indhalfspace_rejects_zero_a():
    """Zero ``a`` would zero the projection denominator; reject at construction."""
    with pytest.raises(ValueError, match="nonzero"):
        IndHalfSpace(a=np.zeros(3), b=0.0)


def test_indhalfspace_grad_raises():
    """Indicators are not differentiable; grad must raise."""
    f = IndHalfSpace(a=np.array([1.0, 0.0]), b=0.0)
    with pytest.raises(NotImplementedError):
        f.grad(np.array([0.0, 0.0]))


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


# --- Linear ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_linear_returns_linear_expression(method):
    """
    Linear lowers to pure ``rhs - gamma*c`` arithmetic on the
    ``LinearExpression`` — no Step is emitted and no fresh leaf is
    introduced. All three backends inherit from the base class.
    """
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    c = np.array([1.0, -2.0, 3.0])
    out = v.prox(Linear(c=c), rhs, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_linear_emits_no_steps_on_mip():
    """Linear's prox is pure arithmetic — no Step is added to the canonicalizer."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(Linear(c=np.array([1.0, -2.0, 3.0])), rhs, gamma=0.5)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_linear_const_and_coefficient_with_array_c():
    """
    For numeric ``c``, the output decomposition equals
    ``I * rhs_leaf`` (coefficient) and ``const == -gamma*c``.
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    c = np.array([1.0, -2.0, 3.0])
    gamma = 0.5
    out = v.prox(Linear(c=c), rhs, gamma=gamma)

    np.testing.assert_allclose(out.const, -gamma * c)
    # rhs is a leaf — its decomposition is {rhs: I}; after subtracting gamma*c
    # (a constant), the coefficient on rhs is still I.
    np.testing.assert_allclose(
        out.decomposition_dict[rhs].toarray(), np.eye(3)
    )


def test_linear_supports_scalar_c():
    """Scalar ``c`` broadcasts via ``LinearExpression - gamma * scalar``."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(Linear(c=2.5), rhs, gamma=0.4)
    np.testing.assert_allclose(out.const, np.full(3, -0.4 * 2.5))


def test_linear_supports_linexpr_c():
    """
    When ``c`` is itself a ``LinearExpression`` parameter, the output
    references both leaves: ``+I`` on ``rhs`` and ``-gamma*I`` on ``c``.
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    c = v.add_param(3, lb=-0.5, ub=0.5)

    gamma = 0.5
    out = v.prox(Linear(c=c), rhs, gamma=gamma)

    np.testing.assert_allclose(
        out.decomposition_dict[rhs].toarray(), np.eye(3)
    )
    np.testing.assert_allclose(
        out.decomposition_dict[c].toarray(), -gamma * np.eye(3)
    )


def test_linear_prox_matches_closed_form():
    """Numpy-level: ``Linear(c).prox(rhs, gamma)`` equals ``rhs - gamma*c``."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = rng.standard_normal(4)
        rhs = rng.standard_normal(4)
        gamma = float(rng.uniform(0.1, 2.0))
        np.testing.assert_allclose(
            Linear(c=c).prox(rhs, gamma=gamma), rhs - gamma * c
        )
    # scalar c case
    np.testing.assert_allclose(
        Linear(c=1.5).prox(np.array([2.0, 3.0]), gamma=0.5),
        np.array([2.0, 3.0]) - 0.5 * 1.5,
    )


def test_linear_eval_and_grad():
    """Numpy-level eval and grad sanity."""
    c = np.array([1.0, 2.0, 3.0])
    f = Linear(c=c)
    assert np.isclose(f.eval(np.array([1.0, 1.0, 1.0])), 6.0)
    np.testing.assert_allclose(f.grad(np.zeros(3)), c)


# --- SqL2Norm ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_sql2norm_returns_linear_expression(method):
    """
    SqL2Norm lowers to ``rhs / (1 + gamma)`` — pure scalar shrinkage,
    no Step emitted. All three backends inherit from the base class.
    """
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SqL2Norm(), rhs, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_sql2norm_emits_no_steps_on_mip():
    """SqL2Norm's prox is pure arithmetic — no Step is added."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(SqL2Norm(), rhs, gamma=0.5)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_sql2norm_coefficient_is_scaled_identity():
    """The output decomposition is ``(1/(1+gamma)) * I`` on the rhs leaf."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    gamma = 0.5
    out = v.prox(SqL2Norm(), rhs, gamma=gamma)

    np.testing.assert_allclose(
        out.decomposition_dict[rhs].toarray(), np.eye(3) / (1.0 + gamma)
    )
    np.testing.assert_allclose(out.const, np.zeros(3))


def test_sql2norm_gamma_one_halves_coefficient():
    """At the default ``gamma=1``, the coefficient collapses to ``0.5 * I``."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    out = v.prox(SqL2Norm(), rhs)
    np.testing.assert_allclose(
        out.decomposition_dict[rhs].toarray(), 0.5 * np.eye(3)
    )


def test_sql2norm_prox_matches_closed_form():
    """Numpy-level: ``SqL2Norm().prox(rhs, gamma)`` equals ``rhs/(1+gamma)``."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        rhs = rng.standard_normal(4)
        gamma = float(rng.uniform(0.1, 2.0))
        np.testing.assert_allclose(
            SqL2Norm().prox(rhs, gamma=gamma), rhs / (1.0 + gamma)
        )


def test_sql2norm_eval_and_grad():
    """Numpy-level eval and grad sanity."""
    f = SqL2Norm()
    assert np.isclose(f.eval(np.array([3.0, 4.0])), 12.5)
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(f.grad(x), x)


# --- SumPositive ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_sumpositive_returns_linear_expression(method):
    """
    SumPositive lowers to one SaturatedLinear (clip to [0, gamma]) followed
    by a subtraction. All three backends accept this.
    """
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SumPositive(), rhs, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_sumpositive_emits_one_satlin_on_mip():
    """One SaturatedLinear with l == 0, u == gamma; nothing else."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    gamma = 0.5
    v.prox(SumPositive(), rhs, gamma=gamma)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, np.zeros(3))
    np.testing.assert_allclose(satlins[0].u, np.full(3, gamma))


def test_sumpositive_prox_matches_closed_form():
    """
    Numpy-level: SumPositive.prox matches the piecewise definition on
    a mixed-sign input crossing all three branches.
    """
    f = SumPositive()
    rhs = np.array([2.0, 0.3, -0.5, -2.0, 0.0])
    gamma = 1.0
    expected = np.array([1.0, 0.0, -0.5, -2.0, 0.0])
    np.testing.assert_allclose(f.prox(rhs, gamma=gamma), expected)

    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(6)
        g = float(rng.uniform(0.1, 2.0))
        out = f.prox(x, gamma=g)
        expected = np.where(
            x > g, x - g, np.where(x < 0, x, 0.0)
        )
        np.testing.assert_allclose(out, expected)


def test_sumpositive_eval_and_grad():
    """Numpy-level eval and grad sanity."""
    f = SumPositive()
    assert np.isclose(f.eval(np.array([1.0, -2.0, 3.0])), 4.0)
    np.testing.assert_allclose(
        f.grad(np.array([1.0, -2.0, 3.0, 0.0])),
        np.array([1.0, 0.0, 1.0, 0.0]),
    )


# --- SumNegative ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_sumnegative_returns_linear_expression(method):
    """
    SumNegative lowers to one SaturatedLinear (clip to [-gamma, 0])
    followed by a subtraction. All three backends accept this.
    """
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SumNegative(), rhs, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_sumnegative_emits_one_satlin_on_mip():
    """One SaturatedLinear with l == -gamma, u == 0; nothing else."""
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    gamma = 0.5
    v.prox(SumNegative(), rhs, gamma=gamma)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, np.full(3, -gamma))
    np.testing.assert_allclose(satlins[0].u, np.zeros(3))


def test_sumnegative_prox_matches_closed_form():
    """
    Numpy-level: SumNegative.prox matches the piecewise definition on
    a mixed-sign input crossing all three branches.
    """
    f = SumNegative()
    rhs = np.array([2.0, 0.3, -0.5, -2.0, 0.0])
    gamma = 1.0
    expected = np.array([2.0, 0.3, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(f.prox(rhs, gamma=gamma), expected)

    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.standard_normal(6)
        g = float(rng.uniform(0.1, 2.0))
        out = f.prox(x, gamma=g)
        expected = np.where(
            x < -g, x + g, np.where(x > 0, x, 0.0)
        )
        np.testing.assert_allclose(out, expected)


def test_sumnegative_eval_and_grad():
    """Numpy-level eval and grad sanity."""
    f = SumNegative()
    assert np.isclose(f.eval(np.array([1.0, -2.0, 3.0])), 2.0)
    np.testing.assert_allclose(
        f.grad(np.array([1.0, -2.0, 3.0, 0.0])),
        np.array([0.0, -1.0, 0.0, 0.0]),
    )


def test_sumnegative_is_sumpositive_mirror():
    """
    The reflection identity $f_-(x) = f_+(-x)$ implies
    $\\operatorname{prox}_{\\gamma f_-}(v) = -\\operatorname{prox}_{\\gamma f_+}(-v)$.
    Catches sign drift in either lowering.
    """
    f_pos, f_neg = SumPositive(), SumNegative()
    rng = np.random.default_rng(2)
    for _ in range(8):
        v_vec = rng.standard_normal(5)
        g = float(rng.uniform(0.1, 2.0))
        np.testing.assert_allclose(
            f_neg.prox(v_vec, gamma=g), -f_pos.prox(-v_vec, gamma=g)
        )


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
