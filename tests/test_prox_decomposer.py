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
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import (
    ElasticNet,
    HingeLoss,
    HuberLoss,
    IndAffine,
    IndBallLinf,
    IndBox,
    IndHalfSpace,
    IndNonNegative,
    IndNonPositive,
    IndPoint,
    LabeledHingeLoss,
    LeastSquares,
    Linear,
    NormL1,
    ProxableFunction,
    Quadratic,
    SoftNonNegative,
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
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(NormL1(lambd=0.3), arg, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_norml1_emits_softthreshold_on_mip():
    """
    On MIP, NormL1 lowers to a single SoftThreshold primitive (no recursive
    decomposition — MIP encodes SoftThreshold natively).
    """
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(NormL1(lambd=0.3), arg, gamma=0.5)

    n_soft = sum(1 for s in v.canonicalizer.steps if isinstance(s, SoftThreshold))
    assert n_soft == 1
    # SoftThreshold's lambd should be the scaled value lambd * gamma = 0.15
    soft_step = next(s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold))
    assert np.isclose(soft_step.lambd, 0.15)


def test_norml1_decomposes_to_two_relus_on_sdp():
    """
    On SDP, the SoftThreshold emitted by prox_norml1 recursively decomposes
    into two ReLUs via the existing _DECOMPOSERS registry — confirms that
    the two registry layers (prox + step) compose correctly.
    """
    v = _make_verifier("sdp")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(NormL1(lambd=0.3), arg, gamma=0.5)

    # SDP canonicalizer tracks Relus it encoded
    assert len(v.canonicalizer._relu_steps) == 2
    assert all(isinstance(s, Relu) for s in v.canonicalizer._relu_steps)


# --- ElasticNet ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_elasticnet_returns_linear_expression(method):
    """ElasticNet prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(ElasticNet(alpha=0.3, beta=0.5), arg, gamma=0.4)

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
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(ElasticNet(alpha=0.3, beta=0.5), arg, gamma=0.4)

    soft_steps = [s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)]
    assert len(soft_steps) == 1
    # SoftThreshold's lambd should be alpha * gamma = 0.12
    assert np.isclose(soft_steps[0].lambd, 0.12)


def test_elasticnet_decomposes_to_two_relus_on_sdp():
    """
    On SDP, the SoftThreshold from prox_elasticnet recursively decomposes
    into two ReLUs via _DECOMPOSERS — same composition as NormL1.
    """
    v = _make_verifier("sdp")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(ElasticNet(alpha=0.3, beta=0.5), arg, gamma=0.4)

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
    arg_en = v_en.add_param(3, lb=-1.0, ub=1.0)
    v_en.prox(ElasticNet(alpha=alpha, beta=0.0), arg_en, gamma=gamma)

    v_l1 = _make_verifier("mip")
    rhs_l1 = v_l1.add_param(3, lb=-1.0, ub=1.0)
    v_l1.prox(NormL1(lambd=alpha), rhs_l1, gamma=gamma)

    soft_en = [s for s in v_en.canonicalizer.steps if isinstance(s, SoftThreshold)][0]
    soft_l1 = [s for s in v_l1.canonicalizer.steps if isinstance(s, SoftThreshold)][0]
    assert np.isclose(soft_en.lambd, soft_l1.lambd)


def test_elasticnet_ridge_only_when_alpha_zero():
    """
    With alpha = 0 the prox is pure ridge shrinkage: arg / (1 + 2*beta*gamma).
    Verified numerically on the math wrapper (the lowering still emits a
    SoftThreshold with lambd=0, which is the identity-shaped envelope).
    """
    beta, gamma = 0.5, 0.4
    arg = np.array([-2.0, 0.05, 1.5])
    expected = arg / (1.0 + 2.0 * beta * gamma)
    np.testing.assert_allclose(ElasticNet(alpha=0.0, beta=beta).prox(arg, gamma), expected)


def test_elasticnet_prox_matches_closed_form():
    """
    Numerical math-layer sanity: ElasticNet.prox matches the closed-form
    S_{alpha*gamma}(arg) / (1 + 2*beta*gamma) on a random arg.
    """
    rng = np.random.default_rng(7)
    alpha, beta, gamma = 0.3, 0.5, 0.4
    arg = rng.normal(size=8)
    thresh = alpha * gamma
    expected = np.sign(arg) * np.maximum(np.abs(arg) - thresh, 0.0) / (1.0 + 2.0 * beta * gamma)
    np.testing.assert_allclose(ElasticNet(alpha=alpha, beta=beta).prox(arg, gamma), expected)


# --- HuberLoss ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_huberloss_returns_linear_expression(method):
    """HuberLoss prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(HuberLoss(delta=0.5), arg, gamma=0.4)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_huberloss_emits_softthreshold_on_mip():
    """
    HuberLoss lowers to a single SoftThreshold step on MIP, with the
    threshold scaled to delta * (1 + gamma).
    """
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(HuberLoss(delta=0.5), arg, gamma=0.4)

    soft_steps = [s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)]
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
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(HuberLoss(delta=0.5), arg, gamma=0.4)

    assert len(v.canonicalizer._relu_steps) == 2
    assert all(isinstance(s, Relu) for s in v.canonicalizer._relu_steps)


def test_huberloss_prox_middle_branch():
    """
    For arg entirely inside the quadratic region (|arg_i| <= delta*(1+gamma)),
    the prox is pure linear shrinkage by 1/(1+gamma).
    """
    delta, gamma = 1.0, 1.0
    # theta = 2.0; all |arg_i| < 2.0
    arg = np.array([-1.5, 0.5, 1.9, -0.1])
    expected = arg / (1.0 + gamma)
    np.testing.assert_allclose(HuberLoss(delta=delta).prox(arg, gamma), expected)


def test_huberloss_prox_tail_branch():
    """
    For arg entirely outside the quadratic region (|arg_i| > delta*(1+gamma)),
    the prox subtracts a constant gamma*delta toward zero.
    """
    delta, gamma = 1.0, 1.0
    # theta = 2.0; all |arg_i| > 2.0
    arg = np.array([-5.0, 3.5, 2.5, -4.2])
    expected = arg - gamma * delta * np.sign(arg)
    np.testing.assert_allclose(HuberLoss(delta=delta).prox(arg, gamma), expected)


def test_huberloss_prox_matches_closed_form():
    """
    Numerical math-layer sanity: HuberLoss.prox matches the closed-form
    piecewise formula on a random arg mixing both regimes.
    """
    rng = np.random.default_rng(11)
    delta, gamma = 0.7, 0.6
    arg = rng.normal(scale=2.0, size=12)  # mixes the quadratic and linear regions
    theta = delta * (1.0 + gamma)
    expected = np.where(
        np.abs(arg) <= theta,
        arg / (1.0 + gamma),
        arg - gamma * delta * np.sign(arg),
    )
    np.testing.assert_allclose(HuberLoss(delta=delta).prox(arg, gamma), expected)


def test_huberloss_delta_zero_identity():
    """
    delta = 0 makes h(x) identically zero; the prox is the identity for
    any arg and any gamma.
    """
    arg = np.array([-3.7, 0.0, 0.42, 2.0])
    for gamma in (0.1, 1.0, 10.0):
        np.testing.assert_allclose(HuberLoss(delta=0.0).prox(arg, gamma), arg)


# --- HingeLoss ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_hingeloss_returns_linear_expression(method):
    """Standard hinge lowers to one SaturatedLinear clip on the margin
    residual 1 - arg; all three backends accept it."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    out = v.prox(HingeLoss(), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_hingeloss_emits_one_satlin_on_mip():
    """One SaturatedLinear with l=0, u=γ, applied to the margin residual."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    gamma = 0.5
    v.prox(HingeLoss(), arg, gamma=gamma)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, np.zeros(3))
    np.testing.assert_allclose(satlins[0].u, np.full(3, gamma))


def test_hingeloss_prox_matches_closed_form():
    """Numpy-level: covers all three branches of the piecewise formula."""
    g = HingeLoss()
    # v < 1 - γ: shift up by γ; in middle: clamp to 1; v > 1: identity.
    arg = np.array([-1.0, 0.0, 0.5, 0.8, 1.5, 2.0])
    gamma = 0.5
    expected = np.array([-0.5, 0.5, 1.0, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(g.prox(arg, gamma=gamma), expected)

    rng = np.random.default_rng(0)
    for _ in range(5):
        v_vec = rng.uniform(-3.0, 3.0, size=6)
        g_val = float(rng.uniform(0.1, 2.0))
        out = g.prox(v_vec, gamma=g_val)
        expected = np.where(
            v_vec < 1 - g_val,
            v_vec + g_val,
            np.where(v_vec <= 1, 1.0, v_vec),
        )
        np.testing.assert_allclose(out, expected)


def test_hingeloss_eval_matches_pointwise():
    """eval is the sum of max(0, 1 - x_i)."""
    g = HingeLoss()
    x = np.array([0.0, 0.5, 1.0, 2.0, -1.0])
    assert np.isclose(g.eval(x), 1.0 + 0.5 + 0.0 + 0.0 + 2.0)


def test_hingeloss_grad_subgradient():
    """Subgradient: -1 if x < 1, else 0."""
    g = HingeLoss()
    x = np.array([0.5, 1.0, 1.5, -2.0])
    np.testing.assert_allclose(g.grad(x), [-1.0, 0.0, 0.0, -1.0])


# --- IndNonNegative ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indnonnegative_returns_linear_expression(method):
    """IndNonNegative prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndNonNegative(), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indnonnegative_emits_relu_on_mip():
    """On MIP, IndNonNegative lowers to a single Relu primitive."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndNonNegative(), arg)

    n_relu = sum(1 for s in v.canonicalizer.steps if isinstance(s, Relu))
    assert n_relu == 1


def test_indnonnegative_gamma_is_ignored():
    """Indicator prox is gamma-independent; lowering should still produce one Relu."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndNonNegative(), arg, gamma=42.0)

    n_relu = sum(1 for s in v.canonicalizer.steps if isinstance(s, Relu))
    assert n_relu == 1


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indnonpositive_returns_linear_expression(method):
    """IndNonPositive prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndNonPositive(), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indnonpositive_emits_one_relu_on_mip():
    """
    IndNonPositive uses the -Relu(-arg) reduction, so MIP sees exactly one Relu.
    Its input bounds are the negation/swap of the original arg bounds.
    """
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-2.0, ub=1.0)
    v.prox(IndNonPositive(), arg)

    relus = [s for s in v.canonicalizer.steps if isinstance(s, Relu)]
    assert len(relus) == 1
    # Bounds on -arg are [-arg.ub, -arg.lb] = [-1, 2]
    np.testing.assert_allclose(relus[0].lb, [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(relus[0].ub, [2.0, 2.0, 2.0])


# --- IndAffine ---


def test_indaffine_returns_linear_expression_on_mip():
    """IndAffine introduces a primal u and dual ν via a block KKT system;
    the prox output is u as a LinearExpression of dim n."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.array([0.5, 0.3])
    out = v.prox(IndAffine(A=A, b=b), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indaffine_sdp_works():
    """SDP path works without OBBT — bounds for both u and ν derived
    via solve_linexpr on the block KKT matrix."""
    v = _make_verifier("sdp")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.array([0.5, 0.3])
    out = v.prox(IndAffine(A=A, b=b), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3
    assert np.all(np.isfinite(out.lb))
    assert np.all(np.isfinite(out.ub))


def test_indaffine_sdp_obbt_tightens_registered_bounds():
    """With OBBT enabled on SDP, bounds on both u and ν are at least
    as tight as the solve-derived placeholders, and SDP.update_bounds
    rewrites the bound constraint matrices in place — same regression
    net as Quadratic's equivalent test."""
    from fomtools.linearexpression import LinearExpression, solve_linexpr, stack

    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.array([0.5, 0.3])
    arg_lb, arg_ub = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    M_block = np.block([[np.eye(3), A.T], [A, np.zeros((2, 2))]])

    # Reference: solve-derived bounds (what SDP gets at registration).
    v_ref = _make_verifier("sdp")
    arg_ref = v_ref.add_param(3, lb=arg_lb, ub=arg_ub)
    b_le = LinearExpression(decomposition_dict={}, const=b)
    solve_lb, solve_ub = solve_linexpr(M_block, stack([arg_ref, b_le])).bound_propagation()

    # With OBBT enabled: SDP.update_bounds is called by _tighten_constraint_bounds.
    v_obbt = _make_verifier("sdp", obbt="gurobi")
    arg = v_obbt.add_param(3, lb=arg_lb, ub=arg_ub)
    u = v_obbt.prox(IndAffine(A=A, b=b), arg)
    # _qp_kkt registers the dual ν before the primal x, so ν is the second-to-last.
    nu = v_obbt.iterates[-2]

    # OBBT-tightened bounds must dominate the solve-derived ones.
    assert np.all(u.lb >= solve_lb[:3] - 1e-9)
    assert np.all(u.ub <= solve_ub[:3] + 1e-9)
    assert np.all(nu.lb >= solve_lb[3:] - 1e-9)
    assert np.all(nu.ub <= solve_ub[3:] + 1e-9)
    assert np.all(np.isfinite(u.lb))
    assert np.all(np.isfinite(nu.lb))


def test_indaffine_simulator_supported():
    """Simulator now lowers IndAffine to a batched Affine block-KKT solve,
    returning the primal slice as a LinearExpression of the input dimension."""
    v = _make_verifier("simulator")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndAffine(A=np.array([[1.0, 1.0, 1.0]]), b=np.array([0.0])), arg)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indaffine_introduces_two_iterates():
    """First multi-iterate KKT prox: registers exactly TWO new iterates
    (primal u of dim n, dual ν of dim m)."""
    v = _make_verifier("mip")
    arg = v.add_param(4, lb=-1.0, ub=1.0)

    A = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])  # m=2, n=4
    b = np.array([0.5, -0.3])

    n_before = len(v.iterates)
    v.prox(IndAffine(A=A, b=b), arg)
    assert len(v.iterates) == n_before + 2
    # The two new iterates: ν (dim m=2) then the primal x (dim n=4) — _qp_kkt
    # registers the equality dual before the primal.
    assert v.iterates[-2].ndim == 2
    assert v.iterates[-1].ndim == 4


def test_indaffine_scalar_b_broadcasts_to_m():
    """Regression: a Python scalar ``b`` with m > 1 must be broadcast to
    length m before stacking. Previously ``np.atleast_1d`` produced a
    length-1 const, the stacked LE had dim n+1, and the dual iterate ν
    was registered with mismatched bounds (length 1 for an m-dim var).
    """
    v = _make_verifier("mip")
    arg = v.add_param(4, lb=-1.0, ub=1.0)
    # m = 3 rows, full row rank so the block KKT is nonsingular.
    A = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    v.prox(IndAffine(A=A, b=0.25), arg)
    # ν is registered just before the primal x (so second-to-last), of dim m=3 with
    # finite length-3 bounds derived from broadcasting b = 0.25.
    nu = v.iterates[-2]
    assert nu.ndim == 3
    assert nu.lb.shape == (3,) and nu.ub.shape == (3,)
    assert np.all(np.isfinite(nu.lb)) and np.all(np.isfinite(nu.ub))
    assert np.all(nu.lb <= nu.ub)


def test_indaffine_emits_no_step():
    """Lowering posts equality constraints, never a Step."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    A = np.array([[1.0, 1.0, 0.0]])
    b = np.array([0.5])

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(IndAffine(A=A, b=b), arg)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_indaffine_prox_matches_closed_form():
    """Numpy-level: prox = arg - A.T @ solve(A A.T, A arg - b), and the
    result lies exactly on the set {x : Ax = b}."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(3, 6))
        m = int(rng.integers(1, n))  # m < n so AA.T is invertible
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        arg = rng.standard_normal(n)

        ind = IndAffine(A=A, b=b)
        out = ind.prox(arg)
        expected = arg - A.T @ np.linalg.solve(A @ A.T, A @ arg - b)
        np.testing.assert_allclose(out, expected, atol=1e-10)
        # Projection lands on the affine set.
        np.testing.assert_allclose(A @ out, b, atol=1e-10)


def test_indaffine_gamma_is_ignored():
    """Indicator-prox γ-independence: the result is the same for any γ."""
    A = np.array([[1.0, 2.0, 0.0]])
    b = np.array([1.0])
    arg = np.array([0.5, 0.5, 0.5])
    ind = IndAffine(A=A, b=b)
    np.testing.assert_allclose(ind.prox(arg, gamma=1.0), ind.prox(arg, gamma=10.0))


def test_indaffine_eval():
    """eval returns 0 on the affine set, +inf off it."""
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    ind = IndAffine(A=A, b=b)
    assert ind.eval(np.array([0.5, 0.5])) == 0.0  # on
    assert ind.eval(np.array([0.6, 0.4])) == 0.0  # on (sum 1)
    assert ind.eval(np.array([1.0, 1.0])) == np.inf  # off (sum 2)


def test_indaffine_grad_raises():
    """Indicators are not differentiable; grad must raise."""
    ind = IndAffine(A=np.eye(2), b=np.zeros(2))
    with pytest.raises(NotImplementedError):
        ind.grad(np.zeros(2))


def test_indaffine_eval_and_prox_reject_linexpr_b():
    """Regression: ``eval`` and ``prox`` are numeric-only; a
    ``LinearExpression`` ``b`` must raise ``TypeError`` early.

    Without the guard, ``eval`` silently makes ``self.A @ x - self.b`` an
    LE (via ``__array_priority__``) and then ``np.allclose`` / ``np.abs``
    fails inside numpy with a confusing traceback; ``prox`` hits
    ``np.linalg.solve(..., A @ arg - b_LE)`` and dies on the same path.
    """
    v = _make_verifier("mip")
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b_le = v.add_param(2, lb=-0.5, ub=0.5)
    ind = IndAffine(A=A, b=b_le)
    x = np.array([0.5, 0.0, -0.5])
    with pytest.raises(TypeError, match="LinearExpression"):
        ind.eval(x)
    with pytest.raises(TypeError, match="LinearExpression"):
        ind.prox(x)


def test_indaffine_supports_linexpr_b():
    """``b`` may be a LinearExpression parameter; the posted equalities
    reference arg, u, ν, and b's leaf."""
    v = _make_verifier("mip")
    n, m = 3, 2
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    arg = v.add_param(n, lb=-1.0, ub=1.0)
    b_param = v.add_param(m, lb=-0.5, ub=0.5)

    out = v.prox(IndAffine(A=A, b=b_param), arg)
    assert out.is_leaf
    assert out.ndim == n


def test_indaffine_end_to_end_mip():
    """Pin arg via tight bounds, solve for u, assert u matches the
    closed-form projection and Au == b numerically."""
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.array([0.5, 0.3])
    arg_value = np.array([0.4, 0.6, -0.2])

    v = _make_verifier("mip")
    arg = v.add_param(3, lb=arg_value, ub=arg_value)
    u = v.prox(IndAffine(A=A, b=b), arg)

    expected = arg_value - A.T @ np.linalg.solve(A @ A.T, A @ arg_value - b)

    solver = v.canonicalizer.solver
    for i in range(3):
        solver.set_objective(u[i], "minimize")
        solver.optimize()
        assert solver.is_optimal()
        u_min = solver["objVal"]

        solver.set_objective(u[i], "maximize")
        solver.optimize()
        assert solver.is_optimal()
        u_max = solver["objVal"]

        assert np.isclose(u_min, u_max, atol=1e-6)
        assert np.isclose(u_min, expected[i], atol=1e-6)


def test_indaffine_rejects_nonmatching_b_shape():
    """b must have length equal to A.shape[0]."""
    with pytest.raises(ValueError, match="shape"):
        IndAffine(A=np.eye(3), b=np.zeros(4))


def test_indaffine_rejects_1d_A():
    """A must be 2-D."""
    with pytest.raises(ValueError, match="2-D"):
        IndAffine(A=np.array([1.0, 2.0, 3.0]), b=np.array([0.0]))


# --- IndBallLinf ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indballlinf_returns_linear_expression(method):
    """IndBallLinf prox is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-2.0, ub=2.0)

    out = v.prox(IndBallLinf(r=0.7), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indballlinf_emits_satlin_with_symmetric_bounds():
    """IndBallLinf lowers to SaturatedLinear with l=-r, u=r broadcast to arg.ndim."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-2.0, ub=2.0)
    v.prox(IndBallLinf(r=0.7), arg)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, [-0.7, -0.7, -0.7])
    np.testing.assert_allclose(satlins[0].u, [0.7, 0.7, 0.7])


# --- IndBox ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indbox_returns_linear_expression(method):
    """IndBox prox lowering is registered on the base; every backend inherits."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-2.0, ub=2.0)

    out = v.prox(IndBox(l=-0.5, u=0.5), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indbox_scalar_bounds_broadcast():
    """Scalar l/u get broadcast to arg.ndim before SaturatedLinear sees them."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-2.0, ub=2.0)
    v.prox(IndBox(l=-0.5, u=0.5), arg)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(satlins[0].u, [0.5, 0.5, 0.5])


def test_indbox_array_bounds_forwarded_unchanged():
    """Array l/u pass through to SaturatedLinear unchanged."""
    l = np.array([-1.0, 0.0, -0.5])
    u = np.array([0.5, 1.0, 1.0])
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-2.0, ub=2.0)
    v.prox(IndBox(l=l, u=u), arg)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, l)
    np.testing.assert_allclose(satlins[0].u, u)


def test_indbox_eval_and_prox_reject_linexpr_bounds():
    """Regression: ``eval`` and ``prox`` are numeric-only; a
    ``LinearExpression`` ``l`` or ``u`` must raise ``TypeError`` early.

    Without the guard, ``eval``'s ``x >= self.l`` returns an
    :class:`InequalityConstraint` whose truthiness silently lies, and
    ``prox``'s ``np.minimum(np.maximum(arg, l_LE), u_LE)`` dies in a
    generic ``LinearExpression`` codepath.
    """
    v = _make_verifier("mip")
    l_le = v.add_param(3, lb=-1.0, ub=0.0)
    u_le = v.add_param(3, lb=0.0, ub=1.0)
    x = np.array([0.0, 0.0, 0.0])

    # LE on l alone.
    ind_l = IndBox(l=l_le, u=1.0)
    with pytest.raises(TypeError, match="LinearExpression"):
        ind_l.eval(x)
    with pytest.raises(TypeError, match="LinearExpression"):
        ind_l.prox(x)

    # LE on u alone.
    ind_u = IndBox(l=-1.0, u=u_le)
    with pytest.raises(TypeError, match="LinearExpression"):
        ind_u.eval(x)
    with pytest.raises(TypeError, match="LinearExpression"):
        ind_u.prox(x)


# --- IndHalfSpace ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_indhalfspace_returns_linear_expression(method):
    """
    IndHalfSpace lowers to one scalar Relu plus a rank-one lift back to
    arg.ndim. All three backends accept this.
    """
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndHalfSpace(a=np.array([1.0, 1.0, 0.0]), b=0.5), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indhalfspace_emits_one_scalar_relu_on_mip():
    """Exactly one Relu, applied to a 1-D signed-distance LinearExpression."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndHalfSpace(a=np.array([1.0, 1.0, 0.0]), b=0.5), arg)

    relus = [s for s in v.canonicalizer.steps if isinstance(s, Relu)]
    assert len(relus) == 1
    assert relus[0].arg.ndim == 1


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
    assert relus1[0].arg.ndim == relus2[0].arg.ndim == 1


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
    assert f.eval(np.array([0.5, 100.0])) == 0.0  # inside
    assert f.eval(np.array([1.0, 100.0])) == 0.0  # boundary
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
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(IndPoint(p=np.array([0.1, 0.2, 0.3])), arg)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_indpoint_scalar_broadcast():
    """Scalar p is broadcast to arg.ndim; SaturatedLinear sees l == u arrays."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(IndPoint(p=0.5), arg)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(satlins[0].u, [0.5, 0.5, 0.5])


def test_indpoint_lhs_bounds_collapse_to_p():
    """The SaturatedLinear output bounds collapse to [p, p] for IndPoint."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    p = np.array([0.1, 0.2, 0.3])
    v.prox(IndPoint(p=p), arg)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].output.lb, p)
    np.testing.assert_allclose(satlins[0].output.ub, p)


# --- LabeledHingeLoss ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_labeledhingeloss_returns_linear_expression(method):
    """Labeled hinge lowers to one SaturatedLinear clip on the labeled
    margin residual; all three backends accept it."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    out = v.prox(LabeledHingeLoss(y=np.array([1.0, -1.0, 1.0])), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_labeledhingeloss_emits_one_satlin_on_mip():
    """One SaturatedLinear with l=0, u=γ, applied to the labeled residual."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    gamma = 0.5
    v.prox(LabeledHingeLoss(y=np.array([1.0, -1.0, 1.0])), arg, gamma=gamma)

    satlins = [s for s in v.canonicalizer.steps if isinstance(s, SaturatedLinear)]
    assert len(satlins) == 1
    np.testing.assert_allclose(satlins[0].l, np.zeros(3))
    np.testing.assert_allclose(satlins[0].u, np.full(3, gamma))


def test_labeledhingeloss_prox_matches_closed_form():
    """Numpy-level: branchwise check using the substitution-derived formula."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        y = rng.choice([-1.0, 1.0], size=n)
        arg = rng.uniform(-3.0, 3.0, size=n)
        gamma = float(rng.uniform(0.1, 2.0))

        g = LabeledHingeLoss(y=y)
        out = g.prox(arg, gamma=gamma)
        # u + y * satlin(1 - y*u; 0, gamma)
        clip = np.minimum(np.maximum(1.0 - y * arg, 0.0), gamma)
        expected = arg + y * clip
        np.testing.assert_allclose(out, expected)


def test_labeledhingeloss_y_all_ones_matches_hingeloss():
    """LabeledHingeLoss(y=+1) must reduce to HingeLoss exactly."""
    rng = np.random.default_rng(1)
    for _ in range(3):
        n = int(rng.integers(2, 6))
        arg = rng.uniform(-3.0, 3.0, size=n)
        gamma = float(rng.uniform(0.1, 2.0))

        labeled = LabeledHingeLoss(y=np.ones(n)).prox(arg, gamma=gamma)
        unlabeled = HingeLoss().prox(arg, gamma=gamma)
        np.testing.assert_allclose(labeled, unlabeled)


def test_labeledhingeloss_y_all_minus_ones_flips():
    """LabeledHingeLoss(y=-1).prox(v) == -HingeLoss().prox(-v)."""
    rng = np.random.default_rng(2)
    for _ in range(3):
        n = int(rng.integers(2, 6))
        arg = rng.uniform(-3.0, 3.0, size=n)
        gamma = float(rng.uniform(0.1, 2.0))

        labeled = LabeledHingeLoss(y=-np.ones(n)).prox(arg, gamma=gamma)
        mirror = -HingeLoss().prox(-arg, gamma=gamma)
        np.testing.assert_allclose(labeled, mirror)


def test_labeledhingeloss_eval():
    """eval is the sum of max(0, 1 - y_i x_i)."""
    g = LabeledHingeLoss(y=np.array([1.0, -1.0, 1.0]))
    x = np.array([0.5, -0.5, 2.0])
    # y*x = [0.5, 0.5, 2.0]; max(0, 1 - y*x) = [0.5, 0.5, 0]
    assert np.isclose(g.eval(x), 1.0)


def test_labeledhingeloss_grad():
    """grad: -y_i where y_i x_i < 1, else 0."""
    y = np.array([1.0, -1.0, 1.0])
    g = LabeledHingeLoss(y=y)
    x = np.array([0.5, -0.5, 2.0])
    # y*x = [0.5, 0.5, 2.0]; active set: [True, True, False]
    expected = np.array([-1.0, 1.0, 0.0])
    np.testing.assert_allclose(g.grad(x), expected)


def test_labeledhingeloss_rejects_non_pm1_entries():
    """y must contain only ±1 entries."""
    with pytest.raises(ValueError, match=r"±1"):
        LabeledHingeLoss(y=np.array([0.5, 1.0, -1.0]))


def test_labeledhingeloss_rejects_2d_y():
    """y must be 1-D."""
    with pytest.raises(ValueError, match="1-D"):
        LabeledHingeLoss(y=np.array([[1.0], [-1.0]]))


# --- Linear ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_linear_returns_linear_expression(method):
    """
    Linear lowers to pure ``arg - gamma*c`` arithmetic on the
    ``LinearExpression`` — no Step is emitted and no fresh leaf is
    introduced. All three backends inherit from the base class.
    """
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    c = np.array([1.0, -2.0, 3.0])
    out = v.prox(Linear(c=c), arg, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_linear_emits_no_steps_on_mip():
    """Linear's prox is pure arithmetic — no Step is added to the canonicalizer."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(Linear(c=np.array([1.0, -2.0, 3.0])), arg, gamma=0.5)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_linear_const_and_coefficient_with_array_c():
    """
    For numeric ``c``, the output decomposition equals
    ``I * arg_leaf`` (coefficient) and ``const == -gamma*c``.
    """
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    c = np.array([1.0, -2.0, 3.0])
    gamma = 0.5
    out = v.prox(Linear(c=c), arg, gamma=gamma)

    np.testing.assert_allclose(out.const, -gamma * c)
    # arg is a leaf — its decomposition is {arg: I}; after subtracting gamma*c
    # (a constant), the coefficient on arg is still I.
    np.testing.assert_allclose(out.decomposition_dict[arg].toarray(), np.eye(3))


def test_linear_supports_scalar_c():
    """Scalar ``c`` broadcasts via ``LinearExpression - gamma * scalar``."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(Linear(c=2.5), arg, gamma=0.4)
    np.testing.assert_allclose(out.const, np.full(3, -0.4 * 2.5))


def test_linear_supports_linexpr_c():
    """
    When ``c`` is itself a ``LinearExpression`` parameter, the output
    references both leaves: ``+I`` on ``arg`` and ``-gamma*I`` on ``c``.
    """
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    c = v.add_param(3, lb=-0.5, ub=0.5)

    gamma = 0.5
    out = v.prox(Linear(c=c), arg, gamma=gamma)

    np.testing.assert_allclose(out.decomposition_dict[arg].toarray(), np.eye(3))
    np.testing.assert_allclose(out.decomposition_dict[c].toarray(), -gamma * np.eye(3))


def test_linear_prox_matches_closed_form():
    """Numpy-level: ``Linear(c).prox(arg, gamma)`` equals ``arg - gamma*c``."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = rng.standard_normal(4)
        arg = rng.standard_normal(4)
        gamma = float(rng.uniform(0.1, 2.0))
        np.testing.assert_allclose(Linear(c=c).prox(arg, gamma=gamma), arg - gamma * c)
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


# --- SoftNonNegative ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_softnonnegative_returns_linear_expression(method):
    """SoftNonNegative lowers to one Relu step plus arithmetic;
    all three backends accept it."""
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SoftNonNegative(alpha=0.1), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_softnonnegative_emits_one_relu_on_mip():
    """Exactly one Relu step is emitted by the lowering."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(SoftNonNegative(alpha=0.2), arg, gamma=1.0)

    relus = [s for s in v.canonicalizer.steps if isinstance(s, Relu)]
    assert len(relus) == 1
    # The Relu acts on the input arg leaf directly.
    assert relus[0].arg is arg


def test_softnonnegative_prox_matches_closed_form_gamma_one():
    """Numpy-level at γ=1: prox = α·v + (1-α)·max(v, 0) (standard LeakyReLU)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        alpha = float(rng.uniform(0.05, 0.95))
        arg = rng.uniform(-3.0, 3.0, size=n)

        out = SoftNonNegative(alpha=alpha).prox(arg, gamma=1.0)
        expected = alpha * arg + (1.0 - alpha) * np.maximum(arg, 0.0)
        np.testing.assert_allclose(out, expected)


def test_softnonnegative_prox_matches_closed_form_general_gamma():
    """Numpy-level for several γ: prox uses α̃(γ) = α / (α + γ(1-α))."""
    rng = np.random.default_rng(1)
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        alpha = 0.3
        arg = rng.uniform(-3.0, 3.0, size=4)
        out = SoftNonNegative(alpha=alpha).prox(arg, gamma=gamma)
        alpha_tilde = alpha / (alpha + gamma * (1.0 - alpha))
        expected = alpha_tilde * arg + (1.0 - alpha_tilde) * np.maximum(arg, 0.0)
        np.testing.assert_allclose(out, expected)


def test_softnonnegative_alpha_one_is_identity():
    """α=1 ⇒ f ≡ 0, prox is identity for any γ."""
    rng = np.random.default_rng(2)
    arg = rng.uniform(-3.0, 3.0, size=5)
    for gamma in [0.1, 1.0, 10.0]:
        np.testing.assert_allclose(SoftNonNegative(alpha=1.0).prox(arg, gamma=gamma), arg)


def test_softnonnegative_small_alpha_approaches_relu():
    """α → 0⁺ ⇒ f → I_{R+}, prox → ReLU."""
    arg = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = SoftNonNegative(alpha=1e-6).prox(arg, gamma=1.0)
    np.testing.assert_allclose(out, np.maximum(arg, 0.0), atol=1e-5)


def test_softnonnegative_eval():
    """eval is (1-α)/(2α) · sum(max(0, -x)²); zero on the non-negative orthant."""
    alpha = 0.1
    p = SoftNonNegative(alpha=alpha)
    # Mixed-sign input
    x = np.array([-1.0, 0.5, -2.0])
    coeff = (1.0 - alpha) / (2.0 * alpha)
    expected = coeff * (1.0 + 0.0 + 4.0)
    assert np.isclose(p.eval(x), expected)
    # Non-negative input
    assert p.eval(np.array([0.0, 1.0, 2.0])) == 0.0


def test_softnonnegative_grad():
    """grad is (1-α)/α · min(x, 0): zero on non-negatives, scaled on negatives."""
    alpha = 0.1
    p = SoftNonNegative(alpha=alpha)
    x = np.array([-1.0, 0.5, -2.0])
    coeff = (1.0 - alpha) / alpha
    expected = coeff * np.array([-1.0, 0.0, -2.0])
    np.testing.assert_allclose(p.grad(x), expected)


def test_softnonnegative_rejects_alpha_out_of_range():
    """α must lie in (0, 1]."""
    for bad in [0.0, -0.1, 1.5, 2.0]:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            SoftNonNegative(alpha=bad)


def test_softnonnegative_end_to_end_mip():
    """Pin arg via tight bounds, solve, verify the recovered prox value
    matches the LeakyReLU formula coordinate-wise."""
    alpha = 0.25
    gamma = 1.0
    arg_value = np.array([-1.0, 0.5, -0.3])
    expected = alpha * arg_value + (1.0 - alpha) * np.maximum(arg_value, 0.0)

    v = _make_verifier("mip")
    arg = v.add_param(3, lb=arg_value, ub=arg_value)
    out = v.prox(SoftNonNegative(alpha=alpha), arg, gamma=gamma)

    solver = v.canonicalizer.solver
    for i in range(3):
        solver.set_objective(out[i], "minimize")
        solver.optimize()
        assert solver.is_optimal()
        u_min = solver["objVal"]

        solver.set_objective(out[i], "maximize")
        solver.optimize()
        assert solver.is_optimal()
        u_max = solver["objVal"]

        assert np.isclose(u_min, u_max, atol=1e-6)
        assert np.isclose(u_min, expected[i], atol=1e-6)


# --- SqL2Norm ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_sql2norm_returns_linear_expression(method):
    """
    SqL2Norm lowers to ``arg / (1 + gamma)`` — pure scalar shrinkage,
    no Step emitted. All three backends inherit from the base class.
    """
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SqL2Norm(), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_sql2norm_emits_no_steps_on_mip():
    """SqL2Norm's prox is pure arithmetic — no Step is added."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(SqL2Norm(), arg, gamma=0.5)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_sql2norm_coefficient_is_scaled_identity():
    """The output decomposition is ``(1/(1+gamma)) * I`` on the arg leaf."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    gamma = 0.5
    out = v.prox(SqL2Norm(), arg, gamma=gamma)

    np.testing.assert_allclose(out.decomposition_dict[arg].toarray(), np.eye(3) / (1.0 + gamma))
    np.testing.assert_allclose(out.const, np.zeros(3))


def test_sql2norm_gamma_one_halves_coefficient():
    """At the default ``gamma=1``, the coefficient collapses to ``0.5 * I``."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    out = v.prox(SqL2Norm(), arg)
    np.testing.assert_allclose(out.decomposition_dict[arg].toarray(), 0.5 * np.eye(3))


def test_sql2norm_prox_matches_closed_form():
    """Numpy-level: ``SqL2Norm().prox(arg, gamma)`` equals ``arg/(1+gamma)``."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        arg = rng.standard_normal(4)
        gamma = float(rng.uniform(0.1, 2.0))
        np.testing.assert_allclose(SqL2Norm().prox(arg, gamma=gamma), arg / (1.0 + gamma))


def test_sql2norm_eval_and_grad():
    """Numpy-level eval and grad sanity."""
    f = SqL2Norm()
    assert np.isclose(f.eval(np.array([3.0, 4.0])), 12.5)
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(f.grad(x), x)


# --- Quadratic ---


def test_quadratic_returns_linear_expression_on_mip():
    """
    Quadratic lowers to a single linear equality (I + γQ) u == arg - γq on
    a fresh iterate u. MIP-only: SDP and Simulator backends explicitly
    defer (see the two ``_raises_notimplemented`` tests below).
    """
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    Q = np.eye(3) * 2.0
    q = np.array([0.5, -0.5, 0.1])
    out = v.prox(Quadratic(Q=Q, q=q), arg, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_quadratic_simulator_supported():
    """Simulator now lowers Quadratic to a batched Affine solve of
    (I + gamma*Q) x = arg - gamma*q, returning a LinearExpression."""
    v = _make_verifier("simulator")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(Quadratic(Q=np.eye(3), q=np.zeros(3)), arg)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_quadratic_sdp_works():
    """SDP now accepts Quadratic: finite bounds on u are derived via
    solve_linexpr through M^{-1} @ (arg - γq), with no inverse formed
    in the verification model."""
    v = _make_verifier("sdp")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    out = v.prox(Quadratic(Q=np.eye(3) * 2.0, q=np.zeros(3)), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3
    assert np.all(np.isfinite(out.lb))
    assert np.all(np.isfinite(out.ub))


def test_quadratic_sdp_obbt_tightens_registered_bounds():
    """With OBBT enabled on SDP, the u bounds end up at least as tight as
    the solve-derived placeholder bounds. The SDP model's bound
    constraints are rewritten in place by SDP.update_bounds (sdp.py:198),
    so the final model verifies against the OBBT-tightened bounds."""
    from fomtools.linearexpression import solve_linexpr

    Q = np.eye(3) * 2.0
    q = np.array([0.5, -0.5, 0.1])
    gamma = 0.5
    arg_lb, arg_ub = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    M = np.eye(3) + gamma * Q

    # Reference: solve-derived bounds (what SDP gets at registration).
    v_ref = _make_verifier("sdp")
    arg_ref = v_ref.add_param(3, lb=arg_lb, ub=arg_ub)
    solve_lb, solve_ub = solve_linexpr(M, arg_ref - gamma * q).bound_propagation()

    # With OBBT: SDP.update_bounds is called by _tighten_constraint_bounds.
    v_obbt = _make_verifier("sdp", obbt="gurobi")
    arg = v_obbt.add_param(3, lb=arg_lb, ub=arg_ub)
    u = v_obbt.prox(Quadratic(Q=Q, q=q), arg, gamma=gamma)

    # The OBBT-tightened bounds must dominate (be at least as tight as)
    # the solve-derived ones. For this diagonal Q + box arg the two
    # coincide because solve-derived is already tight; the assertion
    # catches any regression where tightened values fail to reach the
    # LinearExpression / SDP model.
    assert np.all(u.lb >= solve_lb - 1e-9)
    assert np.all(u.ub <= solve_ub + 1e-9)
    assert np.all(np.isfinite(u.lb))
    assert np.all(np.isfinite(u.ub))


def test_quadratic_derived_bounds_match_analytic_diagonal_Q():
    """For diagonal Q, u_i = M_ii^{-1} * (arg_i - γ q_i) with
    M_ii = 1 + γ q_ii. solve_linexpr + bound_propagation must reproduce
    the analytic interval exactly (M_ii > 0 preserves order)."""
    v = _make_verifier("mip")
    diag = np.array([1.0, 4.0, 9.0])
    Q = np.diag(diag)
    q = np.array([0.5, -0.5, 0.0])
    gamma = 0.5
    arg_lb, arg_ub = np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])
    arg = v.add_param(3, lb=arg_lb, ub=arg_ub)

    u = v.prox(Quadratic(Q=Q, q=q), arg, gamma=gamma)
    M_diag = 1.0 + gamma * diag
    expected_lb = (arg_lb - gamma * q) / M_diag
    expected_ub = (arg_ub - gamma * q) / M_diag
    np.testing.assert_allclose(u.lb, expected_lb, atol=1e-10)
    np.testing.assert_allclose(u.ub, expected_ub, atol=1e-10)


def test_quadratic_introduces_one_iterate():
    """The prox lowering introduces exactly one fresh iterate u in verifier.iterates."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    n_before = len(v.iterates)
    v.prox(Quadratic(Q=np.eye(3), q=np.zeros(3)), arg)
    assert len(v.iterates) == n_before + 1
    # The new iterate is the output u of dim 3.
    assert v.iterates[-1].ndim == 3


def test_quadratic_emits_no_step():
    """Quadratic lowers via constraint, not via a Step; canonicalizer.steps unchanged."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(Quadratic(Q=np.eye(3), q=np.zeros(3)), arg)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_quadratic_prox_matches_closed_form():
    """Numpy-level: prox solves (I + γQ) u = arg − γq via np.linalg.solve."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        A = rng.standard_normal((n, n))
        Q = A.T @ A  # PSD
        q = rng.standard_normal(n)
        arg = rng.standard_normal(n)
        gamma = float(rng.uniform(0.1, 2.0))

        out = Quadratic(Q=Q, q=q).prox(arg, gamma=gamma)
        expected = np.linalg.solve(np.eye(n) + gamma * Q, arg - gamma * q)
        np.testing.assert_allclose(out, expected, atol=1e-10)


def test_quadratic_supports_linexpr_q():
    """
    When ``q`` is a LinearExpression parameter, the posted equality must
    reference q's leaf, u's leaf, and arg's leaf. The output ``u`` is still
    a fresh leaf.
    """
    v = _make_verifier("mip")
    n = 3
    arg = v.add_param(n, lb=-1.0, ub=1.0)
    q_param = v.add_param(n, lb=-0.5, ub=0.5)
    Q = np.eye(n) * 2.0

    out = v.prox(Quadratic(Q=Q, q=q_param), arg, gamma=0.5)
    assert out.is_leaf
    assert out.ndim == n
    # ``u`` is freshly created — not the same leaf as arg or q_param.
    assert out is not arg
    assert out is not q_param


def test_quadratic_end_to_end_mip():
    """
    Pin ``arg`` via tight bounds, solve a trivial objective on ``u``, and
    verify the optimum equals the closed-form np.linalg.solve answer.
    This is the strongest numerical check: it exercises the full
    constraint-posting path (canonicalizer + OBBT + solver).
    """
    Q = np.array([[2.0, 0.5], [0.5, 1.5]])
    q = np.array([1.0, -1.0])
    gamma = 0.5
    arg_value = np.array([0.3, -0.2])

    v = _make_verifier("mip")
    arg = v.add_param(2, lb=arg_value, ub=arg_value)  # pinned
    u = v.prox(Quadratic(Q=Q, q=q), arg, gamma=gamma)

    # Closed-form reference
    expected = np.linalg.solve(np.eye(2) + gamma * Q, arg_value - gamma * q)

    # Minimize then maximize each coordinate of u; the feasible value of u
    # is uniquely determined by the equality, so min == max == expected[i].
    solver = v.canonicalizer.solver
    for i in range(2):
        solver.set_objective(u[i], "minimize")
        solver.optimize()
        assert solver.is_optimal()
        u_min = solver["objVal"]

        solver.set_objective(u[i], "maximize")
        solver.optimize()
        assert solver.is_optimal()
        u_max = solver["objVal"]

        assert np.isclose(u_min, u_max, atol=1e-6)
        assert np.isclose(u_min, expected[i], atol=1e-6)


def test_quadratic_eval_and_grad():
    """Numpy eval and grad sanity."""
    Q = np.array([[2.0, 0.0], [0.0, 4.0]])
    q = np.array([1.0, -1.0])
    f = Quadratic(Q=Q, q=q)
    x = np.array([1.0, 2.0])
    # 0.5 * (2*1 + 4*4) + (1*1 + (-1)*2) = 0.5*18 + (-1) = 8
    assert np.isclose(f.eval(x), 8.0)
    np.testing.assert_allclose(f.grad(x), Q @ x + q)


def test_quadratic_eval_and_grad_reject_linexpr_q():
    """Regression: ``eval`` / ``grad`` are numeric-only; a ``LinearExpression``
    ``q`` must raise ``TypeError`` early instead of producing a cryptic
    numpy error deeper inside ``np.dot(self.q, x)`` / ``Q @ x + self.q``.
    """
    v = _make_verifier("mip")
    q_le = v.add_param(2, lb=-1.0, ub=1.0)
    Q = np.eye(2)
    f = Quadratic(Q=Q, q=q_le)
    x = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="LinearExpression"):
        f.eval(x)
    with pytest.raises(TypeError, match="LinearExpression"):
        f.grad(x)


def test_quadratic_rejects_nonsquare_Q():
    """Non-square Q is invalid; raise at construction."""
    with pytest.raises(ValueError, match="square"):
        Quadratic(Q=np.zeros((3, 4)), q=np.zeros(3))


def test_quadratic_rejects_mismatched_q_shape():
    """Mismatched q.shape vs Q is invalid; raise at construction."""
    with pytest.raises(ValueError, match="shape"):
        Quadratic(Q=np.eye(3), q=np.zeros(4))


# --- LeastSquares (inherits from Quadratic) ---


def test_leastsquares_isinstance_quadratic():
    """LeastSquares must IS-A Quadratic. The dispatcher distinguishes by
    concrete class name, but the inheritance contract powers the math
    (prox/grad inherited unchanged from Quadratic)."""
    from fomtools.proxable_functions import Quadratic

    ls = LeastSquares(A=np.eye(3), b=np.zeros(3))
    assert isinstance(ls, Quadratic)


def test_leastsquares_Q_and_q_set_correctly():
    """Constructor must call super().__init__(Q=A.T@A, q=-A.T@b)
    correctly; downstream prox uses these fields generically."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    b = np.array([1.0, 0.0, -1.0])
    ls = LeastSquares(A=A, b=b)
    np.testing.assert_allclose(ls.Q, A.T @ A)
    np.testing.assert_allclose(ls.q, -A.T @ b)


def test_leastsquares_returns_linear_expression_on_mip():
    """LeastSquares lowers via the same prox_quadratic the dispatcher
    finds under the 'leastsquares' alias."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    A = np.eye(3) + 0.1 * np.ones((3, 3))
    b = np.array([0.5, -0.5, 0.1])

    out = v.prox(LeastSquares(A=A, b=b), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_leastsquares_sdp_works():
    """SDP path works without OBBT — inherits Quadratic's solve-based
    bound derivation via solve_linexpr."""
    v = _make_verifier("sdp")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    A = np.eye(3) + 0.1 * np.ones((3, 3))
    b = np.array([0.5, -0.5, 0.1])

    out = v.prox(LeastSquares(A=A, b=b), arg, gamma=0.5)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3
    assert np.all(np.isfinite(out.lb))
    assert np.all(np.isfinite(out.ub))


def test_leastsquares_simulator_supported():
    """Simulator now lowers LeastSquares via the shared Quadratic path
    (Q = A^T A, q = -A^T b), returning a LinearExpression."""
    v = _make_verifier("simulator")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(LeastSquares(A=np.eye(3), b=np.zeros(3)), arg)
    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_leastsquares_prox_matches_closed_form():
    """Numpy-level: prox solves (I + γ A.T@A) u = arg + γ A.T@b."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(2, 5))
        m = int(rng.integers(2, 6))
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        arg = rng.standard_normal(n)
        gamma = float(rng.uniform(0.1, 2.0))

        out = LeastSquares(A=A, b=b).prox(arg, gamma=gamma)
        expected = np.linalg.solve(np.eye(n) + gamma * A.T @ A, arg + gamma * A.T @ b)
        np.testing.assert_allclose(out, expected, atol=1e-10)


def test_leastsquares_eval_matches_squared_norm():
    """eval must return 0.5 * ||Ax - b||^2 — *including* the constant
    that Quadratic.eval drops. Catches a missing override."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 0.5])
    x = np.array([0.5, -0.3])
    ls = LeastSquares(A=A, b=b)
    expected = 0.5 * float(np.sum((A @ x - b) ** 2))
    assert np.isclose(ls.eval(x), expected)
    # Sanity: not equal to Quadratic.eval (which lacks the constant).
    quad_only = 0.5 * float(x @ ls.Q @ x) + float(ls.q @ x)
    assert not np.isclose(ls.eval(x), quad_only)
    assert np.isclose(ls.eval(x) - quad_only, 0.5 * float(b @ b))


def test_leastsquares_eval_and_grad_reject_linexpr_b():
    """Regression: ``eval`` is numeric-only; a ``LinearExpression`` ``b``
    must raise ``TypeError`` early instead of producing the cryptic
    ``NotImplementedError`` from ``LinearExpression.__matmul__`` when
    ``r = A @ x - self.b`` silently becomes symbolic and ``r @ r`` is
    invoked. ``grad`` is inherited from ``Quadratic`` and protected by
    the same guard on ``self.q`` (= ``-A.T @ b``).
    """
    v = _make_verifier("mip")
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_le = v.add_param(2, lb=-1.0, ub=1.0)
    ls = LeastSquares(A=A, b=b_le)
    x = np.array([0.5, -0.3])
    with pytest.raises(TypeError, match="LinearExpression"):
        ls.eval(x)
    # grad inherits Quadratic's guard via self.q (an LE since q = -A.T @ b).
    with pytest.raises(TypeError, match="LinearExpression"):
        ls.grad(x)


def test_leastsquares_grad_matches_AT_residual():
    """Inherited grad: Qx + q = A.T @ (Ax - b)."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([1.0, 0.0, -1.0])
    x = np.array([0.5, -0.3])
    ls = LeastSquares(A=A, b=b)
    np.testing.assert_allclose(ls.grad(x), A.T @ (A @ x - b))


def test_leastsquares_introduces_one_iterate():
    """Same structural property as Quadratic: one fresh iterate appended."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    n_before = len(v.iterates)
    v.prox(LeastSquares(A=np.eye(3), b=np.zeros(3)), arg)
    assert len(v.iterates) == n_before + 1
    assert v.iterates[-1].ndim == 3


def test_leastsquares_emits_no_step():
    """Same structural property as Quadratic: lowering posts a constraint, not a Step."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    n_steps_before = len(v.canonicalizer.steps)
    v.prox(LeastSquares(A=np.eye(3), b=np.zeros(3)), arg)
    assert len(v.canonicalizer.steps) == n_steps_before


def test_leastsquares_supports_linexpr_b():
    """``b`` can be a LinearExpression parameter; q = -A.T @ b flows through
    LinearExpression.__rmatmul__ and the posted equality references b's leaf."""
    v = _make_verifier("mip")
    n, m = 3, 4
    A = np.eye(m, n) + 0.1
    arg = v.add_param(n, lb=-1.0, ub=1.0)
    b_param = v.add_param(m, lb=-0.5, ub=0.5)

    out = v.prox(LeastSquares(A=A, b=b_param), arg, gamma=0.5)
    assert out.is_leaf
    assert out.ndim == n


def test_leastsquares_end_to_end_mip():
    """Pin arg via tight bounds, solve, assert u matches the closed form."""
    A = np.array([[1.0, 2.0], [3.0, 1.0]])
    b = np.array([0.5, -0.5])
    gamma = 0.5
    arg_value = np.array([0.3, -0.2])

    v = _make_verifier("mip")
    arg = v.add_param(2, lb=arg_value, ub=arg_value)
    u = v.prox(LeastSquares(A=A, b=b), arg, gamma=gamma)

    expected = np.linalg.solve(np.eye(2) + gamma * A.T @ A, arg_value + gamma * A.T @ b)

    solver = v.canonicalizer.solver
    for i in range(2):
        solver.set_objective(u[i], "minimize")
        solver.optimize()
        assert solver.is_optimal()
        u_min = solver["objVal"]

        solver.set_objective(u[i], "maximize")
        solver.optimize()
        assert solver.is_optimal()
        u_max = solver["objVal"]

        assert np.isclose(u_min, u_max, atol=1e-6)
        assert np.isclose(u_min, expected[i], atol=1e-6)


def test_leastsquares_rejects_nonmatching_b_shape():
    """b must have length equal to A.shape[0]."""
    with pytest.raises(ValueError, match="shape"):
        LeastSquares(A=np.eye(3), b=np.zeros(4))


def test_leastsquares_rejects_1d_A():
    """A must be 2-D."""
    with pytest.raises(ValueError, match="2-D"):
        LeastSquares(A=np.array([1.0, 2.0, 3.0]), b=np.array([0.0]))


# --- sparse Q / A support (carried end-to-end across all backends) ----------


def test_quadratic_sparse_Q_preserved_and_validated():
    """A scipy.sparse Q is accepted, normalized to CSR, and PSD-validated."""
    rng = np.random.default_rng(0)
    diag = rng.uniform(0.5, 3.0, size=4)
    f = Quadratic(Q=spa.diags(diag), q=rng.normal(size=4))
    assert spa.issparse(f.Q) and f.Q.format == "csr"

    # Symmetry / PSD validation matches QuadraticProgram (same messages).
    with pytest.raises(ValueError, match="symmetric"):
        Quadratic(Q=spa.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0]])), q=np.zeros(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        Quadratic(Q=spa.diags([1.0, -2.0]), q=np.zeros(2))


def test_quadratic_sparse_numpy_reference_matches_dense():
    """eval / grad / prox agree whether Q is built sparse or dense."""
    rng = np.random.default_rng(1)
    n, gamma = 5, 0.7
    diag = rng.uniform(0.5, 3.0, size=n)
    q = rng.normal(size=n)
    x = rng.normal(size=n)
    fs = Quadratic(Q=spa.diags(diag), q=q)
    fd = Quadratic(Q=np.diag(diag), q=q)
    assert np.isclose(fs.eval(x), fd.eval(x))
    np.testing.assert_allclose(fs.grad(x), fd.grad(x), atol=1e-12)
    np.testing.assert_allclose(fs.prox(x, gamma=gamma), fd.prox(x, gamma=gamma), atol=1e-10)


def test_leastsquares_sparse_A_preserved_and_matches_dense():
    """A sparse A yields a sparse Q = A^T A; prox / eval match the dense build."""
    rng = np.random.default_rng(2)
    n, m, gamma = 4, 6, 0.9
    A = spa.csr_matrix(rng.normal(size=(m, n)))
    b = rng.normal(size=m)
    x = rng.normal(size=n)
    ls = LeastSquares(A=A, b=b)
    lsd = LeastSquares(A=A.toarray(), b=b)
    assert spa.issparse(ls.Q)
    np.testing.assert_allclose(ls.prox(x, gamma=gamma), lsd.prox(x, gamma=gamma), atol=1e-10)
    assert np.isclose(ls.eval(x), lsd.eval(x))


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_quadratic_sparse_Q_matches_dense_all_backends(method):
    """A sparse-Q prox produces the same pinned solution as the dense build on
    every backend (the prox is a single (I + gamma*Q) x = arg - gamma*q solve).

    MIP/simulator extract the primal exactly; SDP has no primal extraction, so the
    relaxation is checked via its (tight, at a fixed arg) 2-norm objective bound.
    """
    rng = np.random.default_rng(3)
    n, gamma = 3, 0.5
    diag = rng.uniform(0.5, 3.0, size=n)
    q = rng.normal(size=n)
    arg_value = rng.uniform(-1.0, 1.0, size=n)
    expected = np.linalg.solve(np.eye(n) + gamma * np.diag(diag), arg_value - gamma * q)

    obbt = "gurobi" if method == "mip" else None
    v = _make_verifier(method, obbt=obbt)

    if method == "simulator":
        arg = v.add_param(ndim=n, samples=arg_value[None, :])
        u = v.prox(Quadratic(Q=spa.diags(diag), q=q), arg, gamma=gamma)
        ref = v.add_param(ndim=n, samples=expected[None, :])
        v.set_infinity_norm_objective(u - ref)
        assert v.solve()[:, -1].max() < 1e-5
        return

    if method == "sdp":
        arg = v.add_param(n, lb=arg_value, ub=arg_value)
        u = v.prox(Quadratic(Q=spa.diags(diag), q=q), arg, gamma=gamma)
        v.set_objective_norm(u, 2)
        np.testing.assert_allclose(v.solve(), float(expected @ expected), atol=1e-4)
        return

    arg = v.add_param(n, lb=arg_value, ub=arg_value)
    u = v.prox(Quadratic(Q=spa.diags(diag), q=q), arg, gamma=gamma)
    solver = v.canonicalizer.solver
    for i in range(n):
        solver.set_objective(u[i], "minimize")
        solver.optimize()
        assert solver.is_optimal()
        np.testing.assert_allclose(solver["objVal"], expected[i], atol=1e-5)


# --- SumPositive ---


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_sumpositive_returns_linear_expression(method):
    """
    SumPositive lowers to one SaturatedLinear (clip to [0, gamma]) followed
    by a subtraction. All three backends accept this.
    """
    v = _make_verifier(method)
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SumPositive(), arg, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_sumpositive_emits_one_satlin_on_mip():
    """One SaturatedLinear with l == 0, u == gamma; nothing else."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    gamma = 0.5
    v.prox(SumPositive(), arg, gamma=gamma)

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
    arg = np.array([2.0, 0.3, -0.5, -2.0, 0.0])
    gamma = 1.0
    expected = np.array([1.0, 0.0, -0.5, -2.0, 0.0])
    np.testing.assert_allclose(f.prox(arg, gamma=gamma), expected)

    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(6)
        g = float(rng.uniform(0.1, 2.0))
        out = f.prox(x, gamma=g)
        expected = np.where(x > g, x - g, np.where(x < 0, x, 0.0))
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
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(SumNegative(), arg, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_sumnegative_emits_one_satlin_on_mip():
    """One SaturatedLinear with l == -gamma, u == 0; nothing else."""
    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    gamma = 0.5
    v.prox(SumNegative(), arg, gamma=gamma)

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
    arg = np.array([2.0, 0.3, -0.5, -2.0, 0.0])
    gamma = 1.0
    expected = np.array([2.0, 0.3, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(f.prox(arg, gamma=gamma), expected)

    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.standard_normal(6)
        g = float(rng.uniform(0.1, 2.0))
        out = f.prox(x, gamma=g)
        expected = np.where(x < -g, x + g, np.where(x > 0, x, 0.0))
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
        np.testing.assert_allclose(f_neg.prox(v_vec, gamma=g), -f_pos.prox(-v_vec, gamma=g))


def test_unsupported_prox_raises_on_every_backend():
    """
    A ProxableFunction subclass not registered in any _PROX_DECOMPOSERS dict
    raises NotImplementedError with a backend-named message on every backend.
    """

    class Unregistered(ProxableFunction):
        pass

    for method in ("mip", "sdp", "simulator"):
        v = _make_verifier(method)
        arg = v.add_param(3, lb=-1.0, ub=1.0)

        with pytest.raises(NotImplementedError, match="Unregistered"):
            v.prox(Unregistered(), arg)


def test_unsupported_prox_error_names_the_backend():
    """The error message identifies which decomposer rejected the prox."""

    class Unregistered(ProxableFunction):
        pass

    v = _make_verifier("mip")
    arg = v.add_param(3, lb=-1.0, ub=1.0)

    with pytest.raises(NotImplementedError, match="MIPStepDecomposer"):
        v.prox(Unregistered(), arg)
