"""
Tests for partial-projection ReLU on the SDP and Simulator canonicalizers.

The existing ``tests/test_partial_relu.py`` exercises MIP only. This file
covers:
- Structural invariants of the composite lhs (Step API surface).
- SDP-side: only ``len(proj_indices)`` fresh SDP rows are allocated, lifted
  equalities are emitted only for projected outputs.
- Simulator-side: activation honors ``proj_indices``, non-proj rows are
  passed through.
- End-to-end PGD parity between simulator and SDP-vs-MIP.
"""

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.steps import Relu
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# ---------------------------------------------------------------------------
# Structural / unit tests (no canonicalizer)
# ---------------------------------------------------------------------------


def _leaf(ndim, lb=-1.0, ub=1.0):
    return LinearExpression.from_int(ndim, lb=np.full(ndim, lb), ub=np.full(ndim, ub))


def test_partial_relu_lhs_is_composite():
    x = _leaf(3)
    step = Relu(rhs=x, proj_ranges=(1, 3))
    assert step._lhs_proj_leaf is not None
    assert step._lhs_proj_leaf.ndim == 2
    assert step.lhs.is_leaf is False
    assert step.lhs.ndim == 3


def test_partial_relu_lhs_shares_rhs_for_nonproj():
    """Critical invariant: non-proj rows of step.lhs are structurally
    identical to the corresponding rows of step.rhs (same decomposition)."""
    x = _leaf(3)
    step = Relu(rhs=x, proj_ranges=(1, 3))
    # proj_indices = {1, 2}, so coord 0 is non-proj.
    lhs_row0 = step.lhs[0]
    rhs_row0 = step.rhs[0]
    # Both should resolve to the same coefficient on x (an identity row at 0).
    assert set(lhs_row0.decomposition_dict.keys()).issuperset({x})
    coeff_lhs = lhs_row0.decomposition_dict[x]
    coeff_rhs = rhs_row0.decomposition_dict[x]
    if spa.issparse(coeff_lhs):
        coeff_lhs = coeff_lhs.toarray()
    if spa.issparse(coeff_rhs):
        coeff_rhs = coeff_rhs.toarray()
    assert np.allclose(coeff_lhs, coeff_rhs)


def test_partial_relu_lhs_proj_rows_use_fresh_leaf():
    """Proj rows of the composite lhs reference _lhs_proj_leaf with identity."""
    x = _leaf(3)
    step = Relu(rhs=x, proj_ranges=(1, 3))
    # proj_indices = {1, 2} → positions 0, 1 of _lhs_proj_leaf.
    coeff = step.lhs.decomposition_dict[step._lhs_proj_leaf]
    if spa.issparse(coeff):
        coeff = coeff.toarray()
    expected = np.zeros((3, 2))
    expected[1, 0] = 1.0
    expected[2, 1] = 1.0
    assert np.allclose(coeff, expected)


def test_full_relu_lhs_is_leaf_regression():
    x = _leaf(3)
    step = Relu(rhs=x)  # no proj_ranges → full
    assert step._lhs_proj_leaf is None
    assert step.lhs.is_leaf is True
    assert step.lhs.ndim == 3
    assert step.lhs_leaves == [step.lhs]


def test_partial_relu_proj_leaf_bounds():
    """_lhs_proj_leaf bounds are max(0, rhs_proj_bounds)."""
    x = _leaf(3, lb=-1.0, ub=2.0)
    step = Relu(rhs=x, proj_ranges=(1, 3))
    assert np.allclose(step._lhs_proj_leaf.lb, [0.0, 0.0])
    assert np.allclose(step._lhs_proj_leaf.ub, [2.0, 2.0])


def test_get_ranges_does_not_mutate_caller_list():
    """``_get_ranges`` must not mutate the caller-owned proj_ranges list."""
    ranges = [(1, 3), (5, 8)]
    snapshot = list(ranges)
    result = Relu._get_ranges(ranges)
    assert ranges == snapshot
    assert result == {1, 2, 5, 6, 7}


def test_get_ranges_single_tuple():
    """Single (start, end) pair recovers the half-open integer range."""
    assert Relu._get_ranges((4, 7)) == {4, 5, 6}


def test_get_ranges_tuple_of_tuples():
    """A tuple of (start, end) pairs is accepted as an iterable."""
    assert Relu._get_ranges(((1, 2), (5, 9))) == {1, 5, 6, 7, 8}


# ---------------------------------------------------------------------------
# SDP integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sdp_solver", ["clarabel"])
def test_sdp_partial_relu_var_count(sdp_solver):
    """SDP allocates only len(proj_indices) fresh rows for partial relu."""
    v = Verifier(method="sdp", solver=sdp_solver, obbt_handler=None)
    x = v.add_iterate(3, lb=-1.0, ub=1.0)
    pre_marker = v.canonicalizer.range_marker
    step_lhs = v.step.relu(x, proj_ranges=(1, 3))
    delta = v.canonicalizer.range_marker - pre_marker
    # Without the composite-lhs optimization, delta would equal rhs.ndim = 3.
    # With it, only the proj-leaf rows are fresh (= 2).
    assert delta == 2
    # And the composite lhs is what got returned.
    assert step_lhs.is_leaf is False


@pytest.mark.parametrize("sdp_solver", ["clarabel"])
def test_sdp_partial_relu_aeq_count(sdp_solver):
    """canonicalize_relu emits len(proj_indices) lifted equalities, not rhs.ndim."""
    v = Verifier(method="sdp", solver=sdp_solver, obbt_handler=None)
    x = v.add_iterate(4, lb=-1.0, ub=1.0)
    pre = len(v.canonicalizer.Aeq_matrices)
    v.step.relu(x, proj_ranges=(1, 3))
    delta = len(v.canonicalizer.Aeq_matrices) - pre
    # Full relu would emit 4 equalities (one per output); partial emits only 2.
    assert delta == 2


@pytest.mark.parametrize("sdp_solver", ["clarabel"])
def test_sdp_full_relu_unchanged_var_count(sdp_solver):
    """Regression: full relu still allocates rhs.ndim rows."""
    v = Verifier(method="sdp", solver=sdp_solver, obbt_handler=None)
    x = v.add_iterate(3, lb=-1.0, ub=1.0)
    pre = v.canonicalizer.range_marker
    v.step.relu(x)
    delta = v.canonicalizer.range_marker - pre
    assert delta == 3


def test_sdp_partial_relu_bound_dominates_mip():
    """SDP relaxation residual ≥ MIP residual for a partial-relu problem."""
    np.random.seed(0)
    n = 3
    proj_ranges = (1, 3)

    def build(method, solver, obbt_handler):
        kwargs = {"solver": solver, "obbt_handler": obbt_handler}
        if method is None:
            kwargs["params"] = {"OutputFlag": 0}
        else:
            kwargs["method"] = method
        v = Verifier(**kwargs)
        x = v.add_iterate(n, lb=-1.0, ub=1.0)
        y = v.step.relu(x, proj_ranges=proj_ranges)
        v.set_infinity_norm_objective([y])
        return v.solve()

    mip_res = build(None, "gurobi", "gurobi")
    sdp_res = build("sdp", "clarabel", None)
    assert sdp_res >= mip_res - TEST_EPS


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------


def _partial_relu_np(x, proj_ranges):
    out = x.copy()
    s, e = proj_ranges
    out[..., s:e] = np.maximum(out[..., s:e], 0)
    return out


def test_simulator_partial_relu_runs():
    """Multi-iter partial-relu PGD with the default residual path.

    The default residual now compares the *full composite* outputs across
    iterations, so partial-projection ReLU works even when the caller does
    not register an explicit residual spec.
    """
    np.random.seed(42)
    K = 5
    n = 3
    step_size = 0.05

    vp = Verifier(method="simulator", obbt_handler=None)
    z = [None] * (K + 1)
    z[0] = vp.add_iterate(ndim=n, lb=-10.0, ub=10.0, init=np.full(n, 1.0))

    P = np.random.normal(size=(n, n))
    P = P @ P.T

    rng = np.random.default_rng(0)
    q_samples = rng.uniform(low=-1.0, high=1.0, size=(50, n))
    q_param = vp.add_param(ndim=n, lb=-1.0, ub=1.0, samples=q_samples)

    for k in range(1, K + 1):
        z[k] = vp.step.relu(
            z[k - 1] - step_size * (P @ z[k - 1] + q_param),
            proj_ranges=(1, 3),
        )

    residuals = vp.solve()
    assert residuals.shape[0] == len(q_samples)
    assert residuals.shape[1] == K
    # PGD with partial relu still produces monotonic-ish residual decay.
    assert residuals[0, -1] <= residuals[0, 0] + TEST_EPS


def test_simulator_partial_relu_default_residual_matches_explicit():
    """Default residual (composite-based) and explicit ``z[k]-z[k-1]`` spec
    must agree numerically. Pins the contract that the default path computes
    the convergence residual on the full composite output, not on the small
    fresh proj-leaf.
    """
    np.random.seed(7)
    K = 4
    n = 3
    step_size = 0.05
    P = np.random.normal(size=(n, n))
    P = P @ P.T
    rng = np.random.default_rng(0)
    q_samples = rng.uniform(low=-1.0, high=1.0, size=(20, n))

    def run(with_explicit_residual: bool):
        vp = Verifier(method="simulator", obbt_handler=None)
        z = [None] * (K + 1)
        z[0] = vp.add_iterate(ndim=n, lb=-10.0, ub=10.0, init=np.full(n, 1.0))
        q = vp.add_param(ndim=n, lb=-1.0, ub=1.0, samples=q_samples)
        for k in range(1, K + 1):
            z[k] = vp.step.relu(
                z[k - 1] - step_size * (P @ z[k - 1] + q),
                proj_ranges=(1, 3),
            )
            if with_explicit_residual:
                vp.set_objective_norm(z[k] - z[k - 1], np.inf)
        return np.asarray(vp.solve())

    default = run(with_explicit_residual=False)
    explicit = run(with_explicit_residual=True)
    assert default.shape == explicit.shape
    assert np.allclose(default, explicit, atol=1e-4)


def test_simulator_partial_relu_differs_from_full():
    """Partial-relu output diverges from full-relu output when rhs has a
    negative coord in a non-proj position (full would clamp to 0; partial
    must pass it through)."""
    np.random.seed(0)
    n = 3
    rng = np.random.default_rng(0)
    q_samples = rng.uniform(low=-1.0, high=1.0, size=(20, n))

    def run(proj_ranges):
        vp = Verifier(method="simulator", obbt_handler=None)
        z0 = vp.add_iterate(ndim=n, lb=-2.0, ub=2.0, init=np.full(n, -0.5))
        q = vp.add_param(ndim=n, lb=-1.0, ub=1.0, samples=q_samples)
        kwargs = {} if proj_ranges is None else {"proj_ranges": proj_ranges}
        z1 = vp.step.relu(z0 + q, **kwargs)
        vp.set_objective_norm(z1 - z0, np.inf)
        return vp.solve()

    full_res = run(None)
    partial_res = run((1, 3))  # coord 0 passes through; 1,2 are projected
    # The residual streams must differ — there exists at least one sample
    # whose first coord is negative, where full clamps to 0 but partial leaves
    # it negative, producing different residuals.
    assert not np.allclose(full_res, partial_res, atol=TEST_EPS)


# ---------------------------------------------------------------------------
# Cross-backend end-to-end
# ---------------------------------------------------------------------------


def test_partial_relu_sdp_geq_mip_pgd():
    """Reuse the partial-NNQP setup: one PGD iter, SDP residual ≥ MIP."""
    np.random.seed(0)
    n = 3
    proj_ranges = (1, 3)
    P = np.random.normal(size=(n, n))
    P = P @ P.T
    q = np.random.normal(size=(n,))
    t = 0.01
    q_offset = 0.5

    def build(method, solver, obbt_handler):
        kwargs = {"solver": solver, "obbt_handler": obbt_handler}
        if method is None:
            kwargs["params"] = {"OutputFlag": 0}
        else:
            kwargs["method"] = method
        v = Verifier(**kwargs)
        q_param = v.add_param(n, lb=q - q_offset, ub=q + q_offset)
        z0 = v.add_iterate(n, lb=-1.0, ub=-1.0)
        z1 = v.step.relu(z0 - t * (P @ z0 + q_param), proj_ranges=proj_ranges)
        v.set_infinity_norm_objective([z1 - z0])
        return v.solve()

    mip_res = build(None, "gurobi", "gurobi")
    sdp_res = build("sdp", "clarabel", None)
    assert sdp_res >= mip_res - TEST_EPS
