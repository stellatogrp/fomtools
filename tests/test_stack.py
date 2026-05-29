import numpy as np
import pytest
import scipy.sparse as spa

import fomtools as vp
from fomtools.linearexpression import LinearExpression, stack
from fomtools.verifier import Verifier
from tests.conftest import SOLVERS, TEST_EPS


def _leaf(ndim, lb=-1.0, ub=1.0):
    """Construct a leaf LinearExpression with array-valued bounds.

    ``LinearExpression.from_int`` stores ``lb``/``ub`` as-is, but
    ``bound_propagation`` expects arrays. ``Verifier._create_var``
    broadcasts before calling ``from_int``; for tests that don't need
    a full Verifier, this helper does the same. Defaults are finite to
    keep ``bound_propagation`` from emitting ``inf - inf`` warnings on
    construction (tests that care about bound values pass them explicitly).
    """
    return LinearExpression.from_int(ndim, lb=np.full(ndim, lb), ub=np.full(ndim, ub))


# ---------------------------------------------------------------------------
# Pure expression-level tests (no canonicalizer)
# ---------------------------------------------------------------------------


def test_stack_two_leaves_basic_shape():
    x = _leaf(3)
    nu = _leaf(2)
    y = stack([x, nu])

    assert y.ndim == 5
    assert not y.is_leaf
    assert set(y.decomposition_dict.keys()) == {x, nu}
    assert np.allclose(y.const, np.zeros(5))


def test_stack_preserves_selection_matrices():
    x = _leaf(3)
    nu = _leaf(2)
    y = stack([x, nu])

    coeff_x = y.decomposition_dict[x]
    coeff_nu = y.decomposition_dict[nu]
    if spa.issparse(coeff_x):
        coeff_x = coeff_x.toarray()
    if spa.issparse(coeff_nu):
        coeff_nu = coeff_nu.toarray()

    expected_x = np.vstack([np.eye(3), np.zeros((2, 3))])
    expected_nu = np.vstack([np.zeros((3, 2)), np.eye(2)])

    assert np.allclose(coeff_x, expected_x)
    assert np.allclose(coeff_nu, expected_nu)


def test_stack_with_const_offsets():
    x = _leaf(3)
    nu = _leaf(2)
    y = stack([x + np.ones(3), nu - np.ones(2)])

    assert np.allclose(y.const, np.array([1, 1, 1, -1, -1]))


def test_stack_dedupes_shared_leaves():
    x = _leaf(2)
    y = stack([x, 2 * x])

    assert set(y.decomposition_dict.keys()) == {x}
    coeff = y.decomposition_dict[x]
    if spa.issparse(coeff):
        coeff = coeff.toarray()
    expected = np.vstack([np.eye(2), 2.0 * np.eye(2)])
    assert np.allclose(coeff, expected)


def test_stack_of_derived_expressions():
    x = _leaf(2)
    nu = _leaf(3)
    e1 = 2 * x + np.array([1.0, -1.0])
    e2 = -nu
    y = stack([e1, e2])

    assert y.ndim == 5
    assert set(y.decomposition_dict.keys()) == {x, nu}
    coeff_x = y.decomposition_dict[x]
    coeff_nu = y.decomposition_dict[nu]
    if spa.issparse(coeff_x):
        coeff_x = coeff_x.toarray()
    if spa.issparse(coeff_nu):
        coeff_nu = coeff_nu.toarray()

    expected_x = np.vstack([2.0 * np.eye(2), np.zeros((3, 2))])
    expected_nu = np.vstack([np.zeros((2, 3)), -np.eye(3)])
    assert np.allclose(coeff_x, expected_x)
    assert np.allclose(coeff_nu, expected_nu)
    assert np.allclose(y.const, np.array([1.0, -1.0, 0.0, 0.0, 0.0]))


def test_stack_slicing_recovers_components():
    x = _leaf(3)
    nu = _leaf(2)
    y = stack([x, nu])

    top = y[0:3]
    bot = y[3:5]
    coeff_top_x = top.decomposition_dict[x]
    coeff_bot_nu = bot.decomposition_dict[nu]
    if spa.issparse(coeff_top_x):
        coeff_top_x = coeff_top_x.toarray()
    if spa.issparse(coeff_bot_nu):
        coeff_bot_nu = coeff_bot_nu.toarray()

    assert np.allclose(coeff_top_x, np.eye(3))
    assert np.allclose(coeff_bot_nu, np.eye(2))


def test_stack_bounds_are_concatenation():
    x = _leaf(3, lb=-1.0, ub=2.0)
    nu = _leaf(2, lb=-3.0, ub=4.0)
    y = stack([x, nu])

    assert np.allclose(y.lb, np.array([-1.0, -1.0, -1.0, -3.0, -3.0]))
    assert np.allclose(y.ub, np.array([2.0, 2.0, 2.0, 4.0, 4.0]))


def test_stack_bounds_refresh_after_leaf_tightening():
    x = _leaf(2, lb=-1.0, ub=1.0)
    nu = _leaf(2, lb=-1.0, ub=1.0)
    y = stack([x, nu])

    # Tighten leaf bounds directly; cached stack bounds stay stale until refresh.
    x.lb = np.array([0.0, 0.0])
    x.ub = np.array([0.5, 0.5])

    new_lb, new_ub = y.bound_propagation()
    assert np.allclose(new_lb, np.array([0.0, 0.0, -1.0, -1.0]))
    assert np.allclose(new_ub, np.array([0.5, 0.5, 1.0, 1.0]))


def test_stack_empty_raises():
    with pytest.raises(ValueError):
        stack([])


def test_stack_prunes_dead_leaves():
    """A component whose decomposition carries an all-zero coefficient on
    some leaf must not propagate that leaf into the stack's decomposition.
    Arithmetic ops already prune, so we hand-build a LinearExpression with
    a dead entry to exercise the prune in ``stack()`` directly.
    """
    x = _leaf(2, lb=-1.0, ub=1.0)
    y = _leaf(2, lb=-1.0, ub=1.0)
    component = LinearExpression(
        decomposition_dict={
            x: spa.eye(2),
            y: spa.csc_matrix((2, 2)),  # all-zero — dead entry
        },
        const=np.zeros(2),
    )
    result = stack([component])
    assert x in result.decomposition_dict
    assert y not in result.decomposition_dict


def test_stack_single_expression_is_identity_view():
    x = _leaf(4)
    y = stack([x])

    assert y.ndim == x.ndim
    coeff = y.decomposition_dict[x]
    if spa.issparse(coeff):
        coeff = coeff.toarray()
    assert np.allclose(coeff, np.eye(4))


def test_stack_exposed_at_package_level():
    assert vp.stack is stack


# ---------------------------------------------------------------------------
# Canonicalizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_stack_through_relu_mip(solver):
    """
    Build two equivalent MIPs: one passes a stacked LinearExpression into
    relu, the other constructs the relu input via direct concatenation
    arithmetic. Both must produce the same residual under the inf-norm
    objective (HiGHS-compatible).
    """
    x_lb = np.array([-1.0, -2.0])
    x_ub = np.array([1.0, 0.5])
    nu_lb = np.array([-0.5, -1.5])
    nu_ub = np.array([2.0, 1.0])

    def build(use_stack: bool):
        v = Verifier(solver=solver, obbt_handler=solver, params={"OutputFlag": 0})
        x = v.add_iterate(2, lb=x_lb, ub=x_ub)
        nu = v.add_iterate(2, lb=nu_lb, ub=nu_ub)
        if use_stack:
            y = stack([x, nu])
        else:
            sel_x = spa.csc_matrix(
                (np.ones(2), (np.arange(2), np.arange(2))), shape=(4, 2)
            )
            sel_nu = spa.csc_matrix(
                (np.ones(2), (np.arange(2, 4), np.arange(2))), shape=(4, 2)
            )
            y = sel_x @ x + sel_nu @ nu
        z = v.step.relu(rhs=y)
        v.set_infinity_norm_objective([z])
        return v.solve()

    r_stack = build(True)
    r_direct = build(False)
    assert r_stack == pytest.approx(r_direct, abs=TEST_EPS)


@pytest.mark.parametrize("sdp_solver", ["clarabel"])
def test_stack_through_relu_sdp(sdp_solver):
    """
    SDP version: pass a stacked LinearExpression into relu and confirm the
    SDP relaxation produces a sane upper bound (≥ the MIP optimum, since SDP
    is a relaxation; per memory note ``feedback_sdp_no_solution_value`` we
    compare bounds, not solution values).
    """
    x_lb = np.array([-1.0, -2.0])
    x_ub = np.array([1.0, 0.5])
    nu_lb = np.array([-0.5, -1.5])
    nu_ub = np.array([2.0, 1.0])

    v_sdp = Verifier(method="sdp", solver=sdp_solver, obbt_handler=None)
    x = v_sdp.add_iterate(2, lb=x_lb, ub=x_ub)
    nu = v_sdp.add_iterate(2, lb=nu_lb, ub=nu_ub)
    y = stack([x, nu])
    z = v_sdp.step.relu(rhs=y)
    v_sdp.set_two_norm_objective([z])
    sdp_bound = v_sdp.solve()

    v_mip = Verifier(solver="gurobi", obbt_handler="gurobi", params={"OutputFlag": 0})
    x_m = v_mip.add_iterate(2, lb=x_lb, ub=x_ub)
    nu_m = v_mip.add_iterate(2, lb=nu_lb, ub=nu_ub)
    y_m = stack([x_m, nu_m])
    z_m = v_mip.step.relu(rhs=y_m)
    v_mip.set_two_norm_objective([z_m])
    mip_residual = v_mip.solve()

    assert sdp_bound >= mip_residual - TEST_EPS


@pytest.mark.parametrize("solver", SOLVERS)
def test_stack_through_affine(solver):
    """
    Pass a stacked expression into an affine step. Build the equivalent
    problem without stack() and confirm residuals match.
    """
    np.random.seed(1)
    x_lb = np.array([-1.0, -1.0])
    x_ub = np.array([1.0, 1.0])
    nu_lb = np.array([-1.0, -1.0])
    nu_ub = np.array([1.0, 1.0])
    M = np.random.randn(3, 4)

    def build(use_stack):
        v = Verifier(solver=solver, obbt_handler=solver, params={"OutputFlag": 0})
        x = v.add_iterate(2, lb=x_lb, ub=x_ub)
        nu = v.add_iterate(2, lb=nu_lb, ub=nu_ub)
        if use_stack:
            y = stack([x, nu])
            w = v.step.implicit_linear(rhs=M @ y, lhs_mat=np.eye(3))
        else:
            sel_x = spa.csc_matrix(
                (np.ones(2), (np.arange(2), np.arange(2))), shape=(4, 2)
            )
            sel_nu = spa.csc_matrix(
                (np.ones(2), (np.arange(2, 4), np.arange(2))), shape=(4, 2)
            )
            y = sel_x @ x + sel_nu @ nu
            w = v.step.implicit_linear(rhs=M @ y, lhs_mat=np.eye(3))
        v.set_infinity_norm_objective([w])
        return v.solve()

    r_stack = build(True)
    r_direct = build(False)
    assert r_stack == pytest.approx(r_direct, abs=TEST_EPS)


@pytest.mark.parametrize("solver", SOLVERS)
def test_stack_inherits_leaf_constraint(solver):
    """
    Critical invariant: pinning the leaf ``x`` via its construction bounds
    must apply to the corresponding slice of the stack with no explicit
    constraint on that slice. We pin ``x = c`` at ``add_iterate`` time and
    leave ``nu`` constrained loosely; the relu objective over ``stack([x, nu])``
    must match the case where the relu rhs is built without ``stack()``.
    """
    c = np.array([-0.5, 0.7])
    nu_lb = np.array([-0.3, -0.4])
    nu_ub = np.array([0.6, 0.5])

    def build(use_stack):
        v = Verifier(solver=solver, obbt_handler=solver, params={"OutputFlag": 0})
        x = v.add_iterate(2, lb=c, ub=c)  # pinned at construction
        nu = v.add_iterate(2, lb=nu_lb, ub=nu_ub)
        if use_stack:
            y = stack([x, nu])
        else:
            sel_x = spa.csc_matrix(
                (np.ones(2), (np.arange(2), np.arange(2))), shape=(4, 2)
            )
            sel_nu = spa.csc_matrix(
                (np.ones(2), (np.arange(2, 4), np.arange(2))), shape=(4, 2)
            )
            y = sel_x @ x + sel_nu @ nu
        z = v.step.relu(rhs=y)
        v.set_infinity_norm_objective([z])
        return v.solve()

    r_stack = build(True)
    r_direct = build(False)
    # Both branches see the pinned x constraint; results must coincide.
    assert r_stack == pytest.approx(r_direct, abs=TEST_EPS)
    # Sanity: the upper objective from the loose nu side dominates the relu(c)
    # side (since c has a positive coord 0.7, but nu's upper is 0.6 → max relu
    # output is 0.7 from the x-side coord).
    expected = max(np.maximum(c, 0.0).max(), np.maximum(nu_ub, 0.0).max())
    assert r_stack == pytest.approx(expected, abs=TEST_EPS)


@pytest.mark.filterwarnings("ignore:invalid value encountered in add:RuntimeWarning")
@pytest.mark.parametrize("solver", SOLVERS)
def test_obbt_tightens_unbounded_dual_via_stack(solver):
    """
    KKT-style use case: a dual variable ``nu`` starts unbounded (lb=-inf,
    ub=inf). After adding a coupling constraint ``M @ stack([x, nu]) == 0``
    with ``x`` bounded, OBBT must infer finite bounds on ``nu`` from the
    constraint plus ``x``'s bounds, propagate them back to the original
    ``nu`` leaf, and (via leaf-driven bound_propagation) tighten the stack
    too.

    Concretely we use ``M = [I  I]`` so the equality reads ``x + nu == 0``,
    i.e. ``nu = -x``. With ``x in [-1, 1]^2`` the tight bounds on ``nu``
    are ``[-1, 1]^2``.
    """
    v = Verifier(solver=solver, obbt_handler=solver, params={"OutputFlag": 0})
    x = v.add_iterate(2, lb=-1.0, ub=1.0)
    nu = v.add_iterate(2, lb=-np.inf, ub=np.inf)

    # Sanity: nu is initially unbounded.
    assert np.all(np.isinf(nu.lb))
    assert np.all(np.isinf(nu.ub))

    M = np.hstack([np.eye(2), np.eye(2)])  # shape (2, 4): [x; nu] -> x + nu
    y = stack([x, nu])

    v.add_constraint(M @ y == np.zeros(2))

    # The leaf nu is now bounded by OBBT, derived purely from x's bounds and
    # the coupling constraint.
    assert np.all(np.isfinite(nu.lb))
    assert np.all(np.isfinite(nu.ub))
    assert np.allclose(nu.lb, -1.0, atol=TEST_EPS)
    assert np.allclose(nu.ub, 1.0, atol=TEST_EPS)

    # The stack's bound_propagation reads current leaf bounds, so a fresh
    # call surfaces the tightening on its second half. The first half stays
    # at x's existing [-1, 1].
    new_lb, new_ub = y.bound_propagation()
    assert np.allclose(new_lb, np.array([-1.0, -1.0, -1.0, -1.0]), atol=TEST_EPS)
    assert np.allclose(new_ub, np.array([1.0, 1.0, 1.0, 1.0]), atol=TEST_EPS)
