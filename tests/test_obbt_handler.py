"""
Tests for the OBBT handler as an independent LP canonicalizer.

These cover the fix for the OBBT "architectural leak" (the handler held every
variable but no step relaxation constraints, so bound tightening was a no-op).
They verify that, with the per-primitive OBBT canonicalization hook:

1. OBBT actually tightens step input bounds (vs interval propagation only).
2. OBBT never changes the optimal value of a MIP verification problem.
3. OBBT decreases-or-equals the optimal value of an SDP verification problem.
4. OBBT tightens SDP decomposition auxiliary leaves (the Option-1 guarantee).
"""

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.canonicalizers.sdp import SDP
from fomtools.steps import Affine
from fomtools.utils.math_utils import repair_crossed_bounds
from fomtools.utils.matrix_utils import is_identity_matrix
from fomtools.verifier import Verifier
from tests.conftest import (
    SDP_SOLVERS,
    SOLVERS,
    TEST_EPS,
    build_lasso_ista,
    mip_params,
    step_input_widths,
)

# --------------------------------------------------------------------------
# Local builders for ReLU / saturated-linear chains (constant bounds so the
# saturated-linear step stays a MIP primitive and exercises the OBBT
# canonicalize_saturatedlinear envelope rows).
# --------------------------------------------------------------------------


def build_relu_chain(obbt, solver, *, method="mip", K=3, seed=1, n=3, m=3, t=0.1):
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    W = np.eye(n) - t * A.T @ A
    G = t * A.T
    params = mip_params(solver) if method == "mip" else None
    v = Verifier(method=method, solver=solver, obbt_solver=obbt, params=params)
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    b = v.add_param(m, lb=-1.0 * np.ones(m), ub=1.0 * np.ones(m))
    iterates = [z]
    for _ in range(K):
        z = v.step.relu(arg=W @ z + G @ b)
        iterates.append(z)
    return v, iterates


def build_satlin_chain(obbt, solver, *, method="mip", K=3, seed=2, n=3, m=3, t=0.1):
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    W = np.eye(n) - t * A.T @ A
    G = t * A.T
    l = -0.3 * np.ones(n)
    u = 0.3 * np.ones(n)
    params = mip_params(solver) if method == "mip" else None
    v = Verifier(method=method, solver=solver, obbt_solver=obbt, params=params)
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    b = v.add_param(m, lb=-1.0 * np.ones(m), ub=1.0 * np.ones(m))
    iterates = [z]
    for _ in range(K):
        z = v.step.satlin(arg=W @ z + G @ b, l=l, u=u)
        iterates.append(z)
    return v, iterates


BUILDERS = {
    "softthreshold": build_lasso_ista,
    "relu": build_relu_chain,
    "satlin": build_satlin_chain,
}


def _mip_optimum(v, iterates):
    """Maximize the inf-norm of the last step residual and solve (MILP)."""
    v.set_objective_norm(iterates[-1] - iterates[-2], np.inf)
    return v.solve()


# --------------------------------------------------------------------------
# 1. OBBT tightens soft-threshold input bounds.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_obbt_tightens_softthreshold_input_bounds(solver):
    K = 3
    v_no, _ = build_lasso_ista(obbt=None, solver=solver, K=K)
    v_ob, _ = build_lasso_ista(obbt=solver, solver=solver, K=K)

    w_no = step_input_widths(v_no)
    w_ob = step_input_widths(v_ob)
    assert len(w_no) == len(w_ob) == K

    ratios = []
    for wn, wo in zip(w_no, w_ob):
        # Never widens.
        assert np.all(wo <= wn + TEST_EPS)
        ratios.append(wo.mean() / wn.mean())

    # Step 0 sees only the independent input box, so it cannot beat interval
    # propagation; later steps must strictly tighten.
    assert any(r < 1 - 1e-3 for r in ratios[1:])
    assert np.mean(ratios[1:]) < 1.0


# --------------------------------------------------------------------------
# 2. OBBT preserves the MIP optimum (exactness invariant).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("op", ["softthreshold", "relu", "satlin"])
def test_obbt_preserves_mip_optimum(solver, op):
    build = BUILDERS[op]
    v_no, it_no = build(obbt=None, solver=solver)
    v_ob, it_ob = build(obbt=solver, solver=solver)

    opt_no = _mip_optimum(v_no, it_no)
    opt_ob = _mip_optimum(v_ob, it_ob)
    assert abs(opt_ob - opt_no) <= 1e-4


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("op", ["softthreshold", "relu", "satlin"])
def test_obbt_preserves_mip_optimum_with_convex_hull_cuts(solver, op):
    if solver == "highs":
        pytest.skip("convex_hull_cuts path uses Gurobi-specific relaxation")
    build = BUILDERS[op]
    v_no, it_no = build(obbt=None, solver=solver)
    v_ob, it_ob = build(obbt=solver, solver=solver)

    v_no.set_objective_norm(it_no[-1] - it_no[-2], np.inf)
    v_ob.set_objective_norm(it_ob[-1] - it_ob[-2], np.inf)
    opt_no = v_no.solve(convex_hull_cuts=True)
    opt_ob = v_ob.solve(convex_hull_cuts=True)
    assert abs(opt_ob - opt_no) <= 1e-4


# --------------------------------------------------------------------------
# 3. OBBT decreases-or-equals the SDP optimum (relaxation invariant).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_obbt_decreases_or_equal_sdp_optimum(sdp_solver):
    v_no, it_no = build_lasso_ista(obbt=None, solver=sdp_solver, method="sdp", K=2)
    v_ob, it_ob = build_lasso_ista(obbt="highs", solver=sdp_solver, method="sdp", K=2)

    v_no.set_objective_norm(it_no[-1] - it_no[-2], 2)
    v_ob.set_objective_norm(it_ob[-1] - it_ob[-2], 2)
    opt_no = v_no.solve()
    opt_ob = v_ob.solve()

    assert opt_ob <= opt_no + TEST_EPS
    assert opt_ob >= -TEST_EPS


# --------------------------------------------------------------------------
# 4. OBBT tightens SDP decomposition auxiliary leaves.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_obbt_tightens_sdp_auxiliary_leaves(sdp_solver):
    K = 3
    v_no, _ = build_lasso_ista(obbt=None, solver=sdp_solver, method="sdp", K=K)
    v_ob, _ = build_lasso_ista(obbt="highs", solver=sdp_solver, method="sdp", K=K)

    # Soft-threshold decomposes into ReLU sub-steps under SDP; their output are the
    # auxiliary leaves lifted into the PSD cone.
    relu_no = v_no.canonicalizer._relu_steps
    relu_ob = v_ob.canonicalizer._relu_steps
    assert len(relu_no) == len(relu_ob) > 0

    tightened = False
    for s_no, s_ob in zip(relu_no, relu_ob):
        w_no = s_no.ub - s_no.lb
        w_ob = s_ob.ub - s_ob.lb
        assert np.all(w_ob <= w_no + TEST_EPS)  # never widens
        if np.any(w_ob < w_no - 1e-3):
            tightened = True
    assert tightened, "OBBT did not tighten any SDP auxiliary ReLU leaf"


# --------------------------------------------------------------------------
# 5. Regression: the exact pre-fix signature on the lasso repro.
# --------------------------------------------------------------------------


def test_obbt_lasso_regression():
    K = 3
    v_no, _ = build_lasso_ista(obbt=None, solver="gurobi", K=K)
    v_ob, _ = build_lasso_ista(obbt="gurobi", solver="gurobi", K=K)

    w_no = step_input_widths(v_no)
    w_ob = step_input_widths(v_ob)
    ratios = [wo.mean() / wn.mean() for wn, wo in zip(w_no, w_ob)]

    # Pre-fix all ratios were exactly 1.0 (OBBT was a no-op).
    assert not all(abs(r - 1.0) <= 1e-9 for r in ratios)
    assert min(ratios[1:]) < 1.0


# --------------------------------------------------------------------------
# 6. ReLU / satlin relaxations tighten and preserve the MIP optimum.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("op", ["relu", "satlin"])
def test_obbt_relu_and_satlin_relaxation_tightens(solver, op):
    build = BUILDERS[op]
    K = 3
    v_no, it_no = build(obbt=None, solver=solver, K=K)
    v_ob, it_ob = build(obbt=solver, solver=solver, K=K)

    step_name = {"relu": "Relu", "satlin": "SaturatedLinear"}[op]
    w_no = step_input_widths(v_no, step_name)
    w_ob = step_input_widths(v_ob, step_name)

    for wn, wo in zip(w_no, w_ob):
        assert np.all(wo <= wn + TEST_EPS)  # never widens
    ratios = [wo.mean() / wn.mean() for wn, wo in zip(w_no, w_ob)]
    assert any(r < 1 - 1e-3 for r in ratios[1:])  # a later step shrinks

    # MIP optimum unchanged.
    assert abs(_mip_optimum(v_ob, it_ob) - _mip_optimum(v_no, it_no)) <= 1e-4


# --------------------------------------------------------------------------
# 7. Affine steps participate in OBBT both as inputs and as relaxations.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_affine_obbt_input_tightening(solver):
    """
    Baseline Affine OBBT case: with only an upstream iterate's box in the
    OBBT LP, OBBT cannot improve on interval propagation, so the affine
    input bounds equal ``arg.bound_propagation()``. The affine equality
    is still registered with the OBBT LP, so a downstream ReLU's input
    bounds remain at least as tight as interval propagation.
    """
    from fomtools.steps import Affine

    np.random.seed(3)
    n = 3
    W = np.random.normal(size=(n, n))
    v = Verifier(solver=solver, obbt_solver=solver, params=mip_params(solver))
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    y = v.run_step(z, Affine, weight=W)

    affine_step = v.steps[-1]

    # In this trivial scenario (single box-constrained input, no further
    # constraints in the OBBT LP), OBBT on the affine input equals interval
    # propagation. The point is that this path now *runs* — see the
    # constrained-witness test below for a case where it strictly tightens.
    arg_lb, arg_ub = affine_step.arg.bound_propagation()
    assert np.allclose(affine_step.lb, arg_lb)
    assert np.allclose(affine_step.ub, arg_ub)

    # The affine equality is in the OBBT LP: a downstream ReLU's input can be
    # tightened against it (otherwise OBBT would be a no-op).
    v.step.relu(arg=y)
    relu_step = v.steps[-1]
    relu_lb, relu_ub = relu_step.arg.bound_propagation()
    assert np.all(relu_step.lb >= relu_lb - TEST_EPS)
    assert np.all(relu_step.ub <= relu_ub + TEST_EPS)


@pytest.mark.parametrize("solver", SOLVERS)
def test_affine_obbt_tightens_when_constrained(solver):
    """
    Witness that OBBT input tightening for Affine is not a no-op when there
    is cross-coordinate structure to exploit. Build
    ``iterate -> Affine_1 -> Affine_2``: the equality ``y == W1 @ z`` enters
    the OBBT LP after Affine_1, so OBBT on Affine_2's input ``W2 @ y`` can
    substitute through to ``W2 @ W1 @ z`` and optimize over ``z``'s box
    — strictly tighter than interval propagation, which double-applies
    ``|·|`` across the matrix product.

    Pre-PR (``Affine.skip_obbt = True``), Affine_2's input bounds equalled
    ``arg.bound_propagation()`` exactly. Post-PR they should be tighter or
    equal everywhere, and strictly tighter in at least one coordinate.
    """
    from fomtools.steps import Affine

    np.random.seed(11)
    n = 3
    W1 = np.random.normal(size=(n, n))
    W2 = np.random.normal(size=(n, n))
    v = Verifier(solver=solver, obbt_solver=solver, params=mip_params(solver))
    z = v.add_iterate(n, lb=-1.0 * np.ones(n), ub=1.0 * np.ones(n))
    y = v.run_step(z, Affine, weight=W1)  # Affine_1 equality into OBBT LP
    _ = v.run_step(y, Affine, weight=W2)  # Affine_2 input is what we check

    affine_step = v.steps[-1]
    arg_lb, arg_ub = affine_step.arg.bound_propagation()

    # OBBT never widens.
    assert np.all(affine_step.lb >= arg_lb - TEST_EPS)
    assert np.all(affine_step.ub <= arg_ub + TEST_EPS)

    # OBBT does something: at least one coordinate strictly tightens
    # because Affine_1's equality couples the y coordinates in the OBBT LP.
    tightened = np.any(affine_step.lb > arg_lb + 1e-4) or np.any(affine_step.ub < arg_ub - 1e-4)
    assert tightened, (
        "Expected OBBT to strictly tighten at least one Affine_2 input bound "
        "via the Affine_1 equality in the OBBT LP; got "
        f"lb={affine_step.lb}, ub={affine_step.ub} vs "
        f"arg_lb={arg_lb}, arg_ub={arg_ub}."
    )


# --------------------------------------------------------------------------
# 8. SDP uses OBBT bounds: ReLU cuts, Affine output leaves, crossed bounds.
# --------------------------------------------------------------------------


def build_relu_linsolve_chain(obbt, solver, *, method="mip", K=2, identity=False, **kwargs):
    """ReLU followed by an implicit linear solve ``M w = relu(.) - z`` per iteration."""
    np.random.seed(0)
    n = m = 3
    t = 0.5
    A = np.random.normal(size=(m, n))
    W = np.eye(n) - t * A.T @ A
    G = t * A.T
    M = np.eye(n) if identity else np.eye(n) + t * A.T @ A
    params = mip_params(solver) if method == "mip" else None
    v = Verifier(method=method, solver=solver, obbt_solver=obbt, params=params, **kwargs)
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    b = v.add_param(m, lb=-1.0 * np.ones(m), ub=1.0 * np.ones(m))
    iterates = [z]
    for _ in range(K):
        y = v.step.relu(arg=W @ z + G @ b)
        z = v.step.implicit_linear(arg=y - z, output_mat=M)
        iterates.append(z)
    return v, iterates


def _affine_steps(v):
    return [s for s in v.steps if isinstance(s, Affine)]


def test_repair_crossed_bounds():
    lb = np.array([0.0, 2e-17, -1.0, -np.inf])
    ub = np.array([1.0, 0.0, -1.0, np.inf])
    new_lb, new_ub = repair_crossed_bounds(lb, ub)
    np.testing.assert_array_equal(new_lb, [0.0, 0.0, -1.0, -np.inf])
    np.testing.assert_array_equal(new_ub, [1.0, 2e-17, -1.0, np.inf])


def test_repair_crossed_bounds_rejects_empty_interval():
    with pytest.raises(ValueError, match="cross"):
        repair_crossed_bounds(np.array([1.0]), np.array([-1.0]))
    # The tolerance is relative to the bound magnitude.
    repair_crossed_bounds(np.array([1e6 + 1e-2]), np.array([1e6]), tol=1e-7)
    with pytest.raises(ValueError):
        repair_crossed_bounds(np.array([1e6 + 1.0]), np.array([1e6]), tol=1e-7)


def test_repair_crossed_bounds_buffers_only_crossed_entries():
    lb = np.array([0.5 + 5e-7, 0.0, 3.0 + 2e-6])
    ub = np.array([0.5, 1.0, 3.0])
    new_lb, new_ub = repair_crossed_bounds(lb, ub, tol=1e-6, buffer=1e-6)
    # Crossed entries are swapped and widened by buffer * max(1, |bound|).
    pad = 1e-6 * np.array([1.0, 3.0 + 2e-6])
    np.testing.assert_allclose(new_lb[[0, 2]], ub[[0, 2]] - pad, rtol=0, atol=1e-15)
    np.testing.assert_allclose(new_ub[[0, 2]], lb[[0, 2]] + pad, rtol=0, atol=1e-15)
    # The uncrossed entry is untouched.
    assert (new_lb[1], new_ub[1]) == (0.0, 1.0)
    with pytest.raises(ValueError, match="cross"):
        repair_crossed_bounds(np.array([1e-3]), np.array([0.0]), tol=1e-6, buffer=1e-6)


class _StubLPSolver:
    """Returns a fixed optimal value per (sense, entry) for OBBTHandler.obbt."""

    def __init__(self, values):
        self.values, self.key = values, None

    def set_objective(self, expr, sense=None):
        self.key = (sense, expr)

    def optimize(self):
        pass

    def is_optimal(self):
        return True

    def __getitem__(self, item):
        assert item == "objVal"
        return self.values[self.key]


class _StubExpr(list):
    @property
    def ndim(self):
        return len(self)


def _stub_handler(values):
    from fomtools.canonicalizers.obbt_handler import OBBTHandler

    handler = OBBTHandler.__new__(OBBTHandler)
    handler.solver = _StubLPSolver(values)
    handler.crossing_tol, handler.crossing_repairs = 1e-6, 0
    return handler


def test_obbt_buffers_noise_crossing():
    # Entry "a" is a single point whose LP min lands 5e-7 above its LP max.
    values = {
        ("minimize", "a"): 0.2 + 5e-7,
        ("maximize", "a"): 0.2,
        ("minimize", "b"): -1.0,
        ("maximize", "b"): 1.0,
    }
    handler = _stub_handler(values)
    lb, ub = handler.obbt(_StubExpr(["a", "b"]), np.full(2, -5.0), np.full(2, 5.0))
    assert lb[0] <= 0.2 and ub[0] >= 0.2 + 5e-7
    assert lb[0] >= 0.2 - 2e-6 and ub[0] <= 0.2 + 2e-6
    assert (lb[1], ub[1]) == (-1.0, 1.0)
    assert handler.crossing_repairs == 1


def test_obbt_buffer_does_not_loosen_input_bounds():
    # A ReLU-style output with input bounds [0, 0]: the LP min lands at 2e-17,
    # crossing the input ub. Buffering must not push the result below lb = 0.
    values = {("minimize", "a"): 2e-17, ("maximize", "a"): 2e-17}
    handler = _stub_handler(values)
    lb, ub = handler.obbt(_StubExpr(["a"]), np.zeros(1), np.zeros(1))
    assert (lb[0], ub[0]) == (0.0, 0.0)
    assert handler.crossing_repairs == 1


def test_obbt_still_rejects_large_crossing():
    values = {("minimize", "a"): 0.2 + 1e-3, ("maximize", "a"): 0.2}
    with pytest.raises(ValueError, match="cross"):
        _stub_handler(values).obbt(_StubExpr(["a"]), np.full(1, -5.0), np.full(1, 5.0))


@pytest.mark.parametrize(
    "P, expected",
    [
        (None, True),
        (np.eye(3), True),
        (spa.eye(3, format="csc"), True),
        (2 * np.eye(3), False),
        (spa.csc_matrix(np.diag([1.0, 1.0, 1.0 + 1e-6])), False),
        (np.eye(3, 4), False),
        (spa.eye(3, 4, format="csr"), False),
        (np.ones(3), False),
    ],
)
def test_is_identity_matrix(P, expected):
    assert is_identity_matrix(P) is expected


def test_affine_output_obbt_only_for_non_identity():
    v, _ = build_relu_linsolve_chain(None, "gurobi", K=1, identity=True)
    assert not any(s.obbt_output for s in _affine_steps(v))
    v, _ = build_relu_linsolve_chain(None, "gurobi", K=1, identity=False)
    assert all(s.obbt_output for s in _affine_steps(v))


@pytest.mark.parametrize("identity", [True, False])
def test_obbt_tightens_affine_output_leaves(identity):
    """Affine output leaves use the tightened input box (identity ``M``) or output OBBT."""
    v_no, _ = build_relu_linsolve_chain(None, "gurobi", K=3, identity=identity)
    v_ob, _ = build_relu_linsolve_chain("gurobi", "gurobi", K=3, identity=identity)

    tightened = False
    for s_no, s_ob in zip(_affine_steps(v_no), _affine_steps(v_ob)):
        out = s_ob.output
        assert np.all(out.lb <= out.ub)
        w_no = s_no.output.ub - s_no.output.lb
        w_ob = out.ub - out.lb
        assert np.all(w_ob <= w_no + TEST_EPS)  # never widens
        tightened |= bool(np.any(w_ob < w_no - 1e-3))
        if identity:
            # output == arg, so the OBBT-tightened input box bounds the output.
            assert np.all(out.lb >= s_ob.lb - TEST_EPS)
            assert np.all(out.ub <= s_ob.ub + TEST_EPS)
    assert tightened, "OBBT did not tighten any Affine output leaf"


def test_obbt_affine_output_preserves_mip_optimum():
    v_no, it_no = build_relu_linsolve_chain(None, "gurobi", K=3)
    v_ob, it_ob = build_relu_linsolve_chain("gurobi", "gurobi", K=3)
    opt_no = _mip_optimum(v_no, it_no)
    opt_ob = _mip_optimum(v_ob, it_ob)
    assert opt_ob == pytest.approx(opt_no, abs=1e-4)


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_obbt_tightens_sdp_with_relu_cuts(sdp_solver):
    """With ReLU cuts on, OBBT strictly lowers the SDP bound and stays above the MIP."""
    K = 3
    cuts = {"triangle_cuts": True, "relu_envelope_cuts": True}
    opts = []
    # The no-OBBT baseline (wide IBP bounds) is too poorly scaled for the cvxpy
    # Clarabel wrapper, so it always runs on native Clarabel.
    for obbt, solver in [(None, "clarabel"), ("highs", sdp_solver)]:
        v, it = build_relu_linsolve_chain(obbt, solver, method="sdp", K=K, **cuts)
        v.set_objective_norm(it[-1] - it[-2], 2)
        opts.append(v.solve())
    opt_no, opt_ob = opts

    v_mip, it_mip = build_relu_linsolve_chain("gurobi", "gurobi", K=K)
    v_mip.set_objective_norm(it_mip[-1] - it_mip[-2], 2)
    opt_mip = v_mip.solve()

    assert opt_ob < opt_no - 1.0
    assert opt_ob >= opt_mip - 1e-3


def test_relu_cut_bounds_use_obbt_bounds():
    """The ReLU cut builders see the OBBT-tightened ``step.lb``/``step.ub``."""
    v, _ = build_relu_linsolve_chain(
        "highs", "clarabel", method="sdp", K=3, triangle_cuts=True, relu_envelope_cuts=True
    )
    tighter = False
    for step in v.canonicalizer._relu_steps:
        lb, ub = SDP._relu_arg_bounds(step)
        prop_lb, prop_ub = step.arg.bound_propagation()
        np.testing.assert_allclose(lb, np.maximum(step.lb, prop_lb))
        np.testing.assert_allclose(ub, np.minimum(step.ub, prop_ub))
        assert np.all(lb <= ub)
        tighter |= bool(np.any(lb > prop_lb + 1e-3) or np.any(ub < prop_ub - 1e-3))
    assert tighter, "OBBT bounds were never tighter than propagation"


# --------------------------------------------------------------------------
# OBBT solver parameters
# --------------------------------------------------------------------------


def test_obbt_inherits_output_flag_only():
    v = Verifier(solver="gurobi", params={"OutputFlag": 0, "TimeLimit": 5})
    model = v.obbt_handler.solver.model
    assert model.Params.OutputFlag == 0
    # Main-solve options are not forwarded to the OBBT LPs.
    assert model.Params.TimeLimit != 5


def test_obbt_params_override():
    v = Verifier(solver="gurobi", params={"OutputFlag": 0}, obbt_params={"OutputFlag": 1})
    assert v.obbt_handler.solver.model.Params.OutputFlag == 1


def test_obbt_params_from_highs_key():
    assert Verifier._obbt_params_from({"output_flag": False}, "highs") == {"output_flag": False}
    assert Verifier._obbt_params_from({"OutputFlag": 0}, "highs") is None
    assert Verifier._obbt_params_from(None, "gurobi") is None
