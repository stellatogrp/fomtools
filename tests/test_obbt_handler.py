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

from fomtools.verifier import Verifier
from tests.conftest import (
    SDP_SOLVERS,
    SOLVERS,
    TEST_EPS,
    build_lasso_ista,
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
    params = {"OutputFlag": 0} if method == "mip" else None
    v = Verifier(method=method, solver=solver, obbt_solver=obbt, params=params)
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    b = v.add_param(m, lb=-1.0 * np.ones(m), ub=1.0 * np.ones(m))
    iterates = [z]
    for _ in range(K):
        z = v.step.relu(rhs=W @ z + G @ b)
        iterates.append(z)
    return v, iterates


def build_satlin_chain(obbt, solver, *, method="mip", K=3, seed=2, n=3, m=3, t=0.1):
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    W = np.eye(n) - t * A.T @ A
    G = t * A.T
    l = -0.3 * np.ones(n)
    u = 0.3 * np.ones(n)
    params = {"OutputFlag": 0} if method == "mip" else None
    v = Verifier(method=method, solver=solver, obbt_solver=obbt, params=params)
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    b = v.add_param(m, lb=-1.0 * np.ones(m), ub=1.0 * np.ones(m))
    iterates = [z]
    for _ in range(K):
        z = v.step.satlin(rhs=W @ z + G @ b, l=l, u=u)
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


# NOTE: the convex_hull_cuts (Huchette) subsystem only supports ReLU and
# soft-threshold; it raises AttributeError on SaturatedLinear independently of
# OBBT, so satlin is excluded here. Huchette cuts are out of scope for OBBT.
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("op", ["softthreshold", "relu"])
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

    # Soft-threshold decomposes into ReLU sub-steps under SDP; their lhs are the
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
# 7. Affine steps skip OBBT input tightening but still feed the OBBT LP.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("solver", SOLVERS)
def test_skip_obbt_affine_still_holds(solver):
    from fomtools.steps import Affine

    np.random.seed(3)
    n = 3
    W = np.random.normal(size=(n, n))
    v = Verifier(solver=solver, obbt_solver=solver, params={"OutputFlag": 0})
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    y = v.run_step(z, Affine, weight=W)

    affine_step = v.steps[-1]
    assert affine_step.skip_obbt is True

    # Affine bounds come from analytic interval propagation of the input.
    rhs_lb, rhs_ub = affine_step.rhs.bound_propagation()
    assert np.allclose(affine_step.lb, rhs_lb)
    assert np.allclose(affine_step.ub, rhs_ub)

    # The affine equality is in the OBBT LP: a downstream ReLU's input can be
    # tightened against it (otherwise OBBT would be a no-op).
    v.step.relu(rhs=y)
    relu_step = v.steps[-1]
    relu_lb, relu_ub = relu_step.rhs.bound_propagation()
    assert np.all(relu_step.lb >= relu_lb - TEST_EPS)
    assert np.all(relu_step.ub <= relu_ub + TEST_EPS)
