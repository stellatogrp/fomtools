"""Solution-consistency regression tests for the MIP encoding.

These tests close the coverage gap that let a MIP soft-threshold bug slip through:
the suite only asserted cross-method inequalities (``mip_inf <= sqrt(sdp)``) or
extracted *parameters*, never that a returned **iterate** lies on the operator
graph. A too-loose (relaxation-only) MIP encoding passes the inequality tests
because the SDP is even looser.

The bug (now fixed): ``canonicalize_softthreshold``'s "left-or-middle" case
(``lambda >= ub >= -lambda >= lb``) failed to pin the middle (zero) branch output
to 0 — the pin's guard tested ``lb`` inside a block already requiring
``lb < -lambda`` (always false), so the MIP admitted off-graph points
(``output in [arg-lambda, 0]`` while ``|arg| < lambda``), over-stating the
worst-case objective.

The core assertion here is the one that was missing: every step's returned output
equals the true operator applied to its returned input, and the reported objective
equals the value obtained by forward-simulating the true operators from the
worst-case point. Soft-threshold is covered thoroughly across all case branches;
ReLU and satlin get smoke checks.
"""

import numpy as np
import pytest

from fomtools.steps import Relu, SaturatedLinear, SoftThreshold
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Every test here solves an exact MIP with a squared two-norm objective, which is a
# MIQP that only Gurobi can do (HiGHS raises). Skip the whole module without Gurobi.
pytest.importorskip("gurobipy", reason="exact two-norm MIP requires Gurobi")

GUROBI_SILENT = {"OutputFlag": 0, "LogToConsole": 0}


def _st(v, lam):
    """True soft-threshold (shrinkage) operator."""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def _assert_softthreshold_steps_on_graph(vp, atol=TEST_EPS):
    """Assert every soft-threshold step's returned output equals
    ``soft_threshold(returned input)``. Returns the number of scalar rows checked.
    """
    checked = 0
    for step in vp.steps:
        if isinstance(step, SoftThreshold):
            arg = np.atleast_1d(vp.get_solution_value(step.arg))
            out = np.atleast_1d(vp.get_solution_value(step.output))
            true = _st(arg, step.lambd)
            assert np.allclose(out, true, atol=atol), (
                f"off-graph soft-threshold output: arg={arg}, out={out}, "
                f"true={true}, lambd={step.lambd}"
            )
            checked += step.arg.ndim
    return checked


def _softthreshold_cases_seen(vp):
    """Classify each soft-threshold row into its encoding case (for coverage
    assertions): which of the exact/uncertain branches the bounds select."""
    seen = set()
    for step in vp.steps:
        if not isinstance(step, SoftThreshold):
            continue
        lam = step.lambd
        lb = np.atleast_1d(step.lb)
        ub = np.atleast_1d(step.ub)
        for i in range(step.arg.ndim):
            if lb[i] >= lam:
                seen.add("right")
            elif ub[i] <= -lam:
                seen.add("left")
            elif lb[i] >= -lam and ub[i] <= lam:
                seen.add("zero")
            elif lb[i] < -lam and ub[i] > lam:
                seen.add("full-mixed")
            elif -lam <= lb[i] <= lam < ub[i]:
                seen.add("right-or-middle")
            elif lam >= ub[i] >= -lam >= lb[i]:
                seen.add("left-or-middle")
    return seen


def _build_ista(seed, K, box=1.5, lam_scale=2.0, obbt_solver=None, n=2, m=3):
    """Chained ISTA MIP whose straddling iterate box drives soft-threshold rows
    into the uncertain (mixed) cases. Returns everything needed to forward-simulate
    the true trajectory from the solved worst-case point."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m,))
    t = 1.0 / float(np.max(np.linalg.eigvalsh(A.T @ A)))
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    lam = lam_scale * t

    vp = Verifier(
        method="mip", solver="gurobi", obbt_solver=obbt_solver, params=dict(GUROBI_SILENT)
    )
    z = [vp.add_iterate(n, lb=-box, ub=box)]
    bp = vp.add_param(m, lb=b - 0.5, ub=b + 0.5)
    for k in range(K):
        z.append(vp.step.soft_threshold(At @ z[k] + Bt @ bp, lambd=lam))
    vp.set_objective_norm(z[K] - z[K - 1], 2)
    return vp, z, bp, At, Bt, lam, K


def _ista_forward_objective(vp, z, bp, At, Bt, lam, K):
    """Forward-simulate the TRUE soft-threshold trajectory from the solved
    worst-case (z0, b) and return the resulting squared-two-norm objective."""
    z0 = vp.get_solution_value(z[0])
    b_star = vp.get_solution_value(bp)
    zt = [z0]
    for _ in range(K):
        zt.append(_st(At @ zt[-1] + Bt @ b_star, lam))
    return float(np.sum((zt[K] - zt[K - 1]) ** 2))


# ============================================================
# Targeted regression: the exact instance that exposed the bug
# ============================================================


def test_softthreshold_seed10_regression():
    """ISTA seed=10 produced a left-or-middle row with an off-graph output
    (out=-0.270 where |arg|<lambda => true 0), inflating the objective from the
    true 0.7325 to 0.8791. Assert the returned iterates are on-graph and the
    reported objective equals the true forward-simulated objective."""
    vp, z, bp, At, Bt, lam, K = _build_ista(seed=10, K=2)
    objval = vp.solve()

    # The regression must actually exercise the bug regime.
    assert "left-or-middle" in _softthreshold_cases_seen(vp)

    n_checked = _assert_softthreshold_steps_on_graph(vp)
    assert n_checked > 0

    true_obj = _ista_forward_objective(vp, z, bp, At, Bt, lam, K)
    assert objval == pytest.approx(true_obj, abs=TEST_EPS), (
        f"MIP objective {objval} != true forward-sim objective {true_obj} "
        f"(MIP admitted a non-physical point)"
    )


# ============================================================
# Broad soft-threshold consistency across seeds / depth / OBBT
# ============================================================


@pytest.mark.parametrize("obbt_solver", [None, "highs"])
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("seed", [10, 0, 3])
def test_softthreshold_solution_on_graph(seed, K, obbt_solver):
    """Every soft-threshold iterate is on-graph and the reported objective matches
    the true forward-simulated objective — across seeds, depths, and OBBT on/off
    (the OBBT fresh/stale-bounds path)."""
    vp, z, bp, At, Bt, lam, K = _build_ista(seed=seed, K=K, obbt_solver=obbt_solver)
    objval = vp.solve()
    n_checked = _assert_softthreshold_steps_on_graph(vp)
    assert n_checked > 0  # the sweep must actually contain soft-threshold rows
    true_obj = _ista_forward_objective(vp, z, bp, At, Bt, lam, K)
    assert objval == pytest.approx(true_obj, abs=TEST_EPS)


def test_softthreshold_all_uncertain_branches_exercised():
    """Across the parametrized seeds/depths, the uncertain branches that carry the
    binary encoding (full-mixed, left-or-middle, right-or-middle) are all reached,
    so the on-graph tests above genuinely cover them."""
    seen = set()
    for seed in (10, 0, 3, 1, 7):
        for K in (1, 2, 3):
            vp, *_ = _build_ista(seed=seed, K=K)
            vp.solve()
            seen |= _softthreshold_cases_seen(vp)
    for case in ("full-mixed", "left-or-middle", "right-or-middle"):
        assert case in seen, f"branch {case!r} never exercised; seen={seen}"


def test_softthreshold_single_step_branches_exact():
    """Single soft-threshold step with input boxes hand-placed in each case branch:
    the optimum must be on-graph for every branch (exact encoding)."""
    lam = 1.0
    # (lb, ub) -> case: exact pieces + the three uncertain branches.
    boxes = [
        (1.5, 3.0),  # right (lb >= lam)
        (-3.0, -1.5),  # left (ub <= -lam)
        (-0.5, 0.5),  # zero (-lam <= lb <= ub <= lam)
        (-2.0, 2.0),  # full-mixed
        (-0.5, 2.0),  # right-or-middle
        (-2.0, 0.5),  # left-or-middle (the buggy case)
    ]
    for lb, ub in boxes:
        for sense in (1.0, -1.0):  # try to push the optimum either way
            vp = Verifier(
                method="mip",
                solver="gurobi",
                obbt_solver=None,
                params=dict(GUROBI_SILENT),
            )
            x = vp.add_iterate(1, lb=lb, ub=ub)
            y = vp.step.soft_threshold(x, lambd=lam)
            # Linear objective (max +/- y) to probe both directions of the graph.
            vp.set_objective_norm(sense * y, 1)
            vp.solve()
            arg = np.atleast_1d(vp.get_solution_value(x))
            out = np.atleast_1d(vp.get_solution_value(y))
            assert np.allclose(out, _st(arg, lam), atol=TEST_EPS), (
                f"box=({lb},{ub}) sense={sense}: arg={arg} out={out} true={_st(arg, lam)}"
            )


# ============================================================
# Smoke checks: ReLU and satlin stay on-graph
# ============================================================


def test_relu_solution_on_graph_smoke():
    """NNLS-style chained ReLU (mixed regime): returned outputs equal relu(input)."""
    rng = np.random.default_rng(0)
    n, m = 2, 3
    A = rng.normal(size=(m, n))
    t = 1.0 / float(np.max(np.linalg.eigvalsh(A.T @ A)))
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None, params=dict(GUROBI_SILENT))
    x = vp.add_param(m, lb=0.0, ub=2.0)
    z = [vp.add_iterate(n, lb=-2.0, ub=2.0)]
    for k in range(2):
        z.append(vp.step.relu(z[k] - t * A.T @ (A @ z[k] - x)))
    vp.set_objective_norm(z[2] - z[1], 2)
    vp.solve()
    for step in vp.steps:
        if isinstance(step, Relu):
            arg = np.atleast_1d(vp.get_solution_value(step.arg))
            out = np.atleast_1d(vp.get_solution_value(step.output))
            assert np.allclose(out, np.maximum(0.0, arg), atol=TEST_EPS)


def test_satlin_solution_on_graph_smoke():
    """Box-PGD chained satlin: returned outputs equal clip(input, l, u)."""
    rng = np.random.default_rng(0)
    n = 2
    P = np.eye(n)
    q = rng.normal(size=n)
    lo = np.array([-1.0, -0.5])
    hi = np.array([0.5, 1.0])
    t = 0.5
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None, params=dict(GUROBI_SILENT))
    q_param = vp.add_param(n, lb=q - 0.5, ub=q + 0.5)
    z = [vp.add_iterate(n, lb=-1.0, ub=1.0)]
    for k in range(2):
        z.append(vp.step.satlin(z[k] - t * (P @ z[k] + q_param), l=lo, u=hi))
    vp.set_objective_norm(z[2] - z[1], 2)
    vp.solve()
    for step in vp.steps:
        if isinstance(step, SaturatedLinear):
            arg = np.atleast_1d(vp.get_solution_value(step.arg))
            out = np.atleast_1d(vp.get_solution_value(step.output))
            assert np.allclose(out, np.clip(arg, lo, hi), atol=TEST_EPS)


# ---- satlin mixed-regime consistency (regression for the lower-or-middle bug) ---- #
#
# The native canonicalize_saturatedlinear had the same bug class as soft-threshold:
# its "lower-or-middle" case (lb < l <= ub <= u) failed to pin the lower-saturation
# branch to l, admitting off-graph outputs (e.g. output=-0.577 where arg=-1.236 < l=-1
# should give exactly l). These tests drive satlin into all uncertain sub-cases via an
# iterate box straddling the saturation faces, and assert the returned iterates are
# on-graph and the reported objective matches the true forward-simulated value.


def _build_boxpgd(seed, K, obbt_solver=None):
    """Chained box-PGD whose wide iterate box drives satlin rows into the
    uncertain (mixed) cases. Returns the data needed to forward-simulate."""
    rng = np.random.default_rng(seed)
    n = 2
    P = np.eye(n) + 0.1 * rng.normal(size=(n, n))
    P = (P + P.T) / 2
    q = rng.normal(size=n)
    lo = np.array([-1.0, -0.5])
    hi = np.array([0.5, 1.0])
    t = 0.5

    vp = Verifier(
        method="mip", solver="gurobi", obbt_solver=obbt_solver, params=dict(GUROBI_SILENT)
    )
    q_param = vp.add_param(n, lb=q - 0.5, ub=q + 0.5)
    z = [vp.add_iterate(n, lb=-1.5, ub=1.5)]  # straddles [lo, hi] faces
    for k in range(K):
        z.append(vp.step.satlin(z[k] - t * (P @ z[k] + q_param), l=lo, u=hi))
    vp.set_objective_norm(z[K] - z[K - 1], 2)
    return vp, z, q_param, P, q, lo, hi, t, K


def _boxpgd_forward_objective(vp, z, q_param, P, q, lo, hi, t, K):
    q_star = vp.get_solution_value(q_param)
    zt = [vp.get_solution_value(z[0])]
    for _ in range(K):
        zt.append(np.clip(zt[-1] - t * (P @ zt[-1] + q_star), lo, hi))
    return float(np.sum((zt[K] - zt[K - 1]) ** 2))


@pytest.mark.parametrize("obbt_solver", [None, "highs"])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("seed", [22, 13, 0, 7])
def test_satlin_solution_on_graph_mixed(seed, K, obbt_solver):
    """Every satlin iterate is on-graph (output == clip(arg, l, u)) and the reported
    objective equals the true forward-simulated objective — across seeds, depths, and
    OBBT on/off, in the box-straddling regime that exposed the lower-or-middle bug."""
    vp, z, q_param, P, q, lo, hi, t, K = _build_boxpgd(seed, K, obbt_solver)
    objval = vp.solve()
    for step in vp.steps:
        if isinstance(step, SaturatedLinear):
            arg = np.atleast_1d(vp.get_solution_value(step.arg))
            out = np.atleast_1d(vp.get_solution_value(step.output))
            assert np.allclose(out, np.clip(arg, lo, hi), atol=TEST_EPS), (
                f"off-graph satlin: arg={arg}, out={out}, clip={np.clip(arg, lo, hi)}"
            )
    true_obj = _boxpgd_forward_objective(vp, z, q_param, P, q, lo, hi, t, K)
    assert objval == pytest.approx(true_obj, abs=TEST_EPS)


def test_satlin_lower_or_middle_branch_exercised():
    """Confirm the regression actually drives a satlin row into the lower-or-middle
    case (lb < l <= ub <= u) — the branch that carried the bug — so the on-graph
    test above is meaningful."""
    vp, *_ = _build_boxpgd(22, 3)
    vp.solve()
    lo = np.array([-1.0, -0.5])
    hi = np.array([0.5, 1.0])
    seen_lower_or_middle = False
    for step in vp.steps:
        if isinstance(step, SaturatedLinear):
            lb = np.atleast_1d(step.lb)
            ub = np.atleast_1d(step.ub)
            for j in range(step.arg.ndim):
                if lb[j] < lo[j] and lo[j] <= ub[j] <= hi[j]:
                    seen_lower_or_middle = True
    assert seen_lower_or_middle, "regression did not exercise the lower-or-middle case"
