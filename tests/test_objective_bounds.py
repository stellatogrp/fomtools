"""
Tests for tightening the infinity-norm objective's bounds with OBBT.

The objective's big-M constants come from bounds on the residual. Replacing the
interval-propagation bounds with OBBT bounds must keep every feasible residual
inside them, leave the MIP optimum unchanged, and only tighten the relaxation.
"""

import warnings

import gurobipy as gp
import numpy as np
import pytest

from tests.test_convex_hull_cuts import _pgd_chain
from tests.test_obbt_handler import build_relu_chain, build_satlin_chain


def _objective_bounds(vp):
    _, _, lb_list, ub_list = vp.canonicalizer._objective_spec
    return np.hstack(lb_list), np.hstack(ub_list)


@pytest.mark.parametrize("kind", ["relu", "satlin"])
def test_tightened_bounds_contain_residuals(kind):
    K = 4
    vp, q, z, simulate = _pgd_chain(kind, K, const=0.1, obbt="gurobi")
    Z = simulate(np.random.default_rng(0).uniform(-1, 1, size=(20000, len(q.lb))))
    for k in range(1, K + 1):
        vp.set_objective_norm(z[k] - z[k - 1], np.inf)
        lb0, ub0 = _objective_bounds(vp)
        vp.solve(tighten_objective_bounds=True)
        lb, ub = _objective_bounds(vp)
        r = Z[k] - Z[k - 1]
        assert np.all(r >= lb[None, :]) and np.all(r <= ub[None, :])
        # Never looser than interval propagation, and tighter somewhere past K=1.
        assert np.all(lb >= lb0) and np.all(ub <= ub0)
        if k > 1:
            assert vp.objective_bound_stats["width_ratio"] > 1.0


@pytest.mark.parametrize("kind", ["relu", "satlin"])
@pytest.mark.parametrize("cuts", [False, True])
def test_optimum_unchanged(kind, cuts):
    K = 4
    vp_ref, _, z_ref, _ = _pgd_chain(kind, K, const=0.1, obbt="gurobi")
    vp_new, _, z_new, _ = _pgd_chain(kind, K, const=0.1, obbt="gurobi")
    for k in range(1, K + 1):
        vp_ref.set_objective_norm(z_ref[k] - z_ref[k - 1], np.inf)
        vp_new.set_objective_norm(z_new[k] - z_new[k - 1], np.inf)
        expected = vp_ref.solve(tighten_objective_bounds=False)
        got = vp_new.solve(tighten_objective_bounds=True, convex_hull_cuts=cuts)
        assert got == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("solver", ["gurobi", "highs"])
@pytest.mark.parametrize("build", [build_relu_chain, build_satlin_chain])
def test_optimum_unchanged_chain_builders(solver, build):
    v_ref, it_ref = build(obbt=solver, solver=solver)
    v_new, it_new = build(obbt=solver, solver=solver)
    v_ref.set_objective_norm(it_ref[-1] - it_ref[-2], np.inf)
    v_new.set_objective_norm(it_new[-1] - it_new[-2], np.inf)
    assert v_new.solve(tighten_objective_bounds=True) == pytest.approx(
        v_ref.solve(tighten_objective_bounds=False), abs=1e-5
    )


def _root_lp(vp):
    """LP relaxation value of the model the next solve would build."""
    solver = vp.canonicalizer.solver.copy()
    vp.canonicalizer._apply_objective(solver)
    relaxed = solver.model.relax()
    relaxed.Params.OutputFlag = 0
    relaxed.optimize()
    assert relaxed.Status == gp.GRB.OPTIMAL
    return relaxed.ObjVal


def test_relaxation_tighter():
    K = 4
    vp, _, z, _ = _pgd_chain("satlin", K, const=0.0, obbt="gurobi")
    for k in range(1, K + 1):
        vp.set_objective_norm(z[k] - z[k - 1], np.inf)
        before = _root_lp(vp)
        stats = vp.canonicalizer.tighten_objective_bounds()
        assert _root_lp(vp) <= before + 1e-9
        assert stats["M_after"] <= stats["M_before"]


def test_padding_and_clipping():
    vp, _, z, _ = _pgd_chain("relu", 2, const=0.1, obbt="gurobi")
    vp.set_objective_norm(z[2] - z[1], np.inf)
    _, expr_list, lb0, ub0 = vp.canonicalizer._objective_spec
    raw_lb, raw_ub = vp.obbt_handler.obbt(expr_list[0], lb0[0], ub0[0])
    vp.canonicalizer.tighten_objective_bounds(pad=1e-4)
    lb, ub = _objective_bounds(vp)
    tight = (raw_lb > lb0[0]) & (raw_ub < ub0[0])
    assert np.all(lb[tight] < raw_lb[tight]) and np.all(ub[tight] > raw_ub[tight])
    assert np.all(lb >= lb0[0]) and np.all(ub <= ub0[0])


def test_noop_without_obbt():
    vp_ref, _, z_ref, _ = _pgd_chain("satlin", 2, const=0.0, obbt=None)
    vp_new, _, z_new, _ = _pgd_chain("satlin", 2, const=0.0, obbt=None)
    vp_ref.set_objective_norm(z_ref[2] - z_ref[1], np.inf)
    vp_new.set_objective_norm(z_new[2] - z_new[1], np.inf)
    spec_before = _objective_bounds(vp_new)
    with pytest.warns(UserWarning, match="tighten_objective_bounds"):
        got = vp_new.solve(tighten_objective_bounds=True)
    assert got == pytest.approx(vp_ref.solve(), abs=1e-6)
    assert vp_new.objective_bound_stats is None
    for a, b in zip(spec_before, _objective_bounds(vp_new)):
        assert np.array_equal(a, b)


def test_default_tightens_with_obbt_and_is_silent_without():
    vp, _, z, _ = _pgd_chain("satlin", 2, const=0.0, obbt="gurobi")
    vp.set_objective_norm(z[2] - z[1], np.inf)
    vp.solve()
    assert vp.objective_bound_stats is not None

    vp, _, z, _ = _pgd_chain("satlin", 2, const=0.0, obbt=None)
    vp.set_objective_norm(z[2] - z[1], np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vp.solve()
    assert vp.objective_bound_stats is None


def test_noop_for_two_norm():
    vp, _, z, _ = _pgd_chain("relu", 2, const=0.0, obbt="gurobi")
    vp.set_objective_norm(z[2] - z[1], 2)
    spec = vp.canonicalizer._objective_spec
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vp.solve(tighten_objective_bounds=True)
    assert vp.objective_bound_stats is None
    assert vp.canonicalizer._objective_spec is spec


# --------------------------------------------------------------------------
# Solver defaults: branching priority on the objective binaries and MIPFocus.
# --------------------------------------------------------------------------


def _priorities(vp):
    """BranchPriority of the binaries of the solved model, split by origin."""
    base = {v.VarName for v in vp.canonicalizer.solver.model.getVars()}
    model = vp.canonicalizer._solved_copy.model
    obj, step = [], []
    for v in model.getVars():
        if v.VType == gp.GRB.BINARY:
            (step if v.VarName in base else obj).append(v.BranchPriority)
    return obj, step


def test_objective_binaries_get_branch_priority():
    vp, _, z, _ = _pgd_chain("satlin", 3, const=0.0, obbt="gurobi")
    vp.set_objective_norm(z[3] - z[2], np.inf)
    vp.solve()
    obj, step = _priorities(vp)
    assert obj and all(p == vp.canonicalizer.objective_branch_priority for p in obj)
    assert step and all(p == 0 for p in step)


def test_branch_priority_can_be_disabled(monkeypatch):
    from fomtools.canonicalizers.mip import MIP

    monkeypatch.setattr(MIP, "objective_branch_priority", 0)
    vp, _, z, _ = _pgd_chain("satlin", 3, const=0.0, obbt="gurobi")
    vp.set_objective_norm(z[3] - z[2], np.inf)
    vp.solve()
    obj, step = _priorities(vp)
    assert all(p == 0 for p in obj + step)


def test_mipfocus_default_and_override():
    from fomtools.verifier import Verifier

    assert Verifier(solver="gurobi", params={"OutputFlag": 0}).canonicalizer.solver.model.Params.MIPFocus == 2
    assert Verifier(solver="gurobi", params={"OutputFlag": 0, "MIPFocus": 0}).canonicalizer.solver.model.Params.MIPFocus == 0
    # The default carries into the solve copy.
    vp, _, z, _ = _pgd_chain("relu", 2, const=0.0, obbt="gurobi")
    vp.set_objective_norm(z[2] - z[1], np.inf)
    vp.solve()
    assert vp.canonicalizer._solved_copy.model.Params.MIPFocus == 2


def test_highs_ignores_branch_priority():
    v_ref, it_ref = build_relu_chain(obbt="highs", solver="highs")
    v_ref.set_objective_norm(it_ref[-1] - it_ref[-2], np.inf)
    v_gur, it_gur = build_relu_chain(obbt="gurobi", solver="gurobi")
    v_gur.set_objective_norm(it_gur[-1] - it_gur[-2], np.inf)
    assert v_ref.solve() == pytest.approx(v_gur.solve(), abs=1e-5)
