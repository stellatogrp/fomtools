"""
Tests for the MIP warm starts: the previous worst case run one more step, and the
worst of the user's samples.

The decisive check is that every start is a feasible point of the solve model:
fixing every variable to its start value must leave a feasible MIP whose value is
the start's residual. That verifies the forward evaluation and every binary rule.
"""

import gurobipy as gp
import numpy as np
import pytest

import fomtools as ft
from fomtools.verifier import Verifier
from tests.test_convex_hull_cuts import MIP_EXACT, _pgd_chain
from tests.test_obbt_handler import build_relu_chain


def _fixed_start_value(vp, start_index):
    """Fix the next solve's model to one of its starts; return the objective, or None if infeasible."""
    canon = vp.canonicalizer
    solver = canon.solver.copy()
    canon._apply_objective(solver)
    canon._solved_copy = solver
    starts = canon._warm_starts()
    if start_index >= len(starts):
        return None, starts
    for leaf, value in starts[start_index].items():
        if leaf in solver.vector_var_map:
            var = solver.vector_var_map[leaf]
            value = np.asarray(value, dtype=float).reshape(-1)
            var.LB = value
            var.UB = value
    solver.model.Params.OutputFlag = 0
    solver.model.optimize()
    if solver.model.Status != gp.GRB.OPTIMAL:
        return None, starts
    return solver.model.ObjVal, starts


@pytest.mark.parametrize("kind", ["relu", "satlin", "soft"])
@pytest.mark.parametrize("const", [0.0, 0.1])
def test_starts_are_feasible_points(kind, const):
    K = 4
    vp, q, z, simulate = _pgd_chain(kind, K, const=const, obbt="gurobi")
    vp.set_start_samples({q: np.random.default_rng(1).uniform(-1, 1, size=(500, len(q.lb)))})
    for k in range(1, K + 1):
        vp.maximize(ft.norm(z[k] - z[k - 1], np.inf))
        n_starts = 1 if k == 1 else 2
        for s in range(n_starts):
            value, starts = _fixed_start_value(vp, s)
            assert len(starts) == n_starts
            stat = vp.canonicalizer.warm_start_stats[s]
            assert value is not None, f"start {stat['kind']} infeasible at K={k}"
            assert value == pytest.approx(stat["objective"], abs=1e-6)
            assert stat["seeded"] == stat["leaves"]
        vp.solve()


def test_partial_relu_and_affine_starts_are_feasible():
    n = 3
    rng = np.random.default_rng(4)
    M = rng.normal(size=(n, n))
    P = M @ M.T / n + 0.2 * np.eye(n)
    vp = Verifier(method="mip", solver="gurobi", obbt_solver="gurobi", params=MIP_EXACT)
    x = vp.parameter(n, lb=-np.ones(n), ub=np.ones(n))
    z = [vp.iterate(n, lb=np.zeros(n), ub=np.zeros(n))]
    for k in range(1, 4):
        y = vp.implicit_linear(z[-1] + 0.3 * x, output_mat=np.eye(n) + 0.5 * P)
        z.append(vp.relu(y - 0.05, proj_ranges=(0, 2)))
        vp.maximize(ft.norm(z[k] - z[k - 1], np.inf))
        if k > 1:
            value, _ = _fixed_start_value(vp, 0)
            assert value is not None
            assert value == pytest.approx(
                vp.canonicalizer.warm_start_stats[0]["objective"], abs=1e-6
            )
        vp.solve()


def test_sample_start_matches_numpy_sample_max():
    K = 3
    vp, q, z, simulate = _pgd_chain("satlin", K, const=0.1, obbt="gurobi")
    Q = np.random.default_rng(2).uniform(-1, 1, size=(2000, len(q.lb)))
    vp.set_start_samples({q: Q})
    Z = simulate(Q)
    for k in range(1, K + 1):
        vp.maximize(ft.norm(z[k] - z[k - 1], np.inf))
        vp.solve()
        sample = [s for s in vp.canonicalizer.warm_start_stats if s["kind"] == "sample_max"]
        assert sample[0]["objective"] == pytest.approx(np.abs(Z[k] - Z[k - 1]).max(), abs=1e-9)


@pytest.mark.parametrize("kind", ["relu", "satlin", "soft"])
def test_incumbent_at_least_start_and_optimum_unchanged(kind):
    K = 4
    vp_ref, _, z_ref, _ = _pgd_chain(kind, K, const=0.1, obbt="gurobi")
    vp, q, z, _ = _pgd_chain(kind, K, const=0.1, obbt="gurobi")
    vp.set_start_samples({q: np.random.default_rng(3).uniform(-1, 1, size=(500, len(q.lb)))})
    for k in range(1, K + 1):
        vp_ref.maximize(ft.norm(z_ref[k] - z_ref[k - 1], np.inf))
        vp.maximize(ft.norm(z[k] - z[k - 1], np.inf))
        expected = vp_ref.solve(warm_start=False)
        got = vp.solve()
        # Equal to Gurobi's feasibility tolerance (1e-6), which a start can sit on.
        assert got == pytest.approx(expected, abs=1e-5)
        best_start = max(s["objective"] for s in vp.canonicalizer.warm_start_stats)
        assert got >= best_start - 1e-6


def test_without_samples_only_sequential():
    vp, _, z, _ = _pgd_chain("relu", 2, const=0.0, obbt="gurobi")
    vp.maximize(ft.norm(z[1] - z[0], np.inf))
    vp.solve()
    assert vp.canonicalizer.warm_start_stats == []
    vp.maximize(ft.norm(z[2] - z[1], np.inf))
    vp.solve()
    assert [s["kind"] for s in vp.canonicalizer.warm_start_stats] == ["sequential"]


def test_warm_start_disabled():
    vp, _, z, _ = _pgd_chain("relu", 2, const=0.0, obbt="gurobi")
    for k in (1, 2):
        vp.maximize(ft.norm(z[k] - z[k - 1], np.inf))
        vp.solve(warm_start=False)
        assert vp.canonicalizer.warm_start_stats is None


def test_highs_ignores_starts():
    v_h, it_h = build_relu_chain(obbt="highs", solver="highs", K=3)
    v_g, it_g = build_relu_chain(obbt="gurobi", solver="gurobi", K=3)
    for k in (2, 3):
        v_h.maximize(ft.norm(it_h[k] - it_h[k - 1], np.inf))
        v_g.maximize(ft.norm(it_g[k] - it_g[k - 1], np.inf))
        assert v_h.solve() == pytest.approx(v_g.solve(), abs=1e-5)


def test_lp_step_gets_partial_start():
    """Steps without a forward evaluation leave their leaves to the solver."""
    C = np.vstack([np.eye(2), -np.eye(2)])
    v = Verifier(method="mip", solver="gurobi", obbt_solver="gurobi", params={"OutputFlag": 0})
    q = v.parameter(2, lb=np.array([0.5, -1.0]), ub=np.array([1.0, -0.5]))
    for _ in range(2):
        u = v.linear_program(
            q=q, C=C, d=np.ones(4), dual_ub=10.0, primal_bounds=(-np.ones(2), np.ones(2))
        )
        v.maximize(ft.norm([u], np.inf))
        v.solve()
    stats = v.canonicalizer.warm_start_stats
    assert stats and stats[0]["kind"] == "sequential"
    assert stats[0]["seeded"] < stats[0]["leaves"]
