"""
Tests for ``update_bounds`` on the Gurobi and HiGHS solvers.

``update_bounds`` once assigned bounds to the linear expressions returned by
``expression_to_solver`` instead of the variables, so every OBBT and theory
tightening was silently dropped from the solver models. These tests read the
bounds back from the model and check that solves see them.
"""

import numpy as np
import pytest

from fomtools.canonicalizers.obbt_handler import OBBTHandler
from fomtools.linearexpression import LinearExpression
from tests.conftest import SOLVERS, mip_params


def col_bounds(solver, leaf):
    """Return the model bounds ``(lb, ub)`` of the columns of ``leaf``."""
    var = solver.vector_var_map[leaf]
    if hasattr(var, "LB"):
        return np.asarray(var.LB, dtype=float), np.asarray(var.UB, dtype=float)
    lp = solver.model.getLp()
    indices = [v.index for v in var]
    return np.asarray(lp.col_lower_)[indices], np.asarray(lp.col_upper_)[indices]


def _handler(solver):
    return OBBTHandler(solver, mip_params(solver))


def _lp(solver):
    """maximize x0 + 2 x1 s.t. x0 + x1 <= 3, 0 <= x <= 5: optimum 6 at (0, 3)."""
    handler = _handler(solver)
    x = LinearExpression.from_int(2, lb=0.0, ub=5.0)
    handler.add_var(x)
    handler.add_constraint(LinearExpression({x: np.ones((1, 2))}) <= np.array([3.0]))
    obj = LinearExpression({x: np.array([[1.0, 2.0]])})
    return handler, x, obj


def _solve(handler, obj, sense, from_scratch=False):
    handler.solver.set_objective(obj, sense)
    if from_scratch:
        handler.solver.optimize_from_scratch()
    else:
        handler.solver.optimize()
    assert handler.solver.is_optimal()
    return handler.solver["objVal"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_update_bounds_writes_each_entry(solver):
    handler = _handler(solver)
    x = LinearExpression.from_int(4, lb=-10.0, ub=10.0)
    handler.add_var(x)
    lb = np.array([-np.inf, -1.0, 0.5, 2.0])
    ub = np.array([3.0, np.inf, 0.5, 7.0])
    handler.update_bounds(x, lb, ub)
    got_lb, got_ub = col_bounds(handler.solver, x)
    # Gurobi stores infinite bounds as +-GRB.INFINITY (1e100).
    np.testing.assert_array_equal(np.where(got_lb <= -1e100, -np.inf, got_lb), lb)
    np.testing.assert_array_equal(np.where(got_ub >= 1e100, np.inf, got_ub), ub)


@pytest.mark.parametrize("solver", SOLVERS)
def test_update_bounds_changes_only_the_target_leaf(solver):
    handler = _handler(solver)
    a = LinearExpression.from_int(2, lb=-1.0, ub=1.0)
    b = LinearExpression.from_int(3, lb=-2.0, ub=2.0)
    c = LinearExpression.from_int(1, lb=-3.0, ub=3.0)
    for leaf in (a, b, c):
        handler.add_var(leaf)
    handler.update_bounds(b, np.array([0.0, 0.1, 0.2]), np.array([1.0, 1.1, 1.2]))
    for leaf, (lb, ub) in [(a, (-1.0, 1.0)), (c, (-3.0, 3.0))]:
        got_lb, got_ub = col_bounds(handler.solver, leaf)
        np.testing.assert_array_equal(got_lb, np.full(leaf.ndim, lb))
        np.testing.assert_array_equal(got_ub, np.full(leaf.ndim, ub))
    got_lb, got_ub = col_bounds(handler.solver, b)
    np.testing.assert_array_equal(got_lb, [0.0, 0.1, 0.2])
    np.testing.assert_array_equal(got_ub, [1.0, 1.1, 1.2])


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("from_scratch", [False, True])
def test_update_bounds_reaches_the_optimum_in_both_senses(solver, from_scratch):
    handler, x, obj = _lp(solver)
    assert _solve(handler, obj, "maximize") == pytest.approx(6.0)
    # After tightening x1 <= 1 the optimum is 4 at (2, 1); this value needs the
    # tightened bound (else 6), the constraint (else 7) and the objective weights.
    handler.update_bounds(x, np.zeros(2), np.array([5.0, 1.0]))
    assert _solve(handler, obj, "maximize", from_scratch) == pytest.approx(4.0)
    np.testing.assert_allclose(handler.solver.get_solution_value(x), [2.0, 1.0], atol=1e-7)
    handler.update_bounds(x, np.array([1.0, 0.5]), np.array([5.0, 1.0]))
    assert _solve(handler, obj, "minimize", from_scratch) == pytest.approx(2.0)
    np.testing.assert_allclose(handler.solver.get_solution_value(x), [1.0, 0.5], atol=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
def test_update_bounds_sets_rather_than_intersects(solver):
    handler, x, obj = _lp(solver)
    for ub, opt in [([5.0, 1.0], 4.0), ([0.5, 0.5], 1.5), ([5.0, 5.0], 6.0), ([5.0, 2.0], 5.0)]:
        handler.update_bounds(x, np.zeros(2), np.array(ub))
        assert _solve(handler, obj, "maximize") == pytest.approx(opt)


@pytest.mark.parametrize("solver", SOLVERS)
def test_update_bounds_broadcasts_scalars(solver):
    handler = _handler(solver)
    x = LinearExpression.from_int(3, lb=-5.0, ub=5.0)
    handler.add_var(x)
    handler.update_bounds(x, -1.0, 2.0)
    got_lb, got_ub = col_bounds(handler.solver, x)
    np.testing.assert_array_equal(got_lb, np.full(3, -1.0))
    np.testing.assert_array_equal(got_ub, np.full(3, 2.0))


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("composite", ["sum", "scaled"])
def test_update_bounds_rejects_a_composite_expression(solver, composite):
    handler = _handler(solver)
    x = LinearExpression.from_int(2, lb=-5.0, ub=5.0)
    y = LinearExpression.from_int(2, lb=-5.0, ub=5.0)
    handler.add_var(x)
    handler.add_var(y)
    expr = x + y if composite == "sum" else 2.0 * x
    with pytest.raises(TypeError, match="leaf"):
        handler.update_bounds(expr, np.zeros(2), np.ones(2))
    for leaf in (x, y):
        got_lb, got_ub = col_bounds(handler.solver, leaf)
        np.testing.assert_array_equal(got_lb, np.full(2, -5.0))
        np.testing.assert_array_equal(got_ub, np.full(2, 5.0))


@pytest.mark.parametrize("solver", SOLVERS)
def test_update_bounds_registers_an_unknown_leaf_with_its_bounds(solver):
    handler = _handler(solver)
    x = LinearExpression.from_int(2, lb=-5.0, ub=5.0)
    handler.update_bounds(x, np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    got_lb, got_ub = col_bounds(handler.solver, x)
    np.testing.assert_array_equal(got_lb, [0.0, 1.0])
    np.testing.assert_array_equal(got_ub, [2.0, 3.0])


@pytest.mark.parametrize("solver", SOLVERS)
def test_copy_keeps_updated_bounds_and_is_independent(solver):
    handler, x, obj = _lp(solver)
    handler.update_bounds(x, np.zeros(2), np.array([5.0, 1.0]))
    original = handler.solver
    copy = original.copy()
    np.testing.assert_array_equal(col_bounds(copy, x)[1], [5.0, 1.0])

    copy.update_bounds(x, np.zeros(2), np.array([0.5, 0.5]))
    np.testing.assert_array_equal(col_bounds(original, x)[1], [5.0, 1.0])
    original.update_bounds(x, np.zeros(2), np.array([5.0, 2.0]))
    np.testing.assert_array_equal(col_bounds(copy, x)[1], [0.5, 0.5])

    handler.solver = copy
    assert _solve(handler, obj, "maximize") == pytest.approx(1.5)
    handler.solver = original
    assert _solve(handler, obj, "maximize") == pytest.approx(5.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_relaxation_uses_updated_bounds(solver):
    handler, x, obj = _lp(solver)
    handler.solver.set_objective(obj, "maximize")
    handler.update_bounds(x, np.zeros(2), np.array([5.0, 1.0]))
    handler.solver.relax_model()
    assert handler.solver.relaxed_is_optimal()
    relaxed = handler.solver._relaxed_model
    value = relaxed.ObjVal if solver == "gurobi" else relaxed.getInfo().objective_function_value
    assert value == pytest.approx(4.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_obbt_uses_updated_bounds(solver):
    # The path the bug broke: a tightening pushed to the OBBT LP must tighten
    # later OBBT bounds of expressions of that leaf.
    handler = _handler(solver)
    x = LinearExpression.from_int(2, lb=-5.0, ub=5.0)
    handler.add_var(x)
    handler.update_bounds(x, np.zeros(2), np.array([1.0, 2.0]))
    e = LinearExpression({x: np.ones((1, 2))})
    lb, ub = handler.obbt(e, np.full(1, -10.0), np.full(1, 10.0))
    assert lb[0] == pytest.approx(0.0, abs=1e-5) and lb[0] <= 0.0
    assert ub[0] == pytest.approx(3.0, abs=1e-5) and ub[0] >= 3.0
