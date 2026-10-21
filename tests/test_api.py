"""Tests for the CVXPY-style user API of ``Verifier``.

Covers ``iterate`` / ``parameter`` and their deprecated ``add_*`` aliases, the
top-level step methods, ``ft.norm`` with ``maximize``, ``LinearExpression.value``,
and the ``verbose`` flag, which silences every solver by default.
"""

import numpy as np
import pytest

import fomtools as ft
from fomtools.linearexpression import LinearExpression
from fomtools.solvers.highs import HiGHSSolver
from tests.conftest import TEST_EPS

try:
    import gurobipy  # noqa: F401

    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not HAS_GUROBI, reason="gurobipy is not installed")


def _pgd(vp, K=3, n=3, seed=0):
    """Projected gradient descent on a box-parameter QP; returns ``(q, z, z_prev)``."""
    A = np.random.default_rng(seed).normal(size=(n, n))
    P = A @ A.T
    q = vp.parameter(n, lb=-1, ub=1)
    z = vp.iterate(n, lb=0, ub=0)
    for _ in range(K):
        z_prev, z = z, vp.relu(z - 0.05 * (P @ z + q))
    return q, z, z_prev


# --- iterate / parameter --------------------------------------------------------------


def test_iterate_and_parameter_are_registered_leaves():
    vp = ft.Verifier()
    x = vp.parameter(2, lb=-1, ub=np.array([1.0, 2.0]))
    z = vp.iterate(3, lb=0, ub=0)
    assert x.is_leaf and z.is_leaf
    assert x.ndim == 2 and z.ndim == 3
    np.testing.assert_array_equal(x.lb, [-1, -1])
    np.testing.assert_array_equal(x.ub, [1, 2])
    assert any(z is it for it in vp.iterates)
    assert not any(x is it for it in vp.iterates)


def test_parameter_samples_for_the_simulator():
    vp = ft.Verifier(method="simulator")
    with pytest.raises(ValueError, match=r"parameter\(ndim=2\)"):
        vp.parameter(2, samples=np.zeros((5, 3)))
    vp.parameter(2, samples=np.zeros((5, 2)))
    assert not vp._unsampled_params


@pytest.mark.parametrize(
    "old, new, args",
    [("add_iterate", "iterate", (2, 0, 1)), ("add_param", "parameter", (2, -1, 1))],
)
def test_add_aliases_are_deprecated(old, new, args):
    vp = ft.Verifier()
    with pytest.warns(DeprecationWarning, match=f"use Verifier.{new}"):
        var = getattr(vp, old)(*args)
    assert var.is_leaf and var.ndim == 2
    np.testing.assert_array_equal(var.lb, [args[1]] * 2)
    np.testing.assert_array_equal(var.ub, [args[2]] * 2)


# --- step methods ---------------------------------------------------------------------


def _step_value(build):
    vp = ft.Verifier()
    x = vp.parameter(3, lb=-2, ub=2)
    out = build(vp, x)
    vp.maximize(ft.norm(out, np.inf))
    return vp.solve(), out


STEPS = {
    "relu": ({}, lambda v: np.maximum(v, 0)),
    "satlin": ({"l": np.full(3, -0.5), "u": np.full(3, 0.5)}, lambda v: np.clip(v, -0.5, 0.5)),
    "soft_threshold": ({"lambd": 0.3}, lambda v: np.sign(v) * np.maximum(np.abs(v) - 0.3, 0)),
}


@pytest.mark.parametrize("name", STEPS)
def test_step_methods_match_the_step_proxy(name):
    kwargs, ref = STEPS[name]
    top, out = _step_value(lambda vp, x: getattr(vp, name)(x - 0.5, **kwargs))
    proxy, _ = _step_value(lambda vp, x: getattr(vp.step, name)(x - 0.5, **kwargs))
    assert abs(top - proxy) <= TEST_EPS
    # The worst case over x in [-2, 2]^3 is attained at a vertex.
    assert abs(top - np.max(np.abs(ref(np.array([-2.5, 1.5]))))) <= TEST_EPS


def test_implicit_linear_matches_the_step_proxy():
    M = np.array([[2.0, 0.0], [1.0, 1.0]])

    def build(vp, x, proxy):
        step = vp.step.implicit_linear if proxy else vp.implicit_linear
        return step(x[:2], output_mat=M)

    top, _ = _step_value(lambda vp, x: build(vp, x, False))
    proxy, _ = _step_value(lambda vp, x: build(vp, x, True))
    assert abs(top - proxy) <= TEST_EPS
    # w = M^{-1} x, so w_1 = x_1 / 2 and w_2 = x_2 - x_1 / 2; the worst case is 3.
    assert abs(top - 3.0) <= TEST_EPS


def test_relu_forwards_relax_binary_vars():
    vp = ft.Verifier()
    x = vp.parameter(2, lb=-1, ub=1)
    vp.relu(x, relax_binary_vars=True)
    assert vp.steps[-1].relax_binary_vars


# --- norm / maximize ------------------------------------------------------------------


@pytest.mark.parametrize("p, expected", [(1, 1), (np.inf, np.inf), ("inf", np.inf)])
def test_norm_atom(p, expected):
    x = ft.Verifier().parameter(2)
    atom = ft.norm(x, p)
    assert atom.p == expected and atom.exprs == (x,)
    assert ft.norm([x, x], p).exprs == (x, x)


def test_norm_rejects_two_norm_and_unknown_orders():
    x = ft.Verifier().parameter(2)
    with pytest.raises(ValueError, match="sum_squares"):
        ft.norm(x, 2)
    with pytest.raises(ValueError, match="Unsupported norm order"):
        ft.norm(x, 3)


@pytest.mark.parametrize("p", [1, np.inf])
def test_maximize_matches_set_objective_norm(p):
    vp_new, vp_old = ft.Verifier(), ft.Verifier()
    _, z, z_prev = _pgd(vp_new)
    vp_new.maximize(ft.norm(z - z_prev, p))
    _, z, z_prev = _pgd(vp_old)
    vp_old.set_objective_norm(z - z_prev, p)
    assert abs(vp_new.solve() - vp_old.solve()) <= TEST_EPS
    assert vp_new.objective[1] == p and len(vp_new.objective[0]) == 1


def test_maximize_norm_of_a_list_stacks_the_expressions():
    vp = ft.Verifier()
    x = vp.parameter(2, lb=-1, ub=1)
    y = vp.parameter(1, lb=-3, ub=2)
    vp.maximize(ft.norm([x, y], np.inf))
    assert abs(vp.solve() - 3.0) <= TEST_EPS
    vp.maximize(ft.norm([x, y], 1))
    assert abs(vp.solve() - 5.0) <= TEST_EPS


def test_maximize_sum_squares_matches_two_norm_objective():
    rng = np.random.default_rng(0)
    samples = rng.uniform(-1, 1, size=(50, 3))
    results = []
    for use_maximize in (True, False):
        vp = ft.Verifier(method="simulator")
        x = vp.parameter(3, samples=samples)
        z = vp.relu(x - 0.2)
        if use_maximize:
            vp.maximize(ft.sum_squares(z))
        else:
            vp.set_two_norm_objective(z)
        results.append(np.asarray(vp.solve()))
        assert vp.objective[1] == 2
    np.testing.assert_allclose(results[0], results[1])


def test_maximize_rejects_other_objectives():
    vp = ft.Verifier()
    x = vp.parameter(2)
    with pytest.raises(TypeError, match="ft.norm"):
        vp.maximize(x)


# --- value ----------------------------------------------------------------------------


def test_value_matches_get_solution_value():
    vp = ft.Verifier()
    q, z, z_prev = _pgd(vp)
    vp.maximize(ft.norm(z - z_prev, np.inf))
    bound = vp.solve()
    np.testing.assert_allclose(q.value, vp.get_solution_value(q))
    np.testing.assert_allclose((z - z_prev).value, z.value - z_prev.value)
    # The incumbent attains the worst case up to the MIP gap.
    assert abs(np.max(np.abs((z - z_prev).value)) - bound) <= 1e-4 * max(1.0, bound)


def test_value_of_a_constant_expression():
    const = LinearExpression({}, const=np.array([1.0, 2.0]))
    np.testing.assert_array_equal(const.value, [1.0, 2.0])


def test_value_errors():
    with pytest.raises(RuntimeError, match="not registered"):
        LinearExpression.from_int(2).value
    vp = ft.Verifier()
    x = vp.parameter(2, lb=-1, ub=1)
    with pytest.raises(RuntimeError, match="incumbent"):
        x.value
    sim = ft.Verifier(method="simulator")
    y = sim.parameter(2, samples=np.zeros((3, 2)))
    with pytest.raises(NotImplementedError, match="simulator"):
        y.value


# --- verbose --------------------------------------------------------------------------


def _solve_pgd(**kwargs):
    solve_kwargs = kwargs.pop("solve_kwargs", {})
    vp = ft.Verifier(**kwargs)
    _, z, z_prev = _pgd(vp)
    vp.maximize(ft.norm(z - z_prev, np.inf))
    return vp, vp.solve(**solve_kwargs)


def test_highs_is_silent_by_default(capfd):
    _solve_pgd(solve_kwargs={"convex_hull_cuts": True})
    assert capfd.readouterr().out == ""


def test_highs_verbose_prints_the_log(capfd):
    _solve_pgd(verbose=True)
    assert "HiGHS" in capfd.readouterr().out


def test_explicit_verbosity_param_overrides_verbose(capfd):
    vp, _ = _solve_pgd(params={"output_flag": True})
    assert vp.canonicalizer.solver_params["output_flag"] is True
    assert "HiGHS" in capfd.readouterr().out
    vp, _ = _solve_pgd(verbose=True, params={"output_flag": False})
    assert capfd.readouterr().out == ""


def test_solve_verbose_logs_only_that_solve(capfd):
    vp = ft.Verifier()
    _, z, z_prev = _pgd(vp)
    vp.maximize(ft.norm(z - z_prev, np.inf))
    assert capfd.readouterr().out == ""
    vp.solve(verbose=True)
    assert "HiGHS" in capfd.readouterr().out
    vp.solve()
    assert capfd.readouterr().out == ""


def test_obbt_solver_follows_verbose():
    vp = ft.Verifier(verbose=True)
    assert vp.obbt_handler.solver_params == {"output_flag": True}
    vp = ft.Verifier()
    assert vp.obbt_handler.solver_params == {"output_flag": False}
    # An OBBT backend different from the main solver still follows ``verbose``.
    vp = ft.Verifier(method="sdp", solver="clarabel", obbt_solver="highs")
    assert vp.obbt_handler.solver_params == {"output_flag": False}


def test_sdp_verbose_option():
    vp = ft.Verifier(method="sdp", solver="clarabel", obbt_solver=None, verbose=True)
    assert vp.canonicalizer.solver_params == {"verbose": True}
    vp = ft.Verifier(method="sdp", solver="clarabel", obbt_solver=None)
    assert not vp.canonicalizer.solver_params


def test_highs_copies_keep_the_options():
    solver = HiGHSSolver({"output_flag": False, "mip_rel_gap": 0.25})
    copy = solver.copy()
    assert copy.model.getOptionValue("output_flag")[1] is False
    assert copy.model.getOptionValue("mip_rel_gap")[1] == 0.25
    solver.relax_model()
    assert solver._relaxed_model.getOptionValue("output_flag")[1] is False


@requires_gurobi
def test_gurobi_is_silent_by_default(capfd):
    vp, _ = _solve_pgd(solver="gurobi", solve_kwargs={"convex_hull_cuts": True})
    assert vp.canonicalizer.solver.model.Params.OutputFlag == 0
    assert capfd.readouterr().out == ""


@requires_gurobi
def test_gurobi_verbose_prints_the_log(capfd):
    vp, _ = _solve_pgd(solver="gurobi", verbose=True)
    assert vp.canonicalizer.solver.model.Params.OutputFlag == 1
    assert "Gurobi" in capfd.readouterr().out
