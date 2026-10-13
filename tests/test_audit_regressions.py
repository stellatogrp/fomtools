"""Failure cases from the paper/library audit, including solver acceptance gates."""

from types import SimpleNamespace

import clarabel
import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as spa

from fomtools import LinearExpression, Verifier
from fomtools.proxable_functions import ElasticNet, IndAffine, NormL1
from fomtools.solvers.highs import HiGHSSolver
from tests.conftest import mip_params


@pytest.mark.parametrize("has_incumbent", [False, True])
@pytest.mark.parametrize("bound", [3.0, np.inf, np.nan, 1e100])
def test_gurobi_time_limit_bound(has_incumbent, bound):
    gp = pytest.importorskip("gurobipy")
    from fomtools.solvers.gurobi import GurobiSolver

    solver = GurobiSolver.__new__(GurobiSolver)
    solver.model = SimpleNamespace(
        Status=gp.GRB.TIME_LIMIT, IsMIP=1, ObjBound=bound, ObjVal=1.0, SolCount=int(has_incumbent)
    )
    if np.isfinite(bound) and bound < 1e100:
        assert solver.get_best_bound() == 3.0
    else:
        with pytest.raises(RuntimeError, match="finite"):
            solver.get_best_bound()


@pytest.mark.parametrize(
    "status_name", ["INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "NUMERIC", "SUBOPTIMAL"]
)
def test_gurobi_rejects_bad_status_even_with_finite_value(status_name):
    gp = pytest.importorskip("gurobipy")
    from fomtools.solvers.gurobi import GurobiSolver

    solver = GurobiSolver.__new__(GurobiSolver)
    solver.model = SimpleNamespace(Status=getattr(gp.GRB, status_name), ObjBound=1.0)
    with pytest.raises(RuntimeError, match="status"):
        solver.get_best_bound()


@pytest.mark.parametrize("has_incumbent", [False, True])
@pytest.mark.parametrize(
    "status_name,bound,accepted",
    [
        ("kTimeLimit", 3.0, True),
        ("kTimeLimit", np.inf, False),
        ("kInfeasible", 3.0, False),
        ("kUnbounded", 3.0, False),
        ("kUnboundedOrInfeasible", 3.0, False),
        ("kSolveError", 3.0, False),
    ],
)
def test_highs_status_bound(has_incumbent, status_name, bound, accepted):
    import highspy as hp

    solver = HiGHSSolver.__new__(HiGHSSolver)
    info = SimpleNamespace(
        valid=True,
        mip_dual_bound=bound,
        objective_function_value=1.0,
        primal_solution_status=2 if has_incumbent else 0,
    )
    solver.model = SimpleNamespace(
        getModelStatus=lambda: getattr(hp.HighsModelStatus, status_name),
        getInfo=lambda: info,
        getLp=lambda: SimpleNamespace(integrality_=[hp.HighsVarType.kInteger]),
    )
    if accepted:
        assert solver.get_best_bound() == 3.0
    else:
        with pytest.raises(RuntimeError):
            solver.get_best_bound()


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_solve_returns_bound_not_incumbent(monkeypatch, solver):
    v = Verifier(solver=solver, obbt_solver=None, params=mip_params(solver))
    x = v.add_iterate(1, lb=-1, ub=2)
    v.set_objective_norm(x, np.inf)
    cls = type(v.canonicalizer.solver)
    monkeypatch.setattr(cls, "get_best_bound", lambda self: 3.0)
    assert v.solve() == 3.0
    assert v.canonicalizer._solved_copy["objVal"] == pytest.approx(2.0)


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_infeasible_model_has_no_scalar_or_solution(solver):
    v = Verifier(solver=solver, obbt_solver=None, params=mip_params(solver))
    x = v.add_iterate(1, lb=0, ub=1)
    v.add_constraint(x >= 2)
    v.set_zero_objective()
    with pytest.raises(RuntimeError):
        v.solve()
    with pytest.raises(RuntimeError, match="incumbent"):
        v.get_solution_value(x + 1)


@pytest.mark.parametrize(
    "status",
    [
        clarabel.SolverStatus.AlmostSolved,
        clarabel.SolverStatus.MaxIterations,
        clarabel.SolverStatus.PrimalInfeasible,
        clarabel.SolverStatus.DualInfeasible,
        clarabel.SolverStatus.NumericalError,
    ],
)
def test_native_sdp_rejects_unaccepted_solution(monkeypatch, status):
    v = Verifier(method="sdp", solver="clarabel", obbt_solver=None)
    x = v.add_iterate(1, lb=-1, ub=1)
    v.set_objective_norm(x, 2)
    monkeypatch.setattr(
        clarabel,
        "DefaultSolver",
        lambda *args: SimpleNamespace(
            solve=lambda: SimpleNamespace(status=status, obj_val=-1.0, obj_val_dual=-1.1)
        ),
    )
    with pytest.raises(RuntimeError, match="terminated"):
        v.solve()


@pytest.mark.parametrize(
    "status", [cp.OPTIMAL_INACCURATE, cp.USER_LIMIT, cp.INFEASIBLE, cp.UNBOUNDED]
)
def test_cvxpy_sdp_rejects_unaccepted_solution(monkeypatch, status):
    def solve(problem, **kwargs):
        problem._status = status
        return 1.0

    monkeypatch.setattr(cp.Problem, "solve", solve)
    v = Verifier(method="sdp", solver="cvxpy", obbt_solver=None)
    x = v.add_iterate(1, lb=-1, ub=1)
    v.set_objective_norm(x, 2)
    with pytest.raises(RuntimeError, match="terminated"):
        v.solve()


@pytest.mark.parametrize("solver", ["clarabel", "cvxpy"])
def test_sdp_iteration_limit_is_rejected(solver):
    v = Verifier(method="sdp", solver=solver, obbt_solver=None)
    x = v.add_iterate(2, lb=-1, ub=1)
    z = v.step.relu(x + 0.1)
    v.set_objective_norm(z - x, 2)
    with pytest.raises(RuntimeError):
        v.solve(max_iter=0)


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_non_singleton_theory_and_postprocess_bounds(solver):
    v = Verifier(
        solver=solver, obbt_solver=solver, theory_func=lambda k: 0.5, params=mip_params(solver)
    )
    ref = v.add_param(2, lb=np.array([1.0, 2.0]), ub=np.array([3.0, 5.0]))
    target = v.add_iterate(2, lb=-10, ub=10)
    assert v.theory_bound(1, target, ref) == 1.0
    np.testing.assert_allclose(target.lb, [0.5, 1.5])
    np.testing.assert_allclose(target.ub, [3.5, 5.5])
    assert v.theory_bound(1, target, ref) == 0.0
    v.set_objective_norm(target, np.inf)
    assert v.solve() == pytest.approx(5.5)
    # The OBBT model receives the same bounds as the Python leaf and main model.
    lb, ub = v.obbt_handler.obbt(target, np.full(2, -10.0), np.full(2, 10.0))
    np.testing.assert_allclose(lb, target.lb, atol=1e-5)
    np.testing.assert_allclose(ub, target.ub, atol=1e-5)
    later = v.add_iterate(2, lb=-10, ub=10)
    assert v.post_process(later, ref, [0.5, 1.0]) == 1.0
    np.testing.assert_allclose(later.lb, [-0.5, 0.5])
    np.testing.assert_allclose(later.ub, [4.5, 6.5])
    assert v.post_process(later, ref, [np.inf]) == 0.0
    v.set_objective_norm(later, np.inf)
    assert v.solve() == pytest.approx(6.5)
    for invalid in ([-1.0], [np.nan], [], [[1.0]]):
        with pytest.raises(ValueError):
            v.post_process(later, ref, invalid)


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_expression_values_preserve_constants(solver):
    v = Verifier(solver=solver, obbt_solver=None, params=mip_params(solver))
    x = v.add_iterate(2, lb=np.array([1.0, -2.0]), ub=np.array([1.0, -2.0]))
    A = np.array([[1.0, 2.0], [3.0, -1.0]])
    expr = A @ x + np.array([4.0, -7.0])
    v.set_zero_objective()
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(expr), [1.0, -2.0])
    np.testing.assert_allclose(
        v.get_solution_value(LinearExpression({}, np.array([2.0, 3.0]))), [2.0, 3.0]
    )


def test_vector_multiplication_and_bad_shapes_are_rejected():
    x = LinearExpression.from_int(2)
    for A in (np.array([[1.0, 2.0], [3.0, 4.0]]), spa.csr_matrix([[1.0, 2.0], [3.0, 4.0]])):
        expr = A @ x
        with pytest.raises(TypeError, match="scalar"):
            _ = expr * np.array([2.0, 3.0])
        with pytest.raises(TypeError, match="scalar"):
            _ = np.array([2.0, 3.0]) * expr
    for coeff, const in [
        (np.ones(2), None),
        (np.ones((2, 3)), None),
        (np.eye(2), np.ones((2, 1))),
        (np.eye(2), np.ones(3)),
    ]:
        with pytest.raises(ValueError):
            LinearExpression({x: coeff}, const)


@pytest.mark.parametrize("kind", ["affine", "elastic"])
def test_default_simulator_residuals_follow_prox_outputs(kind):
    v = Verifier(method="simulator", obbt_solver=None)
    v.add_param(1, samples=np.zeros((3, 1)))
    initial = np.array([2.0, -1.0])
    z = v.add_iterate(2, lb=initial, ub=initial)
    if kind == "affine":
        prox = IndAffine(A=np.array([[1.0, 1.0]]), b=np.array([0.5]))

        def forward(x):
            return x - (x.sum() - 0.5) / 2
    else:
        prox = ElasticNet(alpha=0.2, beta=0.5)
        forward = prox.prox
    expected = []
    current = initial
    for _ in range(2):
        z = v.prox(prox, z)
        next_value = forward(current)
        expected.append(np.max(np.abs(next_value - current)))
        current = next_value
    residuals = v.solve(device="cpu")
    np.testing.assert_allclose(residuals, np.tile(expected, (3, 1)), atol=1e-6)


def test_prox_subclass_inherits_lowering():
    class WeightedL1(NormL1):
        pass

    v = Verifier(method="simulator", obbt_solver=None)
    p = v.add_param(2, samples=np.array([[1.0, -2.0]]))
    out = v.prox(WeightedL1(lambd=0.5), p)
    v.set_objective_norm(out, np.inf)
    np.testing.assert_allclose(v.solve(), [[1.5]])


def test_obbt_padding_covers_small_lp_error(monkeypatch):
    v = Verifier(params={"output_flag": False})
    x = v.add_iterate(1, lb=-2, ub=2)
    handler = v.obbt_handler
    # A computed interval slightly inside the exact [-1, 1] range.
    values = iter([-0.99999995, 0.99999995])
    monkeypatch.setattr(handler.solver, "optimize", lambda: None)
    monkeypatch.setattr(handler.solver, "is_optimal", lambda: True)
    monkeypatch.setattr(type(handler.solver), "__getitem__", lambda self, key: next(values))
    lb, ub = handler.obbt(x, x.lb, x.ub)
    assert lb[0] <= -1 and ub[0] >= 1
    assert lb[0] >= -2 and ub[0] <= 2


def test_unsupported_theory_cut_does_not_mutate_bounds():
    v = Verifier(method="sdp", solver="clarabel", obbt_solver=None, theory_func=lambda k: 0.5)
    ref = v.add_iterate(1, lb=0, ub=1)
    target = v.add_iterate(1, lb=-10, ub=10)
    with pytest.raises(NotImplementedError):
        v.theory_bound(1, target, ref)
    np.testing.assert_array_equal(target.lb, [-10.0])
    np.testing.assert_array_equal(target.ub, [10.0])
