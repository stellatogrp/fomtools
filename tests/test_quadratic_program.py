"""Tests for the QuadraticProgram step: P validation, the KKT decomposer wiring,
and MIP/SDP backends against a cvxpy reference."""

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import IndPolyhedral
from fomtools.steps import QuadraticProgram
from fomtools.verifier import Verifier
from tests.conftest import PROX_ATOL, TEST_EPS


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    if method == "mip":
        return Verifier(method="mip", solver="gurobi", obbt_solver=obbt, params={"OutputFlag": 0})
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_solver=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_solver=None)
    raise ValueError(f"unknown method: {method}")


def _box(n, lb, ub):
    """Return (C, d) describing the box {lb <= x <= ub} as C x <= d."""
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = np.concatenate([ub, -lb])
    return C, d


# --- construction / validation ---------------------------------------------


def test_rejects_nonsymmetric_P():
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticProgram(P=np.array([[1.0, 2.0], [0.0, 1.0]]), q=np.zeros(2))


def test_rejects_indefinite_P():
    with pytest.raises(ValueError, match="positive semidefinite"):
        QuadraticProgram(P=np.array([[1.0, 0.0], [0.0, -1.0]]), q=np.zeros(2))


def test_accepts_psd_identity_and_sparse():
    QuadraticProgram(P=np.eye(3), q=np.zeros(3))
    QuadraticProgram(P=np.diag([2.0, 0.0, 1.0]), q=np.zeros(3))  # singular PSD ok
    QuadraticProgram(P=spa.eye(3), q=np.zeros(3))


def test_q_none_requires_matching_rhs():
    arg = LinearExpression.from_int(3)
    QuadraticProgram(P=np.eye(3), q=None, arg=arg)  # ok: arg.ndim == n
    with pytest.raises(ValueError, match="arg"):
        QuadraticProgram(P=np.eye(3), q=None, arg=LinearExpression.from_int(2))
    with pytest.raises(ValueError, match="arg"):
        QuadraticProgram(P=np.eye(3), q=None)  # no arg at all


def test_dimension_mismatches():
    with pytest.raises(ValueError, match="q dim"):
        QuadraticProgram(P=np.eye(3), q=np.zeros(2))
    with pytest.raises(ValueError, match="columns"):
        QuadraticProgram(P=np.eye(3), q=np.zeros(3), A=np.ones((1, 2)), b=np.zeros(1))
    with pytest.raises(ValueError, match="b dim"):
        QuadraticProgram(P=np.eye(2), q=np.zeros(2), A=np.ones((2, 2)), b=np.zeros(3))


# --- numeric reference (cvxpy solve) ----------------------------------------


def test_solve_reference_box_projection():
    """solve() projects onto a box (P=I) == elementwise clip."""
    C, d = _box(2, -np.ones(2), np.ones(2))
    qp = QuadraticProgram(P=np.eye(2), q=-np.array([2.0, -3.0]), C=C, d=d, dual_ub=10.0)
    np.testing.assert_allclose(qp.solve(), np.array([1.0, -1.0]), atol=PROX_ATOL)


# --- MIP backend: exact match to the reference ------------------------------

# (P, q, A, b, C, d, primal_bounds, nu_bounds, dual_ub)
_MIP_CASES = {
    "ineq_box": dict(
        P=np.eye(2),
        q=-np.array([2.0, -3.0]),
        C=np.vstack([np.eye(2), -np.eye(2)]),
        d=np.array([1.0, 1.0, 1.0, 1.0]),
        dual_ub=10.0,
    ),
    "eq_only_pd": dict(
        P=np.diag([2.0, 1.0]),
        q=np.array([-1.0, 0.5]),
        A=np.ones((1, 2)),
        b=np.array([1.0]),
        primal_bounds=(-5 * np.ones(2), 5 * np.ones(2)),
        nu_bounds=(np.array([-50.0]), np.array([50.0])),
    ),
    "mixed": dict(
        P=np.eye(2),
        q=-np.array([0.7, 0.7]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([1.0]),
        C=-np.eye(2),
        d=np.zeros(2),  # x >= 0
        dual_ub=10.0,
        primal_bounds=(-2 * np.ones(2), 2 * np.ones(2)),
        nu_bounds=(np.array([-50.0]), np.array([50.0])),
    ),
}


@pytest.mark.parametrize("case", list(_MIP_CASES))
def test_mip_matches_reference(case):
    kw = _MIP_CASES[case]
    ref = QuadraticProgram(**kw).solve()
    v = _make_verifier("mip", obbt="gurobi")
    u = v.quadratic_program(**kw)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), ref, atol=PROX_ATOL)


@pytest.mark.parametrize("P", [np.eye(2), spa.eye(2)], ids=["dense", "sparse"])
def test_mip_dense_and_sparse_P_agree(P):
    """A box projection is identical for dense and sparse P=I."""
    C, d = _box(2, -np.ones(2), np.ones(2))
    x0 = np.array([2.0, -3.0])
    v = _make_verifier("mip", obbt="gurobi")
    xp = v.parameter(2, lb=x0, ub=x0)
    u = v.quadratic_program(P=P, q=-xp, C=C, d=d, dual_ub=10.0)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), np.array([1.0, -1.0]), atol=PROX_ATOL)


def test_mip_parametric_q_b():
    """q = -arg and a parametric equality RHS b are tied to pinned params."""
    x0 = np.array([0.4, 0.1])
    b0 = np.array([1.0])
    A = np.array([[1.0, 1.0]])
    v = _make_verifier("mip", obbt="gurobi")
    xp = v.parameter(2, lb=x0, ub=x0)
    bp = v.parameter(1, lb=b0, ub=b0)
    u = v.quadratic_program(
        P=np.eye(2),
        q=-xp,
        A=A,
        b=bp,
        primal_bounds=(-5 * np.ones(2), 5 * np.ones(2)),
        nu_bounds=(np.array([-50.0]), np.array([50.0])),
    )
    v.set_objective_norm([u], np.inf)
    v.solve()
    ref = QuadraticProgram(P=np.eye(2), q=-x0, A=A, b=b0).solve()
    np.testing.assert_allclose(v.get_solution_value(u), ref, atol=PROX_ATOL)


def test_mip_reproduces_indpolyhedral_projection():
    """P=I, q=-arg, (C, d) reproduces the IndPolyhedral projection."""
    Apoly = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    bpoly = np.array([1.0, 1.0, -0.5])
    x0 = np.array([3.0, 3.0])
    v = _make_verifier("mip", obbt="gurobi")
    xp = v.parameter(2, lb=x0, ub=x0)
    u = v.quadratic_program(P=np.eye(2), q=-xp, C=Apoly, d=bpoly, dual_ub=20.0)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(
        v.get_solution_value(u), IndPolyhedral(Apoly, bpoly).prox(x0), atol=PROX_ATOL
    )


# --- error / deferral paths -------------------------------------------------


@pytest.mark.parametrize("method", ["mip", "sdp"])
def test_without_dual_ub_raises(method):
    """Inequality constraints without dual_ub raise eagerly on both backends."""
    C, d = _box(2, -np.ones(2), np.ones(2))
    v = _make_verifier(method, obbt="gurobi" if method == "mip" else None)
    x = v.parameter(2, lb=np.array([2.0, -3.0]), ub=np.array([2.0, -3.0]))
    with pytest.raises(ValueError, match="dual_ub"):
        v.quadratic_program(P=np.eye(2), q=-x, C=C, d=d)


def test_singular_P_without_primal_bounds_raises():
    """A singular (non-identity) P cannot pin x bounds from stationarity."""
    v = _make_verifier("mip", obbt=None)
    with pytest.raises(ValueError, match="primal_bounds"):
        v.quadratic_program(P=np.diag([1.0, 0.0]), q=np.array([0.0, 0.0]))


def test_simulator_equality_only_supported():
    """Simulator lowers an unconstrained / equality-only QP to a batched Affine
    block-KKT solve."""
    v = _make_verifier("simulator")
    arg = v.parameter(2, lb=-1.0, ub=1.0)
    out = v.quadratic_program(P=np.eye(2), q=-arg)
    assert out.ndim == 2


def test_simulator_inequality_supported():
    """A QP with inequality constraints lowers to a per-sample cvxpy solve."""
    v = _make_verifier("simulator")
    arg = v.parameter(2, lb=-1.0, ub=1.0)
    out = v.quadratic_program(P=np.eye(2), q=-arg, C=np.eye(2), d=np.ones(2), dual_ub=10.0)
    assert out.ndim == 2


# --- SDP backend (complementarity encoding) ---------------------------------


@pytest.mark.parametrize(
    "C, d, x0, dual_ub",
    [
        (np.array([[1.0, 1.0]]), np.array([0.5]), np.array([1.0, 1.0]), 10.0),
        (*_box(2, -np.ones(2), np.ones(2)), np.array([2.0, -3.0]), 10.0),
    ],
)
def test_sdp_projection_bound_tight_at_fixed_point(C, d, x0, dual_ub):
    """At a fixed arg, max ||x||^2 over the relaxed KKT equals ||proj(x0)||^2.

    SDP is a relaxation with no primal extraction, so we assert via the objective bound.
    """
    v = _make_verifier("sdp")
    xp = v.parameter(len(x0), lb=x0, ub=x0)
    u = v.quadratic_program(P=np.eye(len(x0)), q=-xp, C=C, d=d, dual_ub=dual_ub)
    v.set_objective_norm(u, 2)
    obj = v.solve()
    proj = QuadraticProgram(P=np.eye(len(x0)), q=-x0, C=C, d=d, dual_ub=dual_ub).solve()
    np.testing.assert_allclose(obj, float(proj @ proj), atol=1e-3)


def test_sdp_bound_dominates_mip_over_box():
    """SDP two-norm bound dominates the MIP inf-norm over a box of arg."""
    C, d = np.array([[1.0, 1.0]]), np.array([0.5])
    arg_lb, arg_ub = -np.ones(2), np.ones(2)

    vm = _make_verifier("mip", obbt="gurobi")
    xm = vm.parameter(2, lb=arg_lb, ub=arg_ub)
    um = vm.quadratic_program(P=np.eye(2), q=-xm, C=C, d=d, dual_ub=10.0)
    vm.set_objective_norm([um], np.inf)
    mip_inf = vm.solve()

    vs = _make_verifier("sdp")
    xs = vs.parameter(2, lb=arg_lb, ub=arg_ub)
    us = vs.quadratic_program(P=np.eye(2), q=-xs, C=C, d=d, dual_ub=10.0)
    vs.set_objective_norm(us, 2)
    sdp_two = np.sqrt(vs.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS
