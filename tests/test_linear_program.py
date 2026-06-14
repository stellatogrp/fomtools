"""Tests for the LinearProgram step (QuadraticProgram with P = 0)."""

import numpy as np
import pytest

from fomtools.steps import LinearProgram
from fomtools.verifier import Verifier
from tests.conftest import PROX_ATOL


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    if method == "mip":
        return Verifier(method="mip", solver="gurobi", obbt_solver=obbt, params={"OutputFlag": 0})
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_solver=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_solver=None)
    raise ValueError(f"unknown method: {method}")


def _box(n, lb, ub):
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = np.concatenate([ub, -lb])
    return C, d


def test_scalar_q_rejected():
    with pytest.raises(ValueError, match="vector q"):
        LinearProgram(q=1.0)


def test_p_is_zero():
    lp = LinearProgram(q=np.ones(3))
    assert lp.is_lp is True
    assert lp.P.shape == (3, 3)
    assert lp.P.nnz == 0


def test_solve_reference_box_lp():
    """min q^T x over the box [-1, 1]^2 has the vertex solution -sign(q)."""
    C, d = _box(2, -np.ones(2), np.ones(2))
    lp = LinearProgram(q=np.array([1.0, -1.0]), C=C, d=d, dual_ub=10.0)
    np.testing.assert_allclose(lp.solve(), np.array([-1.0, 1.0]), atol=PROX_ATOL)


def test_mip_box_lp_matches_reference():
    C, d = _box(2, -np.ones(2), np.ones(2))
    q = np.array([1.0, -1.0])
    ref = LinearProgram(q=q, C=C, d=d, dual_ub=10.0).solve()
    v = _make_verifier("mip", obbt="gurobi")
    u = v.linear_program(q=q, C=C, d=d, dual_ub=10.0, primal_bounds=(-np.ones(2), np.ones(2)))
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), ref, atol=PROX_ATOL)


def test_mip_lp_with_equality():
    """min q^T x s.t. 1^T x = 1, x in box; optimum puts mass on the cheapest coord."""
    q = np.array([1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    C, d = _box(2, -np.ones(2), np.ones(2))
    ref = LinearProgram(q=q, A=A, b=b, C=C, d=d, dual_ub=10.0).solve()
    v = _make_verifier("mip", obbt="gurobi")
    u = v.linear_program(
        q=q,
        A=A,
        b=b,
        C=C,
        d=d,
        dual_ub=10.0,
        primal_bounds=(-np.ones(2), np.ones(2)),
        nu_bounds=(np.array([-50.0]), np.array([50.0])),
    )
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), ref, atol=PROX_ATOL)


def test_without_dual_ub_raises():
    C, d = _box(2, -np.ones(2), np.ones(2))
    v = _make_verifier("mip", obbt="gurobi")
    with pytest.raises(ValueError, match="dual_ub"):
        v.linear_program(q=np.array([1.0, -1.0]), C=C, d=d)


def test_simulator_not_supported():
    v = _make_verifier("simulator")
    with pytest.raises(NotImplementedError):
        v.linear_program(q=np.array([1.0, -1.0]))
