"""
Tests for the ``sum_squares`` quadratic constraint atom.

``vp.sum_squares(expr) <op> r**2`` (for ``<op>`` in ``<=`` / ``>=`` / ``==``) builds a
:class:`QuadraticConstraint` encoding ``sum_i (a_i^T x + b_i)^2 <op> r^2``:

- **SDP**: linear in the lifted matrix (``tr(C X) <op> r^2`` with ``C = [b;A]^T[b;A]``),
  a valid relaxation for every sense.
- **MIP/Gurobi**: a quadratic constraint (MIQCP); the nonconvex ``>=`` / ``==`` use
  ``NonConvex=2``.
- **MIP/HiGHS**: unsupported — raises at ``add_constraint``.

The primary validity guard is the MIP <= SDP bound-preservation check (SDP is a
relaxation, so we compare bounds, never read solution values).
"""

import numpy as np
import pytest

from fomtools.constraint import QuadraticConstraint, SumSquares, sum_squares
from fomtools.verifier import Verifier
from tests.conftest import SDP_SOLVERS, TEST_EPS

SENSES = {"<=": "__le__", ">=": "__ge__", "==": "__eq__"}


def _ss(expr, sense, bound):
    """Build a sum_squares constraint of the given sense (``<=`` / ``>=`` / ``==``)."""
    return getattr(sum_squares(expr), SENSES[sense])(bound)


# --------------------------------------------------------------------------------------
# Phase 1: API / plumbing
# --------------------------------------------------------------------------------------


def test_sum_squares_returns_atom():
    vp = Verifier(method="sdp", solver="clarabel")
    x = vp.add_iterate(3, lb=-2, ub=2)
    assert isinstance(vp.sum_squares(x), SumSquares)
    assert isinstance(sum_squares(x), SumSquares)  # free function matches the method


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_comparison_builds_quadratic_constraint(sense):
    vp = Verifier(method="sdp", solver="clarabel")
    x = vp.add_iterate(3, lb=-2, ub=2)
    c = _ss(x, sense, 1.5**2)
    assert isinstance(c, QuadraticConstraint)
    assert c.operator == "sum_squares"
    assert c.sense == sense
    assert c.is_linear is False  # never forwarded to the OBBT LP
    assert c.is_convex == (sense == "<=")
    assert abs(c.radius_sq - 1.5**2) < 1e-12
    assert c.ndim == 3


def test_negative_radius_raises():
    vp = Verifier(method="sdp", solver="clarabel")
    x = vp.add_iterate(2, lb=-1, ub=1)
    with pytest.raises(ValueError, match="nonnegative"):
        sum_squares(x) <= -1.0


# --------------------------------------------------------------------------------------
# Phase 2: HiGHS rejects quadratic constraints (every sense), eagerly at add-time
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_highs_rejects_sum_squares(sense):
    vp = Verifier(method="mip", solver="highs", obbt_solver=None)
    x = vp.add_iterate(3, lb=-2, ub=2)
    with pytest.raises(ValueError, match="MIQCP|Gurobi"):
        vp.add_constraint(_ss(x, sense, 1.0))


# --------------------------------------------------------------------------------------
# Phase 3: SDP build — correct matrix block, symmetry, PSD coupling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_sdp_build_appends_expected_block(sdp_solver, sense):
    vp = Verifier(method="sdp", solver=sdp_solver)
    x = vp.add_iterate(3, lb=-2, ub=2)
    canon = vp.canonicalizer

    n_ineq, n_eq = len(canon.Aineq_matrices), len(canon.Aeq_matrices)
    r_sq = 1.3**2
    vp.add_constraint(_ss(x, sense, r_sq))

    if sense == "==":
        assert len(canon.Aeq_matrices) == n_eq + 1
        M, b = canon.Aeq_matrices[-1], canon.beq_consts[-1]
        assert abs(b - r_sq) < 1e-12
    else:
        assert len(canon.Aineq_matrices) == n_ineq + 1
        M, b = canon.Aineq_matrices[-1], canon.bineq_upperbounds[-1]
        # ``>=`` is stored as ``tr(-C X) <= -r^2``.
        assert abs(b - (r_sq if sense == "<=" else -r_sq)) < 1e-12

    # The appended lifted matrix must be symmetric (trace inner product convention).
    dense = M.toarray()
    assert np.abs(dense - dense.T).max() < 1e-12

    # A PSD cone covering the expression's indices must be registered.
    _, idx_bounds = canon.linexpr_to_Amat_and_bounds(x)
    expr_indices = {i for s, e in idx_bounds for i in range(s, e)}
    assert any(expr_indices <= cone for cone in canon.psd_cone_indices)


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_sdp_rank1_lift_not_cut(sdp_solver, sense):
    """The rank-1 lift of a genuinely feasible point satisfies the appended row."""
    vp = Verifier(method="sdp", solver=sdp_solver)
    x = vp.add_iterate(3, lb=-3, ub=3)
    canon = vp.canonicalizer
    r_sq = 1.7**2
    vp.add_constraint(_ss(x, sense, r_sq))

    dim = canon.get_sdp_var_dim()
    _, idx_bounds = canon.linexpr_to_Amat_and_bounds(x)

    # Pick a feasible primal for the sense: interior (<=), exterior (>=), boundary (==).
    targets = {"<=": np.array([0.5, -0.5, 0.5]), ">=": np.array([2.0, -2.0, 1.0])}
    if sense == "==":
        xv = np.array([1.0, -1.0, 0.0])
        xv = xv / np.linalg.norm(xv) * np.sqrt(r_sq)  # land exactly on the sphere
    else:
        xv = targets[sense]
    ss_val = float(xv @ xv)
    if sense == "<=":
        assert ss_val <= r_sq
    elif sense == ">=":
        assert ss_val >= r_sq

    w = np.zeros(dim)
    w[0] = 1.0
    for s, e in idx_bounds:
        w[s:e] = xv  # identity placement for a leaf iterate
    X = np.outer(w, w)

    if sense == "==":
        M, b = canon.Aeq_matrices[-1], canon.beq_consts[-1]
        assert abs(float(w @ (M @ w)) - b) < 1e-9
    else:
        M, b = canon.Aineq_matrices[-1], canon.bineq_upperbounds[-1]
        assert float(w @ (M @ w)) <= b + 1e-9
    # tr(M X) == w^T M w for the rank-1 lift X = w w^T (sanity).
    assert abs((M.toarray() * X).sum() - float(w @ (M @ w))) < 1e-9


# --------------------------------------------------------------------------------------
# Phase 4: Gurobi MIQCP solve — all senses hit the analytic optimum
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sense,expected",
    [("<=", 1.5), (">=", 5.0), ("==", 1.5)],
)
def test_gurobi_miqcp_solve(sense, expected):
    """max ||x||_inf over x in [-5,5]^3 s.t. ||x||_2^2 <op> R^2.

    - ``<=``/``==`` -> the inf-norm optimum is R (mass in one coordinate on the ball/sphere).
    - ``>=`` -> the shell constraint is non-binding at the box corner, optimum is the box.
    """
    R = 1.5
    vp = Verifier(method="mip", solver="gurobi", obbt_solver=None, params={"OutputFlag": 0})
    x = vp.add_iterate(3, lb=-5, ub=5)
    vp.add_constraint(_ss(x, sense, R**2))
    vp.set_objective_norm(x, "inf")
    assert abs(vp.solve() - expected) < 1e-4


# --------------------------------------------------------------------------------------
# Phase 5: MIP <= SDP bound preservation (primary validity guard), every sense
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_mip_bound_preserved(sdp_solver, sense):
    """For each sense, the SDP relaxation upper-bounds the exact MIP value: no
    feasible point is cut by the lifted encoding."""
    R = 1.2

    vs = Verifier(method="sdp", solver=sdp_solver)
    xs = vs.add_iterate(3, lb=-3, ub=3)
    vs.add_constraint(_ss(xs, sense, R**2))
    vs.set_objective_norm(xs, 2)
    sdp_two_sq = vs.solve()

    vm = Verifier(method="mip", solver="gurobi", obbt_solver=None, params={"OutputFlag": 0})
    xm = vm.add_iterate(3, lb=-3, ub=3)
    vm.add_constraint(_ss(xm, sense, R**2))
    vm.set_objective_norm(xm, "inf")
    mip_inf = vm.solve()

    assert mip_inf <= np.sqrt(sdp_two_sq) + TEST_EPS
