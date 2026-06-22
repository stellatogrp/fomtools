import numpy as np
import pytest

from fomtools.constraint import ComplementarityConstraint
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Map MIP solver class names to solver identifier strings
MIP_SOLVER_NAMES = {
    "GurobiSolver": "gurobi",
    "HiGHSSolver": "highs",
}


def test_complementarity_constructor_dim_mismatch(vp):
    """
    ``vp.complementarity`` validates dimensions and returns a non-linear constraint.

    Mismatched dimensions raise ``ValueError``; equal dimensions yield a
    ``ComplementarityConstraint`` flagged ``is_linear is False`` (so the router
    keeps it out of the OBBT LP).
    """
    x = vp.add_iterate(3, lb=0, ub=10)
    y = vp.add_iterate(2, lb=0, ub=10)

    with pytest.raises(ValueError):
        vp.complementarity(x, y)

    z = vp.add_iterate(3, lb=0, ub=10)
    comp = vp.complementarity(x, z)
    assert isinstance(comp, ComplementarityConstraint)
    assert comp.is_linear is False
    assert comp.ndim == 3


def test_complementarity_enforced_in_solution(vp):
    """
    The optimal solution respects complementarity: at most one side nonzero per coord.

    Two iterates ``a, b in [0, 10]`` with an objective that wants both large. Without
    complementarity the optimum would push both to 10; with it, every coordinate must
    have ``min(a_i, b_i) == 0``.
    """
    n = 3
    a = vp.add_iterate(n, lb=0, ub=10)
    b = vp.add_iterate(n, lb=0, ub=10)

    vp.add_constraint(vp.complementarity(a, b))
    vp.set_objective_norm([a + b], np.inf)
    result = vp.solve()

    assert result is not None
    assert np.isfinite(result)

    a_val = vp.get_solution_value(a)
    b_val = vp.get_solution_value(b)
    assert np.all(np.minimum(a_val, b_val) < TEST_EPS), (
        f"Complementarity violated: a={a_val}, b={b_val}"
    )


def test_complementarity_blocks_both_positive(vp):
    """
    Complementarity together with ``a >= 1`` and ``b >= 1`` is infeasible.

    Forcing both sides strictly positive contradicts ``a_i * b_i == 0``. The MIP must
    report infeasibility. Backends differ in how they signal this: Gurobi raises on
    objVal access, HiGHS returns a non-finite objective; accept either.
    """
    n = 2
    a = vp.add_iterate(n, lb=0, ub=5)
    b = vp.add_iterate(n, lb=0, ub=5)

    vp.add_constraint(vp.complementarity(a, b))
    vp.add_constraint(a >= 1)
    vp.add_constraint(b >= 1)
    vp.set_objective_norm([a + b], np.inf)

    raised = False
    result = None
    try:
        result = vp.solve()
    except Exception:
        raised = True

    assert raised or (result is None) or (not np.isfinite(result)), (
        f"Expected infeasibility, got finite result {result}"
    )


def test_complementarity_infinite_bound_raises():
    """
    Big-M encoding requires finite upper bounds; otherwise a clear ``ValueError``.
    """
    vp = Verifier(method="mip", solver="highs", obbt_solver=None, params={"output_flag": False})
    a = vp.add_iterate(2, lb=0, ub=np.inf)
    b = vp.add_iterate(2, lb=0, ub=5)

    with pytest.raises(ValueError, match="finite upper bounds"):
        vp.add_constraint(vp.complementarity(a, b))


def test_router_skips_obbt_for_complementarity(vp):
    """
    With OBBT enabled, a complementarity constraint is not forwarded into the LP.

    The ``vp`` fixture has the OBBT handler enabled. Adding a complementarity
    constraint (``is_linear=False``) must not crash in the OBBT/bound-tightening path,
    and the problem must still solve correctly with complementarity enforced.
    """
    n = 2
    a = vp.add_iterate(n, lb=0, ub=10)
    b = vp.add_iterate(n, lb=0, ub=10)

    # Should not raise (would AttributeError if routed into _tighten_constraint_bounds).
    vp.add_constraint(vp.complementarity(a, b))
    vp.set_objective_norm([a + b], np.inf)
    result = vp.solve()

    assert np.isfinite(result)
    a_val = vp.get_solution_value(a)
    b_val = vp.get_solution_value(b)
    assert np.all(np.minimum(a_val, b_val) < TEST_EPS)


def test_linear_constraint_still_tightens_with_obbt(vp):
    """
    Regression: the linear-constraint OBBT path is unchanged by the router guard.

    Mirrors ``test_add_constraint_tightens_bounds``: ``x == p`` with ``p in [0, 10]``
    should bound ``relu(x)`` to at most 10 even though ``x`` starts at ``[-100, 100]``.
    """
    n = 2
    p = vp.add_param(n, lb=0, ub=10)
    x = vp.add_iterate(n, lb=-100, ub=100)

    vp.add_constraint(x == p)
    z = vp.step.relu(x)
    vp.set_objective_norm([z], np.inf)
    result = vp.solve()

    assert np.isfinite(result)
    assert result <= 10 + TEST_EPS


# --- SDP backend (directly exercises _add_complementarity_constraint_impl) ---


def test_complementarity_sdp_valid_bound_vs_mip(vp_sdp):
    """
    The generic complementarity primitive on SDP is a valid relaxation of the MIP.

    ``a ⊥ b`` with ``a, b in [0, 2]``, maximizing the norm of ``a + b``. The SDP encoding
    relaxes the bilinear ``a_i b_i = 0`` (lifted product + nonnegativity + PSD cone), so it
    yields a valid bound: ``mip_inf <= sqrt(sdp_two)`` (since ``||.||_inf <= ||.||_2``).
    Exercises ``_add_complementarity_constraint_impl`` end-to-end, decoupled from any prox.
    """
    n = 3
    sdp_solver = vp_sdp.canonicalizer.solver_name

    vm = Verifier(method="mip", solver="gurobi", obbt_solver=None, params={"OutputFlag": 0})
    am = vm.add_iterate(n, lb=0, ub=2)
    bm = vm.add_iterate(n, lb=0, ub=2)
    vm.add_constraint(vm.complementarity(am, bm))
    vm.set_objective_norm([am + bm], np.inf)
    mip_inf = vm.solve()

    vs = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    a = vs.add_iterate(n, lb=0, ub=2)
    b = vs.add_iterate(n, lb=0, ub=2)
    vs.add_constraint(vs.complementarity(a, b))
    vs.set_objective_norm(a + b, 2)
    sdp_two = np.sqrt(vs.solve())

    assert np.isfinite(mip_inf)
    assert mip_inf <= sdp_two + TEST_EPS, f"MIP inf {mip_inf} should be <= SDP two {sdp_two}"


def test_complementarity_sdp_disjoint_guard(vp_sdp):
    """
    SDP complementarity between expressions sharing variables raises (the disjoint guard).

    Tests the guard at the primitive level: ``x ⊥ x`` shares the leaf ``x``, which the
    lifted outer-product scatter cannot encode.
    """
    x = vp_sdp.add_iterate(2, lb=0, ub=1)
    with pytest.raises(NotImplementedError, match="sharing variables"):
        vp_sdp.add_constraint(vp_sdp.complementarity(x, x))
