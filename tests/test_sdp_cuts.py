"""
Tests for SDP RLT and lifted triangle cuts.

The cuts are valid inequalities from §4.5 of arXiv:2403.03331 and tighten
the SDP relaxation while preserving the MIP lower-bound property.

This file is built up phase-by-phase alongside the implementation in
fomtools/canonicalizers/sdp.py. Each phase adds a new section.
"""
import numpy as np
import pytest

from fomtools.verifier import Verifier
from tests.conftest import SDP_SOLVERS, TEST_EPS


def _build_nnls_sdp(K, sdp_solver, use_rlt_cuts=False, use_triangle_cuts=False):
    """Build a small NNLS SDP verifier matching tests/test_nnls_sdp_mip.py.

    Returns the solved SDP two-norm-squared objective value.
    """
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    vp = Verifier(
        method='sdp',
        solver=sdp_solver,
        obbt_handler='highs',
        use_rlt_cuts=use_rlt_cuts,
        use_triangle_cuts=use_triangle_cuts,
    )

    x = vp.add_param(m, lb=0, ub=0.1)
    z = [vp.add_iterate(n, lb=5, ub=5.1)]

    for k in range(K):
        z.append(vp.step.relu(z[k] - t * A.T @ (A @ z[k] - x)))

    vp.set_objective_norm(z[K] - z[K - 1], 2)
    return vp, vp.solve()


# ============================================================
# Phase 0: scaffolding — flags exist but do nothing
# ============================================================


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_flags_default_off_matches_baseline(sdp_solver):
    """With both flags off, SDP objective matches the existing baseline."""
    _, baseline = _build_nnls_sdp(K=1, sdp_solver=sdp_solver)
    _, with_flags_off = _build_nnls_sdp(
        K=1, sdp_solver=sdp_solver, use_rlt_cuts=False, use_triangle_cuts=False
    )
    assert abs(baseline - with_flags_off) <= TEST_EPS


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_flags_stored_on_canonicalizer(sdp_solver):
    """The flags reach the SDP canonicalizer instance."""
    vp = Verifier(
        method='sdp',
        solver=sdp_solver,
        obbt_handler=None,
        use_rlt_cuts=True,
        use_triangle_cuts=True,
    )
    assert vp.canonicalizer.use_rlt_cuts is True
    assert vp.canonicalizer.use_triangle_cuts is True

    vp2 = Verifier(method='sdp', solver=sdp_solver, obbt_handler=None)
    assert vp2.canonicalizer.use_rlt_cuts is False
    assert vp2.canonicalizer.use_triangle_cuts is False


# ============================================================
# Phase 1: cut-rebuild infrastructure (no cuts yet)
# ============================================================


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_cuts_dirty_triggers(sdp_solver):
    """add_var, canonicalize_relu, add_constraint, update_bounds all mark dirty."""
    vp = Verifier(
        method='sdp',
        solver=sdp_solver,
        obbt_handler=None,
        use_rlt_cuts=True,
    )
    canon = vp.canonicalizer

    # Fresh canonicalizer: dirty is True (rebuild needed on first solve).
    assert canon._cuts_dirty is True

    # add_var marks dirty.
    canon._cuts_dirty = False
    x = vp.add_param(2, lb=0, ub=1)
    assert canon._cuts_dirty is True

    # canonicalize_relu marks dirty + registers the step.
    canon._cuts_dirty = False
    z0 = vp.add_iterate(2, lb=-1, ub=1)
    z1 = vp.step.relu(z0)
    assert canon._cuts_dirty is True
    assert len(canon._relu_steps) >= 1

    # update_bounds marks dirty.
    canon._cuts_dirty = False
    canon.update_bounds(z0, np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
    assert canon._cuts_dirty is True


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rebuild_runs_with_flags_on_but_helpers_empty(sdp_solver):
    """With flags on but helpers empty (Phase 1), solve still works.

    Cut lists remain empty because _build_rlt_cuts / _build_triangle_cuts
    are stubs in Phase 1.
    """
    vp, obj = _build_nnls_sdp(
        K=1, sdp_solver=sdp_solver, use_rlt_cuts=True, use_triangle_cuts=True
    )
    canon = vp.canonicalizer
    # solve_model copies the canonicalizer and solves the copy; the copy's
    # dirty flag flips to False after the rebuild runs.
    assert canon._solved_copy is not None
    assert canon._solved_copy._cuts_dirty is False
    # Phase 1: helpers are stubs, so cut lists stay empty after rebuild.
    assert canon._solved_copy._cut_ineq_matrices == []
    assert canon._solved_copy._cut_bineq_upperbounds == []
    # Objective should match the no-cuts baseline since helpers are stubs.
    _, baseline = _build_nnls_sdp(K=1, sdp_solver=sdp_solver)
    assert abs(obj - baseline) <= TEST_EPS


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_copy_resets_dirty(sdp_solver):
    """copy() should reset _cuts_dirty so the copy rebuilds on first solve."""
    vp = Verifier(
        method='sdp',
        solver=sdp_solver,
        obbt_handler=None,
        use_rlt_cuts=True,
    )
    z = vp.add_iterate(2, lb=-1, ub=1)
    _ = vp.step.relu(z)
    canon = vp.canonicalizer

    # Force a "rebuild done" state on the original.
    canon._cuts_dirty = False
    copy = canon.copy()
    assert copy._cuts_dirty is True
    assert copy._cut_ineq_matrices == []
    assert copy._cut_bineq_upperbounds == []
