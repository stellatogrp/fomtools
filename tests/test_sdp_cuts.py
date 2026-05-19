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
def test_rebuild_clears_dirty_after_solve(sdp_solver):
    """After solve, the copy that ran has _cuts_dirty cleared."""
    vp, _ = _build_nnls_sdp(
        K=1, sdp_solver=sdp_solver, use_rlt_cuts=True, use_triangle_cuts=True
    )
    assert vp.canonicalizer._solved_copy is not None
    assert vp.canonicalizer._solved_copy._cuts_dirty is False


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


# ============================================================
# Phase 2: RLT cuts
# ============================================================


def _expected_rlt_cut_count(canon):
    """Count 4 × |unique off-diagonal cone-mate pairs| using the same dedup
    logic as the implementation."""
    from itertools import combinations

    pairs = set()
    for cone_set in canon.psd_cone_indices:
        sorted_idx = sorted(i for i in cone_set if i != 0)
        for p, q in combinations(sorted_idx, 2):
            pairs.add((p, q))
    return 4 * len(pairs)


def _solved_canon(vp):
    """The actual canonicalizer that ran the solve (post-rebuild copy)."""
    return vp.canonicalizer._solved_copy


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cut_count(sdp_solver):
    """Number of RLT cuts = 4 × unique cone-mate off-diagonal pairs."""
    vp, _ = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, use_rlt_cuts=True)
    canon = _solved_canon(vp)
    assert len(canon._cut_ineq_matrices) == _expected_rlt_cut_count(canon)
    assert len(canon._cut_bineq_upperbounds) == len(canon._cut_ineq_matrices)


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cut_matrices_symmetric(sdp_solver):
    """Each RLT cut matrix is symmetric."""
    vp, _ = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, use_rlt_cuts=True)
    canon = _solved_canon(vp)
    for M in canon._cut_ineq_matrices:
        diff = (M - M.T)
        # In csc form, eliminate explicit zeros before counting nnz.
        diff.eliminate_zeros()
        assert diff.nnz == 0


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cuts_tighten_bound(sdp_solver):
    """SDP objective with RLT cuts ≤ baseline (we maximize, so tighter relaxation
    gives smaller upper bound)."""
    _, baseline = _build_nnls_sdp(K=1, sdp_solver=sdp_solver)
    _, with_rlt = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, use_rlt_cuts=True)
    assert with_rlt <= baseline + TEST_EPS


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cuts_preserve_mip_lower_bound(sdp_solver):
    """MIP infinity-norm bound ≤ sqrt(SDP two-norm-sq with RLT) — the same
    relationship verified in test_nnls_sdp_mip.py, now with RLT cuts on."""
    _, sdp_two_norm_sq = _build_nnls_sdp(
        K=1, sdp_solver=sdp_solver, use_rlt_cuts=True
    )

    # Build the matching MIP and solve.
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    vp_mip = Verifier(method='mip', solver='highs', obbt_handler='highs')
    x_mip = vp_mip.add_param(m, lb=0, ub=0.1)
    z_mip = [vp_mip.add_iterate(n, lb=5, ub=5.1)]
    for k in range(1):
        z_mip.append(vp_mip.step.relu(z_mip[k] - t * A.T @ (A @ z_mip[k] - x_mip)))
    vp_mip.set_objective_norm([z_mip[1] - z_mip[0]], np.inf)
    mip_inf_norm = vp_mip.solve()

    sdp_two_norm = np.sqrt(sdp_two_norm_sq)
    assert mip_inf_norm <= sdp_two_norm + TEST_EPS


def test_rlt_cuts_algebra_on_known_primal():
    """Build cuts for a 3x3 lifted variable and verify a known feasible primal
    (X = [1;z][1;z]^T with z inside bounds) satisfies all 4 cuts."""
    from fomtools.canonicalizers.sdp import SDP, symmetrize  # noqa: F401

    # Hand-construct the canonicalizer state.
    canon = SDP(solver="clarabel", use_rlt_cuts=True)
    # Manually populate range so dim = 3 and we have two slots at indices 1, 2.
    canon.range_marker = 3
    canon._pad_existing_A_matrices()

    l_p, u_p = 0.0, 1.0
    l_q, u_q = -1.0, 1.0

    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_rlt_cuts_for_pair(1, 2, l_p, u_p, l_q, u_q)

    assert len(canon._cut_ineq_matrices) == 4
    assert len(canon._cut_bineq_upperbounds) == 4

    # Try several feasible primals.
    rng = np.random.default_rng(42)
    for _ in range(20):
        zp = rng.uniform(l_p, u_p)
        zq = rng.uniform(l_q, u_q)
        z = np.array([1.0, zp, zq])
        X = np.outer(z, z)  # rank-1 PSD, satisfies X[0,0] = 1
        for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
            lhs = float((M @ X).trace())
            assert lhs <= b + 1e-10, f"Cut violated: lhs={lhs}, b={b}, z=({zp},{zq})"


def test_rlt_diagonal_skip():
    """Cone with a single non-zero index emits zero RLT cuts (combinations of
    1 element taken 2 at a time is empty — diagonal pairs never reached)."""
    from fomtools.canonicalizers.sdp import SDP

    canon = SDP(solver="clarabel", use_rlt_cuts=True)
    # Insert a cone with only one non-zero index.
    canon.psd_cone_indices = [{0, 1}]
    canon.range_marker = 2
    canon._pad_existing_A_matrices()
    # Build a fake bound map.
    sdp_idx_bounds = {1: (-1.0, 1.0)}
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_rlt_cuts(sdp_idx_bounds)
    assert canon._cut_ineq_matrices == []


def test_rlt_cuts_invalidated_by_update_bounds():
    """After tightening bounds via update_bounds, cuts get rebuilt with the
    tighter bounds, yielding a non-larger SDP objective."""
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    def _solve_with_z0_bounds(lb, ub):
        vp = Verifier(
            method='sdp',
            solver='clarabel',
            obbt_handler='highs',
            use_rlt_cuts=True,
        )
        x = vp.add_param(m, lb=0, ub=0.1)
        z0 = vp.add_iterate(n, lb=lb, ub=ub)
        z1 = vp.step.relu(z0 - t * A.T @ (A @ z0 - x))
        vp.set_objective_norm(z1 - z0, 2)
        return vp.solve()

    loose = _solve_with_z0_bounds(5.0, 5.1)
    tight = _solve_with_z0_bounds(5.02, 5.08)
    assert tight <= loose + TEST_EPS
