"""
Tests for SDP RLT and lifted triangle cuts.

The cuts are valid inequalities from §4.5 of arXiv:2403.03331 and tighten
the SDP relaxation while preserving the MIP lower-bound property.

This file is built up phase-by-phase alongside the implementation in
fomtools/canonicalizers/sdp.py. Each phase adds a new section.
"""

import warnings

import numpy as np
import pytest

from fomtools.verifier import Verifier
from tests.conftest import SDP_SOLVERS, TEST_EPS


def _build_nnls_sdp(
    K,
    sdp_solver,
    rlt_cuts=False,
    triangle_cuts=False,
    relu_envelope_cuts=False,
    rlt_all_pairs=False,
    full_psd_cone=False,
    cross_cut=False,
    subgrad_mono_cut=False,
):
    """Build a small NNLS SDP verifier matching tests/test_nnls_sdp_mip.py.

    Returns the solved SDP two-norm-squared objective value.
    """
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver="highs",
        rlt_cuts=rlt_cuts,
        triangle_cuts=triangle_cuts,
        relu_envelope_cuts=relu_envelope_cuts,
        rlt_all_pairs=rlt_all_pairs,
        full_psd_cone=full_psd_cone,
        cross_cut=cross_cut,
        subgrad_mono_cut=subgrad_mono_cut,
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
        K=1, sdp_solver=sdp_solver, rlt_cuts=False, triangle_cuts=False
    )
    assert abs(baseline - with_flags_off) <= TEST_EPS


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_flags_stored_on_canonicalizer(sdp_solver):
    """The flags reach the SDP canonicalizer instance."""
    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver=None,
        rlt_cuts=True,
        triangle_cuts=True,
    )
    assert vp.canonicalizer.rlt_cuts is True
    assert vp.canonicalizer.triangle_cuts is True

    vp2 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    assert vp2.canonicalizer.rlt_cuts is False
    assert vp2.canonicalizer.triangle_cuts is False


@pytest.mark.parametrize("flag", ["rlt_cuts", "triangle_cuts"])
def test_cut_flags_warn_for_non_sdp_methods(flag):
    """Passing a cut flag with method != 'sdp' should emit a UserWarning."""
    kwargs = {flag: True}
    with pytest.warns(UserWarning, match=flag):
        Verifier(method="mip", solver="highs", obbt_solver=None, **kwargs)


def test_cut_flags_no_warning_when_off_for_non_sdp():
    """When both flags are off (default), no warning is emitted for any method."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Verifier(method="mip", solver="highs", obbt_solver=None)


def test_cut_flags_no_warning_for_sdp():
    """Passing flags with method='sdp' is the normal path, no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Verifier(
            method="sdp",
            solver="clarabel",
            obbt_solver=None,
            rlt_cuts=True,
            triangle_cuts=True,
        )


# ============================================================
# Phase 1: cut-rebuild infrastructure (no cuts yet)
# ============================================================


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_cuts_dirty_triggers(sdp_solver):
    """add_var, canonicalize_relu, add_constraint, update_bounds all mark dirty."""
    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver=None,
        rlt_cuts=True,
    )
    canon = vp.canonicalizer

    # Fresh canonicalizer: dirty is True (rebuild needed on first solve).
    assert canon._cuts_dirty is True

    # add_var marks dirty.
    canon._cuts_dirty = False
    vp.add_param(2, lb=0, ub=1)
    assert canon._cuts_dirty is True

    # canonicalize_relu marks dirty + registers the step.
    canon._cuts_dirty = False
    z0 = vp.add_iterate(2, lb=-1, ub=1)
    vp.step.relu(z0)
    assert canon._cuts_dirty is True
    assert len(canon._relu_steps) >= 1

    # update_bounds marks dirty.
    canon._cuts_dirty = False
    canon.update_bounds(z0, np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
    assert canon._cuts_dirty is True


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rebuild_clears_dirty_after_solve(sdp_solver):
    """After solve, the copy that ran has _cuts_dirty cleared."""
    vp, _ = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, rlt_cuts=True, triangle_cuts=True)
    assert vp.canonicalizer._solved_copy is not None
    assert vp.canonicalizer._solved_copy._cuts_dirty is False


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_copy_resets_dirty(sdp_solver):
    """copy() should reset _cuts_dirty so the copy rebuilds on first solve."""
    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver=None,
        rlt_cuts=True,
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
    vp, _ = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, rlt_cuts=True)
    canon = _solved_canon(vp)
    assert len(canon._cut_ineq_matrices) == _expected_rlt_cut_count(canon)
    assert len(canon._cut_bineq_upperbounds) == len(canon._cut_ineq_matrices)


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cut_matrices_symmetric(sdp_solver):
    """Each RLT cut matrix is symmetric."""
    vp, _ = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, rlt_cuts=True)
    canon = _solved_canon(vp)
    for M in canon._cut_ineq_matrices:
        diff = M - M.T
        # In csc form, eliminate explicit zeros before counting nnz.
        diff.eliminate_zeros()
        assert diff.nnz == 0


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cuts_tighten_bound(sdp_solver):
    """SDP objective with RLT cuts ≤ baseline (we maximize, so tighter relaxation
    gives smaller upper bound)."""
    _, baseline = _build_nnls_sdp(K=1, sdp_solver=sdp_solver)
    _, with_rlt = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, rlt_cuts=True)
    assert with_rlt <= baseline + TEST_EPS


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_rlt_cuts_preserve_mip_lower_bound(sdp_solver):
    """MIP infinity-norm bound ≤ sqrt(SDP two-norm-sq with RLT) — the same
    relationship verified in test_nnls_sdp_mip.py, now with RLT cuts on."""
    _, sdp_two_norm_sq = _build_nnls_sdp(K=1, sdp_solver=sdp_solver, rlt_cuts=True)

    # Build the matching MIP and solve.
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    vp_mip = Verifier(method="mip", solver="highs", obbt_solver="highs")
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
    canon = SDP(solver="clarabel", rlt_cuts=True)
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
            output = float((M @ X).trace())
            assert output <= b + 1e-10, f"Cut violated: output={output}, b={b}, z=({zp},{zq})"


def test_rlt_diagonal_skip():
    """Cone with a single non-zero index emits zero RLT cuts (combinations of
    1 element taken 2 at a time is empty — diagonal pairs never reached)."""
    from fomtools.canonicalizers.sdp import SDP

    canon = SDP(solver="clarabel", rlt_cuts=True)
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
            method="sdp",
            solver="clarabel",
            obbt_solver="highs",
            rlt_cuts=True,
        )
        x = vp.add_param(m, lb=0, ub=0.1)
        z0 = vp.add_iterate(n, lb=lb, ub=ub)
        z1 = vp.step.relu(z0 - t * A.T @ (A @ z0 - x))
        vp.set_objective_norm(z1 - z0, 2)
        return vp.solve()

    loose = _solve_with_z0_bounds(5.0, 5.1)
    tight = _solve_with_z0_bounds(5.02, 5.08)
    assert tight <= loose + TEST_EPS


# ============================================================
# Phase 3: Lifted triangle cuts
# ============================================================


def _build_mixed_relu_problem(triangle_cuts=False, rlt_cuts=False, relu_envelope_cuts=False):
    """A 1D ReLU with z ∈ [-1, 1] so the output is genuinely mixed.

    Returns (verifier, solved_value).
    """
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        triangle_cuts=triangle_cuts,
        rlt_cuts=rlt_cuts,
        relu_envelope_cuts=relu_envelope_cuts,
    )
    z0 = vp.add_iterate(1, lb=-1.0, ub=1.0)
    z1 = vp.step.relu(z0)
    vp.set_objective_norm(z1 - z0, 2)
    return vp, vp.solve()


def test_triangle_cuts_skipped_when_no_mixed_rows():
    """If a ReLU step has only non-mixed rows, no triangle cuts are emitted."""
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        triangle_cuts=True,
    )
    # All-positive input range → relu is identity → step.lb[0] = 1 > 0.
    z0 = vp.add_iterate(2, lb=1.0, ub=2.0)
    _z1 = vp.step.relu(z0)
    vp.set_objective_norm(_z1 - z0, 2)
    vp.solve()
    canon = _solved_canon(vp)
    # No triangle cuts since no row is mixed.
    assert len(canon._cut_ineq_matrices) == 0


def test_triangle_cuts_emitted_when_mixed():
    """A genuinely mixed ReLU step (lb<0<ub) emits the right number of cuts."""
    vp, _ = _build_mixed_relu_problem(triangle_cuts=True)
    canon = _solved_canon(vp)
    # 1D mixed step → 1 mixed row, 1 finite-ub col → exactly 1 triangle cut.
    assert len(canon._cut_ineq_matrices) == 1
    assert len(canon._cut_bineq_upperbounds) == 1


def test_triangle_cut_count_2d():
    """For d-dim ReLU with k mixed rows and all-finite ub, expect k*d cuts."""
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        triangle_cuts=True,
    )
    z0 = vp.add_iterate(2, lb=-1.0, ub=1.0)  # both dims mixed
    z1 = vp.step.relu(z0)
    vp.set_objective_norm(z1 - z0, 2)
    vp.solve()
    canon = _solved_canon(vp)
    # 2 mixed rows × 2 cols → 4 cuts.
    assert len(canon._cut_ineq_matrices) == 4


def test_triangle_cuts_tighten_bound():
    """SDP with triangle cuts ≤ baseline (maximizing, tighter relaxation)."""
    _, baseline = _build_mixed_relu_problem()
    _, with_tri = _build_mixed_relu_problem(triangle_cuts=True)
    assert with_tri <= baseline + TEST_EPS


def test_triangle_cuts_preserve_mip_lower_bound():
    """MIP infinity-norm ≤ sqrt(SDP two-norm-sq with triangle cuts)."""
    _, sdp_two_norm_sq = _build_nnls_sdp(K=1, sdp_solver="clarabel", triangle_cuts=True)
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    vp_mip = Verifier(method="mip", solver="highs", obbt_solver="highs")
    x = vp_mip.add_param(m, lb=0, ub=0.1)
    z = [vp_mip.add_iterate(n, lb=5, ub=5.1)]
    z.append(vp_mip.step.relu(z[0] - t * A.T @ (A @ z[0] - x)))
    vp_mip.set_objective_norm([z[1] - z[0]], np.inf)
    mip_inf_norm = vp_mip.solve()
    assert mip_inf_norm <= np.sqrt(sdp_two_norm_sq) + TEST_EPS


def test_triangle_cut_algebra_on_known_primal():
    """Build a triangle cut by hand on a 1D ReLU and verify it holds on a
    rank-1 lift X = [1; z; y][1; z; y]^T for many feasible (z, y)."""
    from fomtools.canonicalizers.sdp import SDP
    from fomtools.steps import Relu

    # Build the canonicalizer state directly.
    canon = SDP(solver="clarabel", triangle_cuts=True)
    # We need a Relu step whose arg is a leaf variable z with bounds [-1, 1].
    # Construct the leaf and register it.
    from fomtools.linearexpression import LinearExpression

    z = LinearExpression.from_int(1, lb=np.array([-1.0]), ub=np.array([1.0]))
    canon.add_var(z)
    step = Relu(arg=z)
    # step.update_bounds with the same bounds (since arg is leaf z).
    step.update_bounds(z.lb, z.ub)
    canon.add_var(step.output)
    canon.canonicalize_relu(step)

    # Inspect SDP indexing
    z_start, _ = canon.var_idx_map[z]
    y_start, _ = canon.var_idx_map[step.output]

    # Force a triangle-only rebuild for inspection.
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_triangle_cuts()
    assert len(canon._cut_ineq_matrices) == 1

    M = canon._cut_ineq_matrices[0]
    b = canon._cut_bineq_upperbounds[0]

    n = canon.get_sdp_var_dim()
    rng = np.random.default_rng(0)
    for _ in range(50):
        z_val = rng.uniform(-1.0, 1.0)
        y_val = max(0.0, z_val)
        v = np.zeros(n)
        v[0] = 1.0
        v[z_start] = z_val
        v[y_start] = y_val
        X = np.outer(v, v)
        output = float((M @ X).trace())
        assert output <= b + 1e-9, (
            f"Triangle cut violated at z={z_val}, y={y_val}: output={output} > b={b}"
        )


def test_triangle_cut_matrices_symmetric():
    """Each triangle cut matrix is symmetric."""
    vp, _ = _build_mixed_relu_problem(triangle_cuts=True)
    canon = _solved_canon(vp)
    for M in canon._cut_ineq_matrices:
        diff = M - M.T
        diff.eliminate_zeros()
        assert diff.nnz == 0


def test_triangle_cuts_use_fresh_leaf_bounds():
    """Triangle cuts read bounds from arg.bound_propagation() at rebuild time,
    so leaf tightening after canonicalize_relu changes the emitted cut.

    This is the parity property with RLT cuts: both families must pick up
    leaf tightening on the next solve.
    """
    from fomtools.canonicalizers.sdp import SDP
    from fomtools.linearexpression import LinearExpression
    from fomtools.steps import Relu

    canon = SDP(solver="clarabel", triangle_cuts=True)
    z = LinearExpression.from_int(1, lb=np.array([-2.0]), ub=np.array([2.0]))
    canon.add_var(z)
    step = Relu(arg=z)
    step.update_bounds(z.lb, z.ub)
    canon.add_var(step.output)
    canon.canonicalize_relu(step)

    # First rebuild: chord based on arg ∈ [-2, 2].
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_triangle_cuts()
    M_loose = canon._cut_ineq_matrices[0].toarray()

    # Tighten the leaf's bounds in place.
    z.update_bounds(np.array([-1.0]), np.array([1.0]))

    # Second rebuild: chord should now be based on arg ∈ [-1, 1].
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_triangle_cuts()
    M_tight = canon._cut_ineq_matrices[0].toarray()

    # The constant term at X[0,0] differs between the chords:
    # loose: E_ii=0.5, c_i=1.0, b_i=b_j=0, ub_j=2  =>  const = E_ii*b_i*ub_j
    #        + c_i*ub_j - ... = 0 + 1*2 - 0 - 0 = 2.0
    # tight: E_ii=0.5, c_i=0.5, ub_j=1            =>  const = 0 + 0.5*1 - 0 - 0
    #        = 0.5
    # symmetrize halves nothing on the diagonal, so [0,0] entries are exactly
    # these values divided by 2 (since the cut is stored as -symmetrize(N)).
    assert not np.allclose(M_loose, M_tight), (
        "Triangle cut matrix should change when leaf bounds tighten"
    )


# ============================================================
# Phase 4: Integration tests across cut combinations
# ============================================================

# Cartesian product of the two flags. (False, False) is the baseline.
FLAG_COMBOS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


def _build_ista_sdp(
    K,
    sdp_solver,
    rlt_cuts=False,
    triangle_cuts=False,
    relu_envelope_cuts=False,
    rlt_all_pairs=False,
    full_psd_cone=False,
    cross_cut=False,
    subgrad_mono_cut=False,
):
    """ISTA SDP setup mirroring tests/test_ista_sdp_mip.py."""
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=m)
    lambd_max = np.max(np.real(np.linalg.eigvals(A.T @ A)))
    t = 1 / lambd_max
    lambd = 0.5
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver="highs",
        rlt_cuts=rlt_cuts,
        triangle_cuts=triangle_cuts,
        relu_envelope_cuts=relu_envelope_cuts,
        rlt_all_pairs=rlt_all_pairs,
        full_psd_cone=full_psd_cone,
        cross_cut=cross_cut,
        subgrad_mono_cut=subgrad_mono_cut,
    )
    b_param = vp.add_param(m, lb=b - 0.1, ub=b + 0.1)
    z = [vp.add_iterate(n, lb=1, ub=1.1)]
    for k in range(K):
        z.append(vp.step.soft_threshold(At @ z[k] + Bt @ b_param, lambd=lambda_t))
    vp.set_objective_norm(z[K] - z[K - 1], 2)
    return vp, vp.solve()


def _build_boxpgd_sdp(
    K,
    sdp_solver,
    rlt_cuts=False,
    triangle_cuts=False,
    relu_envelope_cuts=False,
    rlt_all_pairs=False,
    full_psd_cone=False,
    cross_cut=False,
    subgrad_mono_cut=False,
):
    """Box PGD SDP setup mirroring tests/test_boxpgd_sdp_mip.py."""
    n = 2
    np.random.seed(0)
    P = np.eye(n)
    q = np.random.normal(size=n)
    l = np.array([-1.0, -0.5])
    u = np.array([0.5, 1.0])
    t = 0.5

    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver="highs",
        rlt_cuts=rlt_cuts,
        triangle_cuts=triangle_cuts,
        relu_envelope_cuts=relu_envelope_cuts,
        rlt_all_pairs=rlt_all_pairs,
        full_psd_cone=full_psd_cone,
        cross_cut=cross_cut,
        subgrad_mono_cut=subgrad_mono_cut,
    )
    q_param = vp.add_param(n, lb=q - 0.1, ub=q + 0.1)
    z = [vp.add_iterate(n, lb=0, ub=0.1)]
    for k in range(K):
        grad_step = z[k] - t * (P @ z[k] + q_param)
        z.append(vp.step.satlin(grad_step, l=l, u=u))
    vp.set_objective_norm(z[K] - z[K - 1], 2)
    return vp, vp.solve()


def _mip_inf_norm_nnls(K):
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01
    vp = Verifier(method="mip", solver="highs", obbt_solver="highs")
    x = vp.add_param(m, lb=0, ub=0.1)
    z = [vp.add_iterate(n, lb=5, ub=5.1)]
    for k in range(K):
        z.append(vp.step.relu(z[k] - t * A.T @ (A @ z[k] - x)))
    vp.set_objective_norm([z[K] - z[K - 1]], np.inf)
    return vp.solve()


def _mip_inf_norm_ista(K):
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=m)
    lambd_max = np.max(np.real(np.linalg.eigvals(A.T @ A)))
    t = 1 / lambd_max
    lambd = 0.5
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    vp = Verifier(method="mip", solver="highs", obbt_solver="highs")
    b_param = vp.add_param(m, lb=b - 0.1, ub=b + 0.1)
    z = [vp.add_iterate(n, lb=1, ub=1.1)]
    for k in range(K):
        z.append(vp.step.soft_threshold(At @ z[k] + Bt @ b_param, lambd=lambda_t))
    vp.set_objective_norm([z[K] - z[K - 1]], np.inf)
    return vp.solve()


def _mip_inf_norm_boxpgd(K):
    n = 2
    np.random.seed(0)
    P = np.eye(n)
    q = np.random.normal(size=n)
    l = np.array([-1.0, -0.5])
    u = np.array([0.5, 1.0])
    t = 0.5
    vp = Verifier(method="mip", solver="highs", obbt_solver="highs")
    q_param = vp.add_param(n, lb=q - 0.1, ub=q + 0.1)
    z = [vp.add_iterate(n, lb=0, ub=0.1)]
    for k in range(K):
        grad_step = z[k] - t * (P @ z[k] + q_param)
        z.append(vp.step.satlin(grad_step, l=l, u=u))
    vp.set_objective_norm([z[K] - z[K - 1]], np.inf)
    return vp.solve()


@pytest.mark.parametrize("use_rlt,use_tri", FLAG_COMBOS)
@pytest.mark.parametrize("K", [1, 2])
def test_nnls_mip_bound_preserved_all_combos(use_rlt, use_tri, K):
    """mip_inf_norm ≤ sqrt(sdp_two_norm_sq) for every flag combination."""
    _, sdp_sq = _build_nnls_sdp(K=K, sdp_solver="clarabel", rlt_cuts=use_rlt, triangle_cuts=use_tri)
    mip_inf = _mip_inf_norm_nnls(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, rlt={use_rlt}, tri={use_tri}: "
        f"MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


@pytest.mark.parametrize("use_rlt,use_tri", FLAG_COMBOS)
@pytest.mark.parametrize("K", [1, 2])
def test_ista_mip_bound_preserved_all_combos(use_rlt, use_tri, K):
    _, sdp_sq = _build_ista_sdp(K=K, sdp_solver="clarabel", rlt_cuts=use_rlt, triangle_cuts=use_tri)
    mip_inf = _mip_inf_norm_ista(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, rlt={use_rlt}, tri={use_tri}: "
        f"MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


@pytest.mark.parametrize("use_rlt,use_tri", FLAG_COMBOS)
@pytest.mark.parametrize("K", [1, 2])
def test_boxpgd_mip_bound_preserved_all_combos(use_rlt, use_tri, K):
    _, sdp_sq = _build_boxpgd_sdp(
        K=K, sdp_solver="clarabel", rlt_cuts=use_rlt, triangle_cuts=use_tri
    )
    mip_inf = _mip_inf_norm_boxpgd(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, rlt={use_rlt}, tri={use_tri}: "
        f"MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


def test_both_flags_compose_no_worse_than_either():
    """SDP with both flags on ≤ min(SDP with RLT only, SDP with triangle only).

    Both cut families are valid inequalities, so adding both can only tighten
    the relaxation further (or leave it unchanged).
    """
    _, baseline = _build_nnls_sdp(K=1, sdp_solver="clarabel")
    _, with_rlt = _build_nnls_sdp(K=1, sdp_solver="clarabel", rlt_cuts=True)
    _, with_tri = _build_nnls_sdp(K=1, sdp_solver="clarabel", triangle_cuts=True)
    _, with_both = _build_nnls_sdp(K=1, sdp_solver="clarabel", rlt_cuts=True, triangle_cuts=True)
    assert with_both <= with_rlt + TEST_EPS
    assert with_both <= with_tri + TEST_EPS
    # Sanity: both ≤ baseline (already implied transitively).
    assert with_both <= baseline + TEST_EPS


# ============================================================
# Phase 5: ReLU linear envelope cuts (relu_envelope_cuts)
# ============================================================
#
# These are the convex-hull lines on the FIRST-MOMENT (vector) variables that
# the lifted ReLU equality + PSD does not imply:
#   - lower envelope  y_i >= a_i^T x + b_i            (every projected row)
#   - upper chord     y_i <= E_ii (a_i^T x + b_i - l_i)  (mixed rows only)
# The y_i >= 0 line is omitted because the y-leaf box bound + PSD imply it.


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_envelope_flag_stored_on_canonicalizer(sdp_solver):
    """relu_envelope_cuts reaches the SDP canonicalizer instance."""
    vp = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None, relu_envelope_cuts=True)
    assert vp.canonicalizer.relu_envelope_cuts is True
    vp2 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    assert vp2.canonicalizer.relu_envelope_cuts is False


@pytest.mark.parametrize("flag", ["relu_envelope_cuts", "rlt_all_pairs", "full_psd_cone"])
def test_new_cut_flags_warn_for_non_sdp_methods(flag):
    """The new SDP-only flags emit a UserWarning when method != 'sdp'."""
    with pytest.warns(UserWarning, match=flag):
        Verifier(method="mip", solver="highs", obbt_solver=None, **{flag: True})


def test_envelope_cuts_emitted_when_mixed():
    """A 1D mixed ReLU emits 1 lower-envelope + 1 upper-chord = 2 envelope cuts."""
    vp, _ = _build_mixed_relu_problem(relu_envelope_cuts=True)
    canon = _solved_canon(vp)
    assert len(canon._cut_ineq_matrices) == 2
    assert len(canon._cut_bineq_upperbounds) == 2


def test_envelope_cuts_passthrough_no_chord():
    """All-positive ReLU rows (passthrough) emit only the lower envelope, no chord."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, relu_envelope_cuts=True)
    z0 = vp.add_iterate(2, lb=1.0, ub=2.0)  # both rows passthrough (lb >= 0)
    z1 = vp.step.relu(z0)
    vp.set_objective_norm(z1 - z0, 2)
    vp.solve()
    canon = _solved_canon(vp)
    # 2 projected rows -> 2 lower-envelope cuts, 0 chords.
    assert len(canon._cut_ineq_matrices) == 2


def test_envelope_cut_matrices_symmetric():
    """Each envelope cut matrix is symmetric."""
    vp, _ = _build_mixed_relu_problem(relu_envelope_cuts=True)
    canon = _solved_canon(vp)
    for M in canon._cut_ineq_matrices:
        diff = M - M.T
        diff.eliminate_zeros()
        assert diff.nnz == 0


def _build_1d_relu_canon(lb, ub, relu_envelope_cuts=True):
    """Hand-build an SDP canonicalizer state for a single 1D ReLU y = max(0, z)
    with z in [lb, ub]. Returns (canon, z_leaf, relu_step)."""
    from fomtools.canonicalizers.sdp import SDP
    from fomtools.linearexpression import LinearExpression
    from fomtools.steps import Relu

    canon = SDP(solver="clarabel", relu_envelope_cuts=relu_envelope_cuts)
    z = LinearExpression.from_int(1, lb=np.array([float(lb)]), ub=np.array([float(ub)]))
    canon.add_var(z)
    step = Relu(arg=z)
    step.update_bounds(z.lb, z.ub)
    canon.add_var(step.output)
    canon.canonicalize_relu(step)
    return canon, z, step


def test_envelope_cut_algebra_on_known_primal():
    """Lower envelope and chord hold on a rank-1 lift X = [1;z;y][1;z;y]^T for
    many feasible (z, y = max(0, z))."""
    canon, z, step = _build_1d_relu_canon(-1.0, 1.0)
    z_start, _ = canon.var_idx_map[z]
    y_start, _ = canon.var_idx_map[step.output]

    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_relu_envelope_cuts()
    assert len(canon._cut_ineq_matrices) == 2

    n = canon.get_sdp_var_dim()
    rng = np.random.default_rng(0)
    for _ in range(50):
        z_val = rng.uniform(-1.0, 1.0)
        y_val = max(0.0, z_val)
        v = np.zeros(n)
        v[0] = 1.0
        v[z_start] = z_val
        v[y_start] = y_val
        X = np.outer(v, v)
        for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
            output = float((M @ X).trace())
            assert output <= b + 1e-9, (
                f"Envelope cut violated at z={z_val}, y={y_val}: {output} > {b}"
            )


def test_envelope_lower_cut_not_redundant():
    """A relaxed X satisfying the lifted ReLU equality + box + PSD can violate the
    lower envelope y >= input, so the cut genuinely tightens.

    Witness (indices [0, z, y]):  X = [[1, .5, 0], [.5, 1, 0], [0, 0, 0]].
    PSD; X[z,z]=1 (box z∈[-1,1]); X[y,y]=0=X[z,y] (relu eq y^2 = z*y); but
    X[0,y]=0 < X[0,z]=0.5 violates y >= z.
    """
    canon, z, step = _build_1d_relu_canon(-1.0, 1.0)
    z_start, _ = canon.var_idx_map[z]
    y_start, _ = canon.var_idx_map[step.output]

    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_relu_envelope_cuts()
    # cut[0] is the lower envelope (appended before the chord).
    M = canon._cut_ineq_matrices[0]
    b = canon._cut_bineq_upperbounds[0]

    n = canon.get_sdp_var_dim()
    X = np.zeros((n, n))
    X[0, 0] = 1.0
    X[0, z_start] = X[z_start, 0] = 0.5
    X[z_start, z_start] = 1.0  # y entries left at 0
    output = float((M @ X).trace())
    assert output > b + 1e-6, (
        "Lower-envelope cut should be violated by the relaxed witness "
        "(demonstrates it is not implied by the existing constraints)"
    )


def test_envelope_cuts_tighten_bound():
    """SDP with envelope cuts ≤ baseline (maximizing, tighter relaxation)."""
    _, baseline = _build_mixed_relu_problem()
    _, with_env = _build_mixed_relu_problem(relu_envelope_cuts=True)
    assert with_env <= baseline + TEST_EPS


def test_envelope_chord_uses_fresh_leaf_bounds():
    """The upper chord reads arg.bound_propagation() at rebuild time, so leaf
    tightening after canonicalize_relu changes the emitted chord (parity with
    the RLT and triangle cut families)."""
    canon, z, step = _build_1d_relu_canon(-2.0, 2.0)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_relu_envelope_cuts()
    chord_loose = canon._cut_ineq_matrices[1].toarray()

    z.update_bounds(np.array([-1.0]), np.array([1.0]))
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_relu_envelope_cuts()
    chord_tight = canon._cut_ineq_matrices[1].toarray()

    assert not np.allclose(chord_loose, chord_tight), (
        "Envelope chord should change when leaf bounds tighten"
    )


# ============================================================
# Phase 6: full PSD coupling (full_psd_cone)
# ============================================================


def test_full_psd_flag_stored_on_canonicalizer():
    """full_psd_cone reaches the SDP canonicalizer instance."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, full_psd_cone=True)
    assert vp.canonicalizer.full_psd_cone is True


def test_full_psd_unions_cones():
    """After solve, the rebuilt copy has a single PSD cone over all indices."""
    vp, _ = _build_nnls_sdp(K=1, sdp_solver="clarabel", full_psd_cone=True)
    canon = _solved_canon(vp)
    dim = canon.get_sdp_var_dim()
    assert canon.psd_cone_indices == [set(range(dim))]


def test_full_psd_rlt_fires_on_all_pairs():
    """With full_psd_cone + rlt_cuts, RLT fires on every off-diagonal pair
    (4 cuts per pair), not just the per-block cone-mates."""
    from itertools import combinations

    vp, _ = _build_nnls_sdp(K=1, sdp_solver="clarabel", full_psd_cone=True, rlt_cuts=True)
    canon = _solved_canon(vp)
    dim = canon.get_sdp_var_dim()
    n_pairs = len(list(combinations(range(1, dim), 2)))
    # Exact equality relies on RLT being the *only* cut source: envelope/triangle are
    # off, and _build_nnls_sdp makes no add_constraint calls, so _ineq_constraints is
    # empty (the box bounds from add_var do not populate it). Any inequality constraint
    # would add box/square/product cuts on top of the 4-per-pair RLT count.
    assert len(canon._cut_ineq_matrices) == 4 * n_pairs


def test_full_psd_tighten_bound():
    """full_psd_cone (bigger PSD block) ≤ baseline."""
    _, baseline = _build_nnls_sdp(K=1, sdp_solver="clarabel")
    _, with_full = _build_nnls_sdp(K=1, sdp_solver="clarabel", full_psd_cone=True)
    assert with_full <= baseline + TEST_EPS


def test_full_psd_does_not_corrupt_original_cones():
    """The cone union happens on the solved copy; the original's cone list is
    unchanged across a solve."""
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver="highs", full_psd_cone=True)
    x = vp.add_param(m, lb=0, ub=0.1)
    z = [vp.add_iterate(n, lb=5, ub=5.1)]
    z.append(vp.step.relu(z[0] - t * A.T @ (A @ z[0] - x)))
    vp.set_objective_norm(z[1] - z[0], 2)

    before = [set(s) for s in vp.canonicalizer.psd_cone_indices]
    vp.solve()
    after = [set(s) for s in vp.canonicalizer.psd_cone_indices]
    assert before == after


# ============================================================
# Phase 7: integration of the new cut families (MIP-bound preserved)
# ============================================================

# The MIP infinity-norm lower-bounds the SDP two-norm. Every new (combination
# of) cut family must preserve this, i.e. never cut off a true feasible point.
NEW_FLAG_KWARGS = [
    {"relu_envelope_cuts": True},
    {"rlt_all_pairs": True},
    {"full_psd_cone": True},
    {"full_psd_cone": True, "rlt_cuts": True},
    {
        "relu_envelope_cuts": True,
        "triangle_cuts": True,
        "rlt_cuts": True,
        "full_psd_cone": True,
    },
]


@pytest.mark.parametrize("kwargs", NEW_FLAG_KWARGS)
@pytest.mark.parametrize("K", [1, 2])
def test_nnls_mip_bound_preserved_new_flags(kwargs, K):
    _, sdp_sq = _build_nnls_sdp(K=K, sdp_solver="clarabel", **kwargs)
    mip_inf = _mip_inf_norm_nnls(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, {kwargs}: MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


@pytest.mark.parametrize("kwargs", NEW_FLAG_KWARGS)
@pytest.mark.parametrize("K", [1, 2])
def test_ista_mip_bound_preserved_new_flags(kwargs, K):
    _, sdp_sq = _build_ista_sdp(K=K, sdp_solver="clarabel", **kwargs)
    mip_inf = _mip_inf_norm_ista(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, {kwargs}: MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


@pytest.mark.parametrize("kwargs", NEW_FLAG_KWARGS)
@pytest.mark.parametrize("K", [1, 2])
def test_boxpgd_mip_bound_preserved_new_flags(kwargs, K):
    _, sdp_sq = _build_boxpgd_sdp(K=K, sdp_solver="clarabel", **kwargs)
    mip_inf = _mip_inf_norm_boxpgd(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, {kwargs}: MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


def test_new_flags_compose_no_worse_than_either():
    """All cut families together ≤ each individually — every family is a valid
    inequality, so composing them can only tighten the relaxation."""
    _, with_env = _build_nnls_sdp(K=1, sdp_solver="clarabel", relu_envelope_cuts=True)
    _, with_full = _build_nnls_sdp(K=1, sdp_solver="clarabel", full_psd_cone=True, rlt_cuts=True)
    _, with_all = _build_nnls_sdp(
        K=1,
        sdp_solver="clarabel",
        relu_envelope_cuts=True,
        triangle_cuts=True,
        rlt_cuts=True,
        full_psd_cone=True,
    )
    assert with_all <= with_env + TEST_EPS
    assert with_all <= with_full + TEST_EPS


# ============================================================
# Phase 8: all-pairs RLT via pairwise PSD minors (rlt_all_pairs)
# ============================================================
#
# Middle tier between cone-mate-only rlt_cuts and the dense full_psd_cone: RLT
# every pair, coupling each via a cheap {0,p,q} 3x3 PSD minor. Pairs already in a
# larger cone add no new minor (add_psd_cone_indices drops subsumed cones).


def test_rlt_all_pairs_flag_stored_on_canonicalizer():
    """rlt_all_pairs reaches the SDP canonicalizer instance."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, rlt_all_pairs=True)
    assert vp.canonicalizer.rlt_all_pairs is True
    vp2 = Verifier(method="sdp", solver="clarabel", obbt_solver=None)
    assert vp2.canonicalizer.rlt_all_pairs is False


def _two_independent_relus(rlt_cuts=False, rlt_all_pairs=False):
    """Two 1-D ReLUs on independent iterates: by default the two PSD cones
    {0,z0,z1} and {0,w0,w1} are disjoint, so (z, w) cross-pairs are not
    cone-mates. Returns the solved canonicalizer."""
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        rlt_cuts=rlt_cuts,
        rlt_all_pairs=rlt_all_pairs,
    )
    z0 = vp.add_iterate(1, lb=-1.0, ub=1.0)
    z1 = vp.step.relu(z0)
    w0 = vp.add_iterate(1, lb=-1.0, ub=1.0)
    _w1 = vp.step.relu(w0)  # registers a second, disjoint ReLU cone
    vp.set_objective_norm(z1 - z0, 2)
    vp.solve()
    return _solved_canon(vp)


def test_rlt_all_pairs_subsumed_when_single_cone():
    """When one cone already spans all variables (a single 2-D ReLU), enabling
    rlt_all_pairs adds no new cones — every {0,p,q} is subsumed. This is the
    self-cleaning property of add_psd_cone_indices."""

    def _cones(rlt_cuts, rlt_all_pairs):
        vp = Verifier(
            method="sdp",
            solver="clarabel",
            obbt_solver=None,
            rlt_cuts=rlt_cuts,
            rlt_all_pairs=rlt_all_pairs,
        )
        z0 = vp.add_iterate(2, lb=-1.0, ub=1.0)
        z1 = vp.step.relu(z0)
        vp.set_objective_norm(z1 - z0, 2)
        vp.solve()
        return [set(s) for s in _solved_canon(vp).psd_cone_indices]

    cone_mate = _cones(rlt_cuts=True, rlt_all_pairs=False)
    all_pairs = _cones(rlt_cuts=False, rlt_all_pairs=True)
    assert cone_mate == all_pairs  # no extra cones registered


def test_rlt_all_pairs_couples_disjoint_cones():
    """With disjoint cones, rlt_all_pairs registers the cross-pair minors and RLT
    fires on every pair: 4 * C(dim-1, 2) cuts, strictly more than cone-mate-only."""
    from itertools import combinations

    canon_mate = _two_independent_relus(rlt_cuts=True)
    canon_all = _two_independent_relus(rlt_all_pairs=True)
    dim = canon_all.get_sdp_var_dim()
    n_all_pairs = len(list(combinations(range(1, dim), 2)))

    assert len(canon_all._cut_ineq_matrices) == 4 * n_all_pairs
    assert len(canon_all._cut_ineq_matrices) > len(canon_mate._cut_ineq_matrices)
    # Every pair now shares a cone.
    for p, q in combinations(range(1, dim), 2):
        assert any({0, p, q} <= c for c in canon_all.psd_cone_indices)


def test_rlt_all_pairs_matrices_symmetric():
    """Each all-pairs RLT cut matrix is symmetric."""
    canon = _two_independent_relus(rlt_all_pairs=True)
    for M in canon._cut_ineq_matrices:
        diff = M - M.T
        diff.eliminate_zeros()
        assert diff.nnz == 0


def test_rlt_all_pairs_relaxation_hierarchy():
    """baseline ≥ rlt_cuts ≥ rlt_all_pairs ≥ full_psd_cone+rlt_cuts (maximizing,
    so tighter relaxations give smaller values)."""
    _, baseline = _build_nnls_sdp(K=2, sdp_solver="clarabel")
    _, rlt = _build_nnls_sdp(K=2, sdp_solver="clarabel", rlt_cuts=True)
    _, allp = _build_nnls_sdp(K=2, sdp_solver="clarabel", rlt_all_pairs=True)
    _, full = _build_nnls_sdp(K=2, sdp_solver="clarabel", full_psd_cone=True, rlt_cuts=True)
    assert rlt <= baseline + TEST_EPS
    assert allp <= rlt + TEST_EPS
    assert full <= allp + TEST_EPS


def test_rlt_all_pairs_does_not_corrupt_original_cones():
    """rlt_all_pairs adds the {0,p,q} minors on the solved copy; the original
    canonicalizer's cone list is unchanged across a solve (matches the
    full_psd_cone non-corruption guard)."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, rlt_all_pairs=True)
    z0 = vp.add_iterate(1, lb=-1.0, ub=1.0)
    z1 = vp.step.relu(z0)
    w0 = vp.add_iterate(1, lb=-1.0, ub=1.0)
    _w1 = vp.step.relu(w0)  # second, disjoint ReLU cone → cross-pairs need new minors
    vp.set_objective_norm(z1 - z0, 2)

    before = [set(s) for s in vp.canonicalizer.psd_cone_indices]
    vp.solve()
    after = [set(s) for s in vp.canonicalizer.psd_cone_indices]
    assert before == after

    # Non-vacuity: the solve genuinely coupled the disjoint cones on the copy, so the
    # test would also fail if the mutation had leaked back onto the original.
    assert len(_solved_canon(vp).psd_cone_indices) > len(before)


# ============================================================
# Phase 9: PEP cross + subgradient-monotonicity cuts
# ============================================================
#
# Two cross-step operator-interpolation cuts (cross_cut, subgrad_mono_cut):
#   - cross:   z_a r_b <= 0    over ReLU output/residual component pairs
#   - subgrad: <s^i/tau_i - s^j/tau_j, z^i> >= 0  over prox-step pairs
# Both couple DIFFERENT steps (the per-step encoding does not) and self-register
# the minimal small PSD cones they need to bind (no full_psd_cone needed). They
# are valid inequalities, so they must never cut off a true trajectory: the core
# tests verify (a) a known feasible rank-1 lift satisfies every emitted cut and
# (b) the MIP lower bound is preserved across all flag combinations.


def _build_two_relu_canon(cross_cut=False, subgrad_mono_cut=False, lb=-1.0, ub=1.0):
    """Hand-build an SDP canonicalizer with two independent 1-D ReLUs
    y = max(0, z) and v = max(0, w), z, w in [lb, ub].

    Returns (canon, info) where info maps names to SDP start indices.
    """
    from fomtools.canonicalizers.sdp import SDP
    from fomtools.linearexpression import LinearExpression
    from fomtools.steps import Relu

    canon = SDP(solver="clarabel", cross_cut=cross_cut, subgrad_mono_cut=subgrad_mono_cut)

    z = LinearExpression.from_int(1, lb=np.array([float(lb)]), ub=np.array([float(ub)]))
    canon.add_var(z)
    sz = Relu(arg=z)
    sz.update_bounds(z.lb, z.ub)
    canon.add_var(sz.output)
    canon.canonicalize_relu(sz)

    w = LinearExpression.from_int(1, lb=np.array([float(lb)]), ub=np.array([float(ub)]))
    canon.add_var(w)
    sw = Relu(arg=w)
    sw.update_bounds(w.lb, w.ub)
    canon.add_var(sw.output)
    canon.canonicalize_relu(sw)

    info = {
        "z": canon.var_idx_map[z][0],
        "w": canon.var_idx_map[w][0],
        "y": canon.var_idx_map[sz.output_leaves[0]][0],
        "v": canon.var_idx_map[sw.output_leaves[0]][0],
    }
    return canon, info


def _rank1_relu_lift(canon, info, z_val, w_val):
    """Build the rank-1 lift X = u u^T for the true trajectory point
    (z=z_val, w=w_val, y=relu(z_val), v=relu(w_val))."""
    n = canon.get_sdp_var_dim()
    u = np.zeros(n)
    u[0] = 1.0
    u[info["z"]] = z_val
    u[info["w"]] = w_val
    u[info["y"]] = max(0.0, z_val)
    u[info["v"]] = max(0.0, w_val)
    return np.outer(u, u)


# ---- flags + plumbing ---------------------------------------------------- #


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_pep_flags_stored_on_canonicalizer(sdp_solver):
    """cross_cut / subgrad_mono_cut reach the SDP canonicalizer instance."""
    vp = Verifier(
        method="sdp",
        solver=sdp_solver,
        obbt_solver=None,
        cross_cut=True,
        subgrad_mono_cut=True,
    )
    assert vp.canonicalizer.cross_cut is True
    assert vp.canonicalizer.subgrad_mono_cut is True

    vp2 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    assert vp2.canonicalizer.cross_cut is False
    assert vp2.canonicalizer.subgrad_mono_cut is False


@pytest.mark.parametrize("flag", ["cross_cut", "subgrad_mono_cut"])
def test_pep_flags_warn_for_non_sdp_methods(flag):
    """The PEP cut flags emit a UserWarning when method != 'sdp'."""
    with pytest.warns(UserWarning, match=flag):
        Verifier(method="mip", solver="highs", obbt_solver=None, **{flag: True})


@pytest.mark.parametrize("sdp_solver", SDP_SOLVERS)
def test_pep_flags_default_off_matches_baseline(sdp_solver):
    """With both PEP flags explicitly off, the objective matches the baseline."""
    _, baseline = _build_ista_sdp(K=2, sdp_solver=sdp_solver)
    _, off = _build_ista_sdp(K=2, sdp_solver=sdp_solver, cross_cut=False, subgrad_mono_cut=False)
    assert abs(baseline - off) <= TEST_EPS


def test_copy_preserves_pep_flags_and_prox_registry():
    """copy() carries the PEP flags and a (shallow) copy of the prox registry."""
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        cross_cut=True,
        subgrad_mono_cut=True,
    )
    z = vp.add_iterate(2, lb=-1, ub=1)
    _ = vp.step.relu(z)
    canon = vp.canonicalizer
    assert len(canon._prox_steps) == 1  # the ReLU registered one prox step

    copy = canon.copy()
    assert copy.cross_cut is True
    assert copy.subgrad_mono_cut is True
    assert len(copy._prox_steps) == len(canon._prox_steps)
    # Shallow copy: a new list, sharing the (immutable) entries.
    assert copy._prox_steps is not canon._prox_steps


# ---- prox registration semantics ----------------------------------------- #


def test_relu_registers_unit_tau_prox():
    """A standalone ReLU registers exactly one prox step with tau = 1."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, subgrad_mono_cut=True)
    z = vp.add_iterate(2, lb=-1, ub=1)
    _ = vp.step.relu(z)
    canon = vp.canonicalizer
    assert len(canon._prox_steps) == 1
    assert canon._prox_steps[0]["tau"] == 1.0


def test_softthreshold_registers_single_composite_prox():
    """A soft-threshold registers ONE composite prox (tau = lambda), not the two
    internal decomposition ReLUs — though both internal ReLUs still join
    _relu_steps (the cross cut wants every ReLU node)."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, subgrad_mono_cut=True)
    z = vp.add_iterate(2, lb=-1, ub=1)
    _ = vp.step.soft_threshold(z, lambd=0.3)
    canon = vp.canonicalizer
    # Two internal ReLUs registered as ReLU nodes ...
    assert len(canon._relu_steps) == 2
    # ... but only the single composite prox is registered for the subgrad cut.
    assert len(canon._prox_steps) == 1
    assert canon._prox_steps[0]["tau"] == pytest.approx(0.3)


# ---- cross cut: emission + validity --------------------------------------- #


def test_cross_cut_count_two_relus():
    """Two 1-D ReLUs -> 2 ordered cross pairs (y,v) and (v,y) -> 2 cuts; the
    same-component diagonals (y,y), (v,v) are skipped."""
    canon, _ = _build_two_relu_canon(cross_cut=True)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_cross_cuts()
    assert len(canon._cut_ineq_matrices) == 2
    assert len(canon._cut_bineq_upperbounds) == 2


def test_cross_cut_single_1d_relu_skips_diagonal():
    """A lone 1-D ReLU has only the (a, a) diagonal pair, which is the
    complementarity identity and is skipped -> 0 cross cuts."""
    canon, _z, _step = _build_1d_relu_canon(-1.0, 1.0, relu_envelope_cuts=False)
    canon.cross_cut = True
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_cross_cuts()
    assert canon._cut_ineq_matrices == []


def test_cross_cut_count_single_2d_relu():
    """A single 2-D ReLU has two output components -> 2 off-diagonal cross cuts."""
    vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, cross_cut=True)
    z = vp.add_iterate(2, lb=-1.0, ub=1.0)
    z1 = vp.step.relu(z)
    vp.set_objective_norm(z1 - z, 2)
    vp.solve()
    canon = _solved_canon(vp)
    assert len(canon._cut_ineq_matrices) == 2


def test_cross_cut_matrices_symmetric():
    """Each cross cut matrix is symmetric."""
    canon, _ = _build_two_relu_canon(cross_cut=True)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_cross_cuts()
    for M in canon._cut_ineq_matrices:
        diff = M - M.T
        diff.eliminate_zeros()
        assert diff.nnz == 0


def test_cross_cut_algebra_on_known_primal():
    """Every emitted cross cut holds on the rank-1 lift of a true trajectory
    point, for many feasible (z, w) — i.e. the cut never removes a true point."""
    canon, info = _build_two_relu_canon(cross_cut=True)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_cross_cuts()
    assert len(canon._cut_ineq_matrices) == 2

    rng = np.random.default_rng(0)
    for _ in range(50):
        z_val = rng.uniform(-1.0, 1.0)
        w_val = rng.uniform(-1.0, 1.0)
        X = _rank1_relu_lift(canon, info, z_val, w_val)
        for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
            output = float((M @ X).trace())
            assert output <= b + 1e-9, f"Cross cut violated at z={z_val}, w={w_val}: {output} > {b}"


def test_cross_cut_not_redundant():
    """A relaxed X feasible for the per-step encoding can violate a cross cut, so
    the cut genuinely tightens. Witness on two 1-D ReLUs (indices 0,z,y,w,v):
    set z=w=0 (so both relu eqns y^2=z*y, v^2=w*v hold at y=v=0), but couple the
    second moment X[y, w-residual] adversarially: take y=1 (free since X[y,y]=1,
    X[0,y]=1 is allowed by box y in [0,1]) while w's residual r=w-v is negative."""
    canon, info = _build_two_relu_canon(cross_cut=True)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_cross_cuts()

    n = canon.get_sdp_var_dim()
    # Construct a symmetric X that satisfies y>=0, v>=0 first moments and the
    # diagonal relu identities but puts a positive value on X[y, w] with w<0,
    # making z_y * (w - v) > 0 (the cut forbids this).
    X = np.zeros((n, n))
    X[0, 0] = 1.0
    iy, iw, iv = info["y"], info["w"], info["v"]
    # First moments: y = 1 (feasible in [0,1]), w = -1, v = relu(-1) = 0.
    X[0, iy] = X[iy, 0] = 1.0
    X[0, iw] = X[iw, 0] = -1.0
    X[0, iv] = X[iv, 0] = 0.0
    X[iy, iy] = 1.0
    X[iw, iw] = 1.0
    # Adversarial cross moment: y*w = +1 (so y*(w - v) = 1 > 0).
    X[iy, iw] = X[iw, iy] = 1.0

    # The (y, v) pair cut encodes z_y * (w - v) <= 0; find a cut violated here.
    violated = False
    for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
        if float((M @ X).trace()) > b + 1e-6:
            violated = True
            break
    assert violated, "Expected at least one cross cut to be violated by the relaxed witness"


def test_cross_cut_tighten_bound():
    """SDP with cross cut <= baseline (maximizing -> tighter relaxation)."""
    _, baseline = _build_ista_sdp(K=2, sdp_solver="clarabel")
    _, with_cross = _build_ista_sdp(K=2, sdp_solver="clarabel", cross_cut=True)
    assert with_cross <= baseline + TEST_EPS


# ---- subgradient-monotonicity cut: emission + validity -------------------- #


def test_subgrad_cut_count_two_proxes():
    """Two prox steps -> 2 ordered pairs (i,j), (j,i) -> 2 subgradient cuts."""
    canon, _ = _build_two_relu_canon(subgrad_mono_cut=True)
    assert len(canon._prox_steps) == 2
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_subgrad_mono_cuts()
    assert len(canon._cut_ineq_matrices) == 2


def test_subgrad_cut_matrices_symmetric():
    """Each subgradient cut matrix is symmetric."""
    canon, _ = _build_two_relu_canon(subgrad_mono_cut=True)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_subgrad_mono_cuts()
    assert len(canon._cut_ineq_matrices) == 2
    for M in canon._cut_ineq_matrices:
        diff = M - M.T
        diff.eliminate_zeros()
        assert diff.nnz == 0


def test_subgrad_cut_algebra_on_known_primal():
    """Every subgradient cut holds on the rank-1 lift of a true trajectory point
    (for two orthant proxes, where the cut reduces to the cross relation)."""
    canon, info = _build_two_relu_canon(subgrad_mono_cut=True)
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_subgrad_mono_cuts()
    assert len(canon._cut_ineq_matrices) == 2

    rng = np.random.default_rng(1)
    for _ in range(50):
        z_val = rng.uniform(-1.0, 1.0)
        w_val = rng.uniform(-1.0, 1.0)
        X = _rank1_relu_lift(canon, info, z_val, w_val)
        for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
            output = float((M @ X).trace())
            assert output <= b + 1e-9, (
                f"Subgrad cut violated at z={z_val}, w={w_val}: {output} > {b}"
            )


def test_subgrad_cut_tau_guard_skips_tiny_threshold():
    """Prox pairs touching a near-zero tau are skipped (avoids 1/tau blow-up)."""
    canon, _ = _build_two_relu_canon(subgrad_mono_cut=True)
    # Force one prox threshold to ~0; both ordered pairs touch it -> 0 cuts.
    canon._prox_steps[0]["tau"] = 0.0
    canon._cut_ineq_matrices = []
    canon._cut_bineq_upperbounds = []
    canon._build_subgrad_mono_cuts()
    assert canon._cut_ineq_matrices == []


def test_subgrad_cut_tighten_bound():
    """SDP with subgradient cut <= baseline (maximizing -> tighter relaxation)."""
    _, baseline = _build_ista_sdp(K=2, sdp_solver="clarabel")
    _, with_subgrad = _build_ista_sdp(K=2, sdp_solver="clarabel", subgrad_mono_cut=True)
    assert with_subgrad <= baseline + TEST_EPS


# ---- validity across all combinations (MIP lower bound preserved) --------- #
#
# The core "do not tighten too much" guarantee: for every combination of the two
# PEP flags, on every problem family and depth, the exact MIP infinity-norm must
# still lower-bound the SDP two-norm. A bug that cuts off a true trajectory would
# drop the SDP below the MIP and trip these.

PEP_FLAG_COMBOS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


@pytest.mark.parametrize("use_cross,use_subgrad", PEP_FLAG_COMBOS)
@pytest.mark.parametrize("K", [1, 2])
def test_nnls_mip_bound_preserved_pep_combos(use_cross, use_subgrad, K):
    _, sdp_sq = _build_nnls_sdp(
        K=K, sdp_solver="clarabel", cross_cut=use_cross, subgrad_mono_cut=use_subgrad
    )
    mip_inf = _mip_inf_norm_nnls(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, cross={use_cross}, subgrad={use_subgrad}: "
        f"MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


@pytest.mark.parametrize("use_cross,use_subgrad", PEP_FLAG_COMBOS)
@pytest.mark.parametrize("K", [1, 2])
def test_ista_mip_bound_preserved_pep_combos(use_cross, use_subgrad, K):
    _, sdp_sq = _build_ista_sdp(
        K=K, sdp_solver="clarabel", cross_cut=use_cross, subgrad_mono_cut=use_subgrad
    )
    mip_inf = _mip_inf_norm_ista(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, cross={use_cross}, subgrad={use_subgrad}: "
        f"MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


@pytest.mark.parametrize("use_cross,use_subgrad", PEP_FLAG_COMBOS)
@pytest.mark.parametrize("K", [1, 2])
def test_boxpgd_mip_bound_preserved_pep_combos(use_cross, use_subgrad, K):
    _, sdp_sq = _build_boxpgd_sdp(
        K=K, sdp_solver="clarabel", cross_cut=use_cross, subgrad_mono_cut=use_subgrad
    )
    mip_inf = _mip_inf_norm_boxpgd(K)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS, (
        f"K={K}, cross={use_cross}, subgrad={use_subgrad}: "
        f"MIP inf norm ({mip_inf}) > SDP two norm ({np.sqrt(sdp_sq)})"
    )


def test_pep_cuts_preserve_mip_bound_with_all_other_cuts():
    """Stacking the PEP cuts on top of every other cut family still preserves the
    MIP lower bound — composing valid inequalities never removes a true point."""
    _, sdp_sq = _build_ista_sdp(
        K=2,
        sdp_solver="clarabel",
        rlt_cuts=True,
        triangle_cuts=True,
        relu_envelope_cuts=True,
        cross_cut=True,
        subgrad_mono_cut=True,
    )
    mip_inf = _mip_inf_norm_ista(2)
    assert mip_inf <= np.sqrt(sdp_sq) + TEST_EPS


def test_pep_cuts_compose_no_worse_than_base():
    """base+both <= base+cross and base+subgrad and base (maximizing). Every PEP
    cut is a valid inequality, so adding more can only tighten the relaxation."""
    _, baseline = _build_ista_sdp(K=2, sdp_solver="clarabel")
    _, with_cross = _build_ista_sdp(K=2, sdp_solver="clarabel", cross_cut=True)
    _, with_subgrad = _build_ista_sdp(K=2, sdp_solver="clarabel", subgrad_mono_cut=True)
    _, with_both = _build_ista_sdp(
        K=2, sdp_solver="clarabel", cross_cut=True, subgrad_mono_cut=True
    )
    assert with_cross <= baseline + TEST_EPS
    assert with_subgrad <= baseline + TEST_EPS
    assert with_both <= with_cross + TEST_EPS
    assert with_both <= with_subgrad + TEST_EPS
    assert with_both <= baseline + TEST_EPS  # explicit (also implied transitively)


# ============================================================
# Phase 10: tightening regression — the cuts close the gap to the exact MIP
# ============================================================
#
# The *_tighten_bound tests above only assert the cuts never *loosen* the bound
# (<= baseline). This locks in the measured *effect*: on mixed-regime instances
# (iterate boxes straddling the operator kinks) the cuts close most of the
# SDP-to-exact-MIP gap, and on ISTA the cross and subgradient families are
# complementary (both strictly beats either alone). Compares against the *exact*
# MIP with a squared two-norm objective, so it requires Gurobi (HiGHS cannot do
# the MIQP); skipped if Gurobi is unavailable.

_PEP_GUROBI = {"OutputFlag": 0, "LogToConsole": 0}


def _pep_mixed_builder(problem):
    """Return a ``build(vp, K)`` for a mixed-regime instance with a real
    SDP-vs-MIP gap (iterate box straddles the kink). Deterministic per problem."""
    if problem == "nnls":
        rng = np.random.default_rng(1)
        m, n, box = 4, 3, 2.0
        A = rng.normal(size=(m, n))
        t = 1.0 / float(np.max(np.linalg.eigvalsh(A.T @ A)))

        def build(vp, K=2):
            x = vp.add_param(m, lb=0.0, ub=box)
            z = [vp.add_iterate(n, lb=-box, ub=box)]
            for k in range(K):
                z.append(vp.step.relu(z[k] - t * A.T @ (A @ z[k] - x)))
            return z[K] - z[K - 1]

        return build

    rng = np.random.default_rng(15)
    m, n, box = 6, 4, 1.5
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m,))
    t = 1.0 / float(np.max(np.linalg.eigvalsh(A.T @ A)))
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    lam = 2.0 * t

    def build(vp, K=2):
        bp = vp.add_param(m, lb=b - 0.5, ub=b + 0.5)
        z = [vp.add_iterate(n, lb=-box, ub=box)]
        for k in range(K):
            z.append(vp.step.soft_threshold(At @ z[k] + Bt @ bp, lambd=lam))
        return z[K] - z[K - 1]

    return build


def _pep_sdp_value(build, **cut_kwargs):
    """SDP (clarabel) squared-two-norm value on the light base + given PEP cuts."""
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        rlt_cuts=True,
        relu_envelope_cuts=True,
        **cut_kwargs,
    )
    expr = build(vp)
    vp.set_objective_norm(expr, 2)
    return vp.solve()


@pytest.mark.parametrize(
    "problem,min_cross,min_subgrad,min_both,complementary",
    [
        # thresholds are conservative vs measured (cross 97.7%, subgrad 97.7%,
        # both 98.7% on NNLS; cross 88.7%, subgrad 80.4%, both 95.0% on ISTA).
        ("nnls", 0.85, 0.65, 0.85, False),
        ("ista", 0.70, 0.55, 0.85, True),
    ],
)
def test_pep_cuts_close_gap_to_mip(problem, min_cross, min_subgrad, min_both, complementary):
    """Regression: the PEP cuts close most of the SDP-to-exact-MIP gap (and on
    ISTA the two families are complementary)."""
    pytest.importorskip("gurobipy", reason="exact two-norm MIP requires Gurobi")
    build = _pep_mixed_builder(problem)

    vp_mip = Verifier(method="mip", solver="gurobi", obbt_solver=None, params=dict(_PEP_GUROBI))
    vp_mip.set_objective_norm(build(vp_mip), 2)
    mip = vp_mip.solve()

    base = _pep_sdp_value(build)
    cross = _pep_sdp_value(build, cross_cut=True)
    subgrad = _pep_sdp_value(build, subgrad_mono_cut=True)
    both = _pep_sdp_value(build, cross_cut=True, subgrad_mono_cut=True)

    # Every relaxation is a valid upper bound on the exact value.
    for label, v in (("base", base), ("cross", cross), ("subgrad", subgrad), ("both", both)):
        assert v >= mip - TEST_EPS, f"{problem}/{label}: sdp {v} < mip {mip} (invalid cut)"

    base_gap = base - mip
    assert base_gap > 0.1, f"{problem}: base instance has no gap to close ({base_gap})"

    def closed(v):
        return (base_gap - (v - mip)) / base_gap

    assert closed(cross) >= min_cross, f"{problem}: cross closed only {closed(cross):.1%}"
    assert closed(subgrad) >= min_subgrad, f"{problem}: subgrad closed only {closed(subgrad):.1%}"
    assert closed(both) >= min_both, f"{problem}: both closed only {closed(both):.1%}"

    # Composition is never worse than either family alone.
    assert both <= cross + TEST_EPS
    assert both <= subgrad + TEST_EPS

    if complementary:
        # ISTA: combining strictly beats either family (>= 10% smaller residual gap).
        assert (both - mip) <= 0.9 * (cross - mip) + TEST_EPS
        assert (both - mip) <= 0.9 * (subgrad - mip) + TEST_EPS


# ============================================================
# Phase 11: PEP box-projection cuts for saturated-linear (satlin)
# ============================================================
#
# satlin = clip(v, l, u) (box projection). It decomposes to relu(v-l)-relu(v-u)+l,
# so the ReLU-level cross/subgrad cuts already fire on its internal ReLUs; these add
# the box-operator-level cuts on the composite z and residual r = v - z (emitted
# under the same cross_cut / subgrad_mono_cut flags):
#   normal-cone (cross_cut): r_j(z_j-l_j)>=0, r_j(z_j-u_j)>=0, r_j^b(z_j^a-z_j^b)<=0
#   monotonicity (subgrad_mono_cut): (z_j^a-z_j^b)(r_j^a-r_j^b)>=0
# Per-coordinate, same-box only (cross-coordinate is invalid; verified in planning).

GUROBI_SILENT_BOX = {"OutputFlag": 0, "LogToConsole": 0}


def _build_boxpgd_mixed(vp, K, P, q, lo, hi, t):
    """Box-PGD objective expr with a straddling iterate box so satlin rows land in
    the uncertain (mixed) cases. Identical build for SDP and MIP."""
    n = P.shape[0]
    qp = vp.add_param(n, lb=q - 0.5, ub=q + 0.5)
    z = [vp.add_iterate(n, lb=-1.5, ub=1.5)]
    for k in range(K):
        z.append(vp.step.satlin(z[k] - t * (P @ z[k] + qp), l=lo, u=hi))
    return z[K] - z[K - 1]


def _boxpgd_mixed_data(seed=22):
    rng = np.random.default_rng(seed)
    n = 2
    P = np.eye(n) + 0.1 * rng.normal(size=(n, n))
    P = (P + P.T) / 2
    q = rng.normal(size=n)
    lo = np.array([-1.0, -0.5])
    hi = np.array([0.5, 1.0])
    return P, q, lo, hi, 0.5


def _sdp_true_trajectory_lift(canon, rng):
    """Rank-1 lift X = u u^T of a true feed-forward trajectory: free leaves
    (params/iterates) sampled in-bounds, ReLU output leaves filled topologically
    from their args. Used to check every emitted cut holds on a true point."""
    dim = canon.get_sdp_var_dim()
    u = np.zeros(dim)
    u[0] = 1.0
    relu_out = {s.output_leaves[0] for s in canon._relu_steps}
    for leaf, (s, e) in canon.var_idx_map.items():
        if leaf in relu_out:
            continue
        lbv = np.atleast_1d(leaf.lb)
        ubv = np.atleast_1d(leaf.ub)
        u[s:e] = rng.uniform(lbv, ubv)
    for step in canon._relu_steps:
        A, idxs = canon.linexpr_to_Amat_and_bounds(step.arg)
        xv = np.concatenate([u[a:b] for (a, b) in idxs])
        argv = np.asarray(A @ xv).flatten() + np.asarray(step.arg.const).flatten()
        ys, ye = canon.var_idx_map[step.output_leaves[0]]
        u[ys:ye] = np.maximum(0.0, argv[sorted(step.proj_indices)])
    return np.outer(u, u)


def test_box_cuts_registered_and_emitted():
    """A satlin step registers a box prox and, with the flags on, emits box cuts."""
    P, q, lo, hi, t = _boxpgd_mixed_data()
    vp = Verifier(
        method="sdp", solver="clarabel", obbt_solver=None, cross_cut=True, subgrad_mono_cut=True
    )
    expr = _build_boxpgd_mixed(vp, 2, P, q, lo, hi, t)
    vp.set_objective_norm(expr, 2)
    # Two satlin steps -> two box-prox entries registered.
    assert len(vp.canonicalizer._box_prox_steps) == 2
    vp.solve()
    # Box cuts are emitted (more than the ReLU-only cross cuts would produce).
    base = Verifier(method="sdp", solver="clarabel", obbt_solver=None, cross_cut=True)
    _ = _build_boxpgd_mixed(base, 2, P, q, lo, hi, t)
    base.set_objective_norm(_, 2)
    base.solve()
    assert len(_solved_canon(vp)._cut_ineq_matrices) > 0


@pytest.mark.parametrize("seed", [22, 0, 7])
def test_box_cuts_valid_on_true_trajectory(seed):
    """Every emitted cut (ReLU cross + box normal-cone + box monotonicity) holds on
    the rank-1 lift of many true box-PGD trajectories — i.e. no cut removes a true
    point."""
    P, q, lo, hi, t = _boxpgd_mixed_data(seed)
    vp = Verifier(
        method="sdp", solver="clarabel", obbt_solver=None, cross_cut=True, subgrad_mono_cut=True
    )
    vp.set_objective_norm(_build_boxpgd_mixed(vp, 3, P, q, lo, hi, t), 2)
    vp.solve()
    canon = _solved_canon(vp)
    rng = np.random.default_rng(123)
    for _ in range(40):
        X = _sdp_true_trajectory_lift(canon, rng)
        for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
            assert float((M @ X).trace()) <= b + 1e-7, "box/ReLU cut cuts a true point"


def test_box_mono_cuts_same_box_only():
    """Box monotonicity (C4) fires only between satlin steps with the SAME box;
    a differently-boxed step shares no C4 cut with the others."""

    # Build through the decomposer path so box proxes register, then count C4 cuts.
    def _count(boxes):
        vp = Verifier(method="sdp", solver="clarabel", obbt_solver=None, subgrad_mono_cut=True)
        x = vp.add_iterate(1, lb=-2.0, ub=2.0)
        for lo, hi in boxes:
            x = vp.step.satlin(x, l=np.array([lo]), u=np.array([hi]))
        vp.set_objective_norm(x, 2)
        vp.solve()
        canon = _solved_canon(vp)
        canon._cut_ineq_matrices = []
        canon._cut_bineq_upperbounds = []
        canon._build_box_mono_cuts()
        return len(canon._cut_ineq_matrices)

    same = _count([(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)])  # 3 same-box -> C(3,2)=3 pairs * 1 coord
    mixed = _count([(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)])  # only 1 same-box pair
    assert same == 3
    assert mixed == 1


def test_box_cuts_close_gap_to_mip():
    """Tightening regression: on a mixed-regime box-PGD instance the box cuts close
    most of the SDP-to-exact-MIP gap (and stay valid)."""
    pytest.importorskip("gurobipy", reason="exact two-norm MIP requires Gurobi")
    P, q, lo, hi, t = _boxpgd_mixed_data(22)
    K = 3

    vp_mip = Verifier(
        method="mip", solver="gurobi", obbt_solver=None, params=dict(GUROBI_SILENT_BOX)
    )
    vp_mip.set_objective_norm(_build_boxpgd_mixed(vp_mip, K, P, q, lo, hi, t), 2)
    mip = vp_mip.solve()

    def sdp(**kw):
        vp = Verifier(
            method="sdp", solver="clarabel", obbt_solver=None,
            rlt_cuts=True, relu_envelope_cuts=True, **kw,
        )
        vp.set_objective_norm(_build_boxpgd_mixed(vp, K, P, q, lo, hi, t), 2)
        return vp.solve()

    base = sdp()
    cross = sdp(cross_cut=True)
    subgrad = sdp(subgrad_mono_cut=True)
    both = sdp(cross_cut=True, subgrad_mono_cut=True)

    for label, v in (("base", base), ("cross", cross), ("subgrad", subgrad), ("both", both)):
        assert v >= mip - TEST_EPS, f"{label}: sdp {v} < mip {mip} (invalid box cut)"

    base_gap = base - mip
    assert base_gap > 0.1
    assert (base_gap - (cross - mip)) / base_gap >= 0.85
    assert (base_gap - (both - mip)) / base_gap >= 0.85
