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
    full_psd_cone=False,
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
        full_psd_cone=full_psd_cone,
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


def _build_mixed_relu_problem(
    triangle_cuts=False, rlt_cuts=False, relu_envelope_cuts=False
):
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
    full_psd_cone=False,
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
        full_psd_cone=full_psd_cone,
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
    full_psd_cone=False,
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
        full_psd_cone=full_psd_cone,
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
    vp = Verifier(
        method="sdp", solver=sdp_solver, obbt_solver=None, relu_envelope_cuts=True
    )
    assert vp.canonicalizer.relu_envelope_cuts is True
    vp2 = Verifier(method="sdp", solver=sdp_solver, obbt_solver=None)
    assert vp2.canonicalizer.relu_envelope_cuts is False


@pytest.mark.parametrize("flag", ["relu_envelope_cuts", "full_psd_cone"])
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
    vp = Verifier(
        method="sdp", solver="clarabel", obbt_solver=None, relu_envelope_cuts=True
    )
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

    vp, _ = _build_nnls_sdp(
        K=1, sdp_solver="clarabel", full_psd_cone=True, rlt_cuts=True
    )
    canon = _solved_canon(vp)
    dim = canon.get_sdp_var_dim()
    n_pairs = len(list(combinations(range(1, dim), 2)))
    # Envelope/triangle off here, no inequality constraints -> all cuts are RLT.
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
    _, with_full = _build_nnls_sdp(
        K=1, sdp_solver="clarabel", full_psd_cone=True, rlt_cuts=True
    )
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
