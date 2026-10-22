"""Hull inequalities for the SDP soft-threshold and satlin decompositions.

``SDPStepDecomposer`` writes soft-threshold and satlin as a difference of two
ReLUs, and registers the convex hull of the composite map ``v -> z`` over the
input bounds as linear inequalities (see ``SDPStepDecomposer._add_hull_rows``).
These tests check that the rows are registered, that every constraint and cut
holds on true trajectories, and that the relaxation is close to the exact
two-norm MIP on small instances that were loose before.
"""

import numpy as np
import pytest

import fomtools as ft
from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

_GUROBI = {"OutputFlag": 0, "LogToConsole": 0}


def _lasso_data(seed=0, p=3, n=5, lambd=0.1):
    """Dense nonstrongly convex lasso (p < n), sparse-coding style parameter box."""
    rng = np.random.default_rng(seed)
    D = rng.normal(scale=np.sqrt(1.0 / p), size=(p, n))
    D /= np.linalg.norm(D, axis=0)
    z_star = rng.normal(size=(n, 100)) * rng.binomial(1, 0.1, size=(n, 100))
    x_samples = (D @ z_star).T
    s0 = np.linalg.pinv(D) @ x_samples[rng.integers(100)]
    eta = 1.0 / float(np.linalg.eigvalsh(D.T @ D)[-1])
    return D, eta, lambd, x_samples.min(axis=0), x_samples.max(axis=0), s0


def _build_ista(vp, K):
    """ISTA on the lasso; returns the KKT residual expression at u^K."""
    D, eta, lambd, x_lb, x_ub, s0 = _lasso_data()
    x = vp.parameter(D.shape[0], lb=x_lb, ub=x_ub)
    u = [vp.iterate(D.shape[1], lb=s0, ub=s0)]
    for _ in range(K):
        u.append(vp.soft_threshold(u[-1] - eta * D.T @ (D @ u[-1] - x), lambd=lambd * eta))
    G = D.T @ D - np.eye(D.shape[1]) / eta
    return G @ (u[K] - u[K - 1])


def _build_boxpgd(vp, K):
    """Box-constrained least-squares PGD; returns the scaled step ``(z^K - z^{K-1})/t``."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    t = 1.0 / float(np.linalg.eigvalsh(A.T @ A)[-1])
    x = vp.parameter(5, lb=-2.0, ub=2.0)
    z = [vp.iterate(3, lb=np.zeros(3), ub=np.zeros(3))]
    for _ in range(K):
        z.append(
            vp.satlin(z[-1] - t * A.T @ (A @ z[-1] - x), l=np.full(3, -0.5), u=np.full(3, 0.5))
        )
    return (1 / t) * (z[K] - z[K - 1])


_BUILDERS = {"softthreshold": _build_ista, "satlin": _build_boxpgd}


def _sdp(build, K, **flags):
    vp = Verifier(
        method="sdp",
        solver="clarabel",
        obbt_solver=None,
        rlt_cuts=True,
        triangle_cuts=True,
        relu_envelope_cuts=True,
        cross_cut=True,
        subgrad_mono_cut=True,
        **flags,
    )
    vp.maximize(ft.sum_squares(build(vp, K)))
    return vp


def _true_lift(canon, rng):
    """Rank-1 lift of a true trajectory: free leaves sampled in bounds, ReLU
    outputs filled in order from their args."""
    u = np.zeros(canon.get_sdp_var_dim())
    u[0] = 1.0
    relu_out = {s.output_leaves[0] for s in canon._relu_steps}
    for leaf, (s, e) in canon.var_idx_map.items():
        if leaf not in relu_out:
            u[s:e] = rng.uniform(np.atleast_1d(leaf.lb), np.atleast_1d(leaf.ub))
    for step in canon._relu_steps:
        A, idxs = canon.linexpr_to_Amat_and_bounds(step.arg)
        xv = np.concatenate([u[a:b] for (a, b) in idxs])
        argv = np.asarray(A @ xv).flatten() + np.asarray(step.arg.const).flatten()
        ys, ye = canon.var_idx_map[step.output_leaves[0]]
        u[ys:ye] = np.maximum(0.0, argv[sorted(step.proj_indices)])
    return np.outer(u, u)


def _value(M, X):
    d = M.shape[0]
    return float((M @ X[:d, :d]).trace())


@pytest.mark.parametrize("step", sorted(_BUILDERS))
def test_hull_rows_registered(step):
    """Each composite step registers one hull constraint (lines and chords for
    soft-threshold, chords for satlin) on top of the bare decomposition."""
    vp = _sdp(_BUILDERS[step], 1)
    n_rows = [c.lhs.ndim for c, _ in vp.canonicalizer._ineq_constraints]
    # One step on a 5-dim (lasso) or 3-dim (box) iterate with finite bounds.
    assert n_rows == ([4 * 5] if step == "softthreshold" else [2 * 3]), n_rows


@pytest.mark.parametrize("step", sorted(_BUILDERS))
@pytest.mark.parametrize("seed", [0, 1])
def test_hull_rows_valid_on_true_trajectory(step, seed):
    """Every constraint and cut, hull rows and their products included, holds on
    the rank-1 lift of true trajectories."""
    vp = _sdp(_BUILDERS[step], 3)
    vp.solve()
    canon = vp.canonicalizer._solved_copy
    rng = np.random.default_rng(seed)
    for _ in range(40):
        X = _true_lift(canon, rng)
        for M, b in zip(canon.Aineq_matrices, canon.bineq_upperbounds):
            assert _value(M, X) <= b + 1e-7, "a constraint cuts a true point"
        for M, b in zip(canon._cut_ineq_matrices, canon._cut_bineq_upperbounds):
            assert _value(M, X) <= b + 1e-7, "a cut cuts a true point"


def _mip(build, K):
    vp = Verifier(
        method="mip",
        solver="gurobi",
        obbt_solver=None,
        params={**_GUROBI, "NonConvex": 2, "MIPGap": 1e-6},
    )
    vp.maximize(ft.sum_squares(build(vp, K)))
    vp.solve()
    return float(vp.canonicalizer._solved_copy["ObjBound"])


@pytest.mark.parametrize(
    "step,K,max_ratio",
    [
        # Measured (norm ratio SDP / exact): before the hull rows 1.79 and 3.39 on the
        # lasso, 1.08 on box-PGD; with them 1.01, 1.01 and 1.05.
        ("softthreshold", 1, 1.05),
        ("softthreshold", 2, 1.05),
        ("satlin", 2, 1.065),
    ],
)
def test_hull_rows_close_gap_to_mip(step, K, max_ratio):
    """Regression: with the hull rows the full-cone SDP is close to the exact MIP."""
    pytest.importorskip("gurobipy", reason="exact two-norm MIP requires Gurobi")
    build = _BUILDERS[step]
    mip = _mip(build, K)
    vp = _sdp(build, K, full_psd_cone=True)
    sdp = vp.solve()
    assert sdp >= mip - TEST_EPS, f"sdp {sdp} < mip {mip} (invalid relaxation)"
    assert np.sqrt(sdp / mip) <= max_ratio, f"sdp/mip = {np.sqrt(sdp / mip):.3f} (norm)"
