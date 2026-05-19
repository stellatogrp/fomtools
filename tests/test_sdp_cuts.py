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
