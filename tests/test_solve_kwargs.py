"""Tests for forwarding native solver options through ``Verifier.solve``.

Keyword arguments passed to ``vp.solve(...)`` that are not consumed by a
reserved FOMTools flag (e.g. ``convex_hull_cuts``) are forwarded verbatim to
the active backend solver as native options. Invalid names/values raise a
``ValueError``. These tests exercise the MIP backends (Gurobi, HiGHS) and the
SDP backends (Clarabel, cvxpy) via the ``vp`` / ``vp_sdp`` fixtures.
"""

import numpy as np
import pytest

from fomtools.verifier import Verifier
from tests.conftest import TEST_EPS

# Valid native option names differ per backend.
VALID_MIP_OPTIONS = {
    "gurobi": {"OutputFlag": 0, "MIPGap": 1e-9},
    "highs": {"output_flag": False, "mip_rel_gap": 1e-9},
}

# Both Clarabel and cvxpy (Clarabel backend) accept ``max_iter``.
VALID_SDP_OPTIONS = {"max_iter": 10000}


def _build_mip_problem(vp):
    """Small ReLU problem with an inf-norm objective (works for both MIP solvers)."""
    n = 3
    x = vp.add_iterate(n, lb=0, ub=10)
    z = vp.step.relu(x - 2)
    vp.set_objective_norm([z], np.inf)
    return z


def _build_sdp_problem(vp_sdp):
    """Two-step NNLS chain mirroring ``test_sdp_nnls`` with a 2-norm objective."""
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp_sdp.add_param(m, lb=0, ub=0.1)
    z0 = vp_sdp.add_iterate(n, lb=5, ub=5.1)
    z1 = vp_sdp.step.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp_sdp.step.relu(z1 - t * A.T @ (A @ z1 - x))
    vp_sdp.set_objective_norm(z2 - z1, 2)


# ---------------------------------------------------------------------------
# MIP backends (Gurobi, HiGHS)
# ---------------------------------------------------------------------------


def test_valid_solver_option_mip(vp):
    """Valid native options are forwarded and do not change the optimum."""
    _build_mip_problem(vp)
    baseline = vp.solve()

    options = VALID_MIP_OPTIONS[vp.canonicalizer.solver_name]
    result = vp.solve(**options)

    assert np.isfinite(result)
    assert np.abs(result - baseline) < TEST_EPS


def test_invalid_solver_option_mip(vp):
    """An unrecognized native option name raises ValueError."""
    _build_mip_problem(vp)
    with pytest.raises(ValueError):
        vp.solve(NotARealOption=123)


def test_reserved_flag_with_solver_option_mip(vp):
    """A reserved flag is consumed; a solver option alongside it is forwarded."""
    _build_mip_problem(vp)
    options = VALID_MIP_OPTIONS[vp.canonicalizer.solver_name]

    # convex_hull_cuts binds to the named parameter (not forwarded as an option).
    result = vp.solve(convex_hull_cuts=True, **options)
    assert np.isfinite(result)


def test_solve_time_option_overrides_construction(vp_param_2):
    """Solve-time options override the construction-time ``params`` dict."""
    # vp_param_2 is constructed with {"TimeLimit": 10, "MIPGap": 1e-4, "OutputFlag": 0}.
    _build_mip_problem(vp_param_2)
    options = VALID_MIP_OPTIONS[vp_param_2.canonicalizer.solver_name]
    result = vp_param_2.solve(**options)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# SDP backends (Clarabel, cvxpy)
# ---------------------------------------------------------------------------


def test_valid_solver_option_sdp(vp_sdp):
    """Valid native Clarabel settings are forwarded through the SDP solve."""
    _build_sdp_problem(vp_sdp)
    baseline = vp_sdp.solve()

    result = vp_sdp.solve(**VALID_SDP_OPTIONS)
    assert np.isfinite(result)
    assert np.abs(result - baseline) < TEST_EPS


def test_invalid_solver_option_sdp(vp_sdp):
    """An unrecognized native option name raises ValueError on the SDP path."""
    _build_sdp_problem(vp_sdp)
    with pytest.raises(ValueError):
        vp_sdp.solve(NotARealOption=123)


# ---------------------------------------------------------------------------
# Construction-time params use the same verbatim path (no name translation)
# ---------------------------------------------------------------------------


def test_highs_construction_param_is_verbatim():
    """HiGHS construction params take native names, not Gurobi-style ones.

    Confirms the old ``param_to_highs`` translation is gone: a Gurobi-style
    name raises, while the native HiGHS option name is accepted.
    """
    with pytest.raises(ValueError):
        Verifier(method="mip", solver="highs", obbt_solver=None, params={"TimeLimit": 1})

    # Native HiGHS option name is accepted.
    Verifier(method="mip", solver="highs", obbt_solver=None, params={"time_limit": 1})
