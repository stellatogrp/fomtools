"""Tests for forwarding native solver options through ``Verifier.solve``.

Keyword arguments passed to ``vp.solve(...)`` that are not consumed by a
reserved FOMTools flag (e.g. ``convex_hull_cuts``) are forwarded verbatim to
the active backend solver as native options. Invalid names/values raise a
``ValueError``. These tests exercise the MIP backends (Gurobi, HiGHS) and the
SDP backends (Clarabel, cvxpy) via the ``vp`` / ``vp_sdp`` fixtures.
"""

import numpy as np
import pytest

import fomtools as ft
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
    x = vp.iterate(n, lb=0, ub=10)
    z = vp.relu(x - 2)
    vp.maximize(ft.norm([z], np.inf))
    return z


def _build_sdp_problem(vp_sdp):
    """Two-step NNLS chain mirroring ``test_sdp_nnls`` with a 2-norm objective."""
    m, n = 3, 2
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    t = 0.01

    x = vp_sdp.parameter(m, lb=0, ub=0.1)
    z0 = vp_sdp.iterate(n, lb=5, ub=5.1)
    z1 = vp_sdp.relu(z0 - t * A.T @ (A @ z0 - x))
    z2 = vp_sdp.relu(z1 - t * A.T @ (A @ z1 - x))
    vp_sdp.maximize(ft.sum_squares(z2 - z1))


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
    """A solve-time option overrides the same construction-time param.

    ``vp_param_2`` is built with a construction-time MIP gap of 1e-4. We
    override it at solve time with a distinctive value and read it back off the
    solved copy to confirm the solve-time value actually won (not merely that
    the solve succeeded).
    """
    _build_mip_problem(vp_param_2)
    solver_name = vp_param_2.canonicalizer.solver_name
    override = 0.137  # distinct from the construction-time 1e-4

    if solver_name == "gurobi":
        vp_param_2.solve(MIPGap=override)
        applied = vp_param_2.canonicalizer._solved_copy.model.Params.MIPGap
    else:  # highs
        vp_param_2.solve(mip_rel_gap=override)
        _, applied = vp_param_2.canonicalizer._solved_copy.model.getOptionValue("mip_rel_gap")

    assert applied == override


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


def test_gurobi_construction_param_validation():
    """Gurobi construction params go through the same verbatim ValueError path.

    A native Gurobi name is accepted; an unrecognized name raises ValueError.
    """
    Verifier(method="mip", solver="gurobi", obbt_solver=None, params={"TimeLimit": 1})

    with pytest.raises(ValueError):
        Verifier(method="mip", solver="gurobi", obbt_solver=None, params={"NotARealParam": 1})


def test_highs_construction_param_survives_solve():
    """HiGHS construction-time options must reach the solved copy.

    ``HiGHSSolver.copy()`` uses ``passModel``, which copies model data but not
    solver options, so without re-applying ``solver_params`` on the copy a
    construction-time option would silently revert to its default during
    ``solve_model``. We set a distinctive ``mip_rel_gap`` and read it back off
    the solved copy to confirm it was applied.
    """
    distinctive_gap = 0.123  # clearly different from the HiGHS default (1e-4)
    vp = Verifier(
        method="mip",
        solver="highs",
        obbt_solver=None,
        params={"output_flag": False, "mip_rel_gap": distinctive_gap},
    )
    _build_mip_problem(vp)
    vp.solve()

    _, applied_gap = vp.canonicalizer._solved_copy.model.getOptionValue("mip_rel_gap")
    assert applied_gap == distinctive_gap


def test_sdp_solve_time_option_overrides_construction(vp_sdp):
    """A solve-time SDP option overrides the same construction-time param.

    Construction supplies an invalid ``max_iter``; left in place it surfaces as
    a ``ValueError`` at solve, while a valid solve-time override replaces it and
    the solve succeeds. This pins precedence deterministically, without relying
    on solver convergence behavior (which would be flaky).
    """
    solver_name = vp_sdp.canonicalizer.solver_name
    vp = Verifier(
        method="sdp",
        solver=solver_name,
        obbt_solver=None,
        params={"max_iter": "not_an_int"},
    )
    _build_sdp_problem(vp)

    # Construction-time value is used when not overridden -> invalid -> raises.
    with pytest.raises(ValueError):
        vp.solve()

    # Solve-time override replaces it with a valid value -> succeeds.
    result = vp.solve(max_iter=10000)
    assert np.isfinite(result)


def test_cuclarabel_rejects_params_at_construction():
    """cuClarabel rejects custom options eagerly, at Verifier construction.

    Its Julia settings are hardcoded, so options cannot be applied. The error
    must surface when the Verifier is built (not only at solve time), and
    without requiring a GPU — the SDP canonicalizer validates before any CUDA
    check.
    """
    with pytest.raises(ValueError, match="cuClarabel does not support"):
        Verifier(method="sdp", solver="cuclarabel", obbt_solver=None, params={"max_iter": 1000})


def test_sdp_construction_param_applied(vp_sdp):
    """SDP construction params are stored and applied at solve time.

    Locks in the fix where passing ``params`` to an SDP verifier previously
    crashed (``NotImplementedError`` in ``set_param``). A valid native option
    now flows through to the Clarabel/cvxpy solve without error.
    """
    solver_name = vp_sdp.canonicalizer.solver_name
    vp = Verifier(method="sdp", solver=solver_name, obbt_solver=None, params={"max_iter": 10000})
    _build_sdp_problem(vp)
    result = vp.solve()
    assert np.isfinite(result)
