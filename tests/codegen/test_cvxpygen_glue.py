"""Unit tests for the cvxpygen C bridge and build wiring (no cvxpygen needed).

These exercise the deterministic emission in :mod:`fomtools.cvxpygen.glue`
directly — the OSQP-style primal/dual residual computation, the runtime
``check_termination`` plumbing, and the per-backend CMake source list — without
importing cvxpygen or running cvxpygen's canonicalizer. The full end-to-end path
(which needs cvxpygen + a compiler) lives in ``test_cvxpygen_e2e.py``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa

from fomtools.codegen.config import Backend, CodegenConfig, Version
from fomtools.codegen.targets import get_target
from fomtools.cvxpygen.builders import FomSpec
from fomtools.cvxpygen.glue import cmake_solver_sources, emit_bridge
from fomtools.cvxpygen.qpdata import QPData


def _qp(*, n=3, n_eq=2, n_ineq=3, with_p=True, seed=0):
    m = n_eq + n_ineq
    rng = np.random.default_rng(seed)
    P = spa.eye(n, format="csc") if with_p else spa.csc_matrix((n, n))
    A = spa.csc_matrix(rng.standard_normal((m, n))) if m else spa.csc_matrix((0, n))
    return QPData(
        P=P,
        q=np.zeros(n),
        A=A,
        b=np.zeros(n_eq),
        u_ineq=np.ones(n_ineq),
        d=0.0,
        n=n,
        n_eq=n_eq,
        n_ineq=n_ineq,
        param_dependent=frozenset({"q"}),
        bounds={"q": (-np.ones(n), np.ones(n))},
    )


def _spec(qp):
    # emit_bridge only reads param_copy / primal_offset / dual_offset off the spec.
    return FomSpec(
        verifier=None,
        next_state=None,
        name="t",
        primal_offset=0,
        primal_dim=qp.n,
        dual_offset=qp.n + qp.m,
        dual_dim=qp.m,
        param_copy=("q",),
        state=(),
        params=(),
    )


def _emit(tmp_path, qp, *, prefix="", has_quad_term=True, **kwargs):
    emit_bridge(_spec(qp), qp, tmp_path, prefix, has_quad_term=has_quad_term, **kwargs)
    return (tmp_path / "src" / "cpg_fom_bridge.c").read_text()


def test_bridge_reports_osqp_primal_dual_residuals(tmp_path):
    qp = _qp()
    src = _emit(tmp_path, qp)

    # Primal residual: ||Ax - clamp(Ax, l, u)||_inf via A x and the bounds.
    assert "for (i = 0; i < 5; i++) bridge_Ax[i] = 0.0;" in src
    assert "bridge_Ax[Canon_Params.A->i[j]] += Canon_Params.A->x[j] * sol_x[i];" in src
    assert "double over = bridge_Ax[i] - Canon_Params.u[i];" in src
    assert "double under = Canon_Params.l[i] - bridge_Ax[i];" in src

    # Dual residual: ||P x + q + A^T y||_inf.
    assert "bridge_g[i] = Canon_Params.q[i];" in src
    assert "bridge_g[Canon_Params.P->i[j]] += Canon_Params.P->x[j] * sol_x[i];" in src
    assert "bridge_g[i] += Canon_Params.A->x[j] * sol_y[Canon_Params.A->i[j]];" in src

    # The old behaviour (both residuals = the FOM state-change metric) is gone.
    assert "fom_pri_res = (double)fom_work.solution.residual;" not in src
    assert "fom_dua_res = (double)fom_work.solution.residual;" not in src
    assert src.count("fom_pri_res = 0.0;") == 1
    assert src.count("fom_dua_res = 0.0;") == 1


def test_bridge_copies_runtime_check_termination(tmp_path):
    src = _emit(tmp_path, _qp())
    assert "fom_work.settings.check_termination = (fom_int)Canon_Settings.check_termination;" in src
    # max_iter / tol stay runtime-tunable too.
    assert "fom_work.settings.max_iter = (fom_int)Canon_Settings.max_iter;" in src
    assert "fom_work.settings.tol = (fom_float)Canon_Settings.tol;" in src


def test_bridge_omits_settings_for_an_unrolled_solver(tmp_path):
    """An unrolled body honors no runtime setting, so cvxpygen emits no struct.

    With ``stgs`` empty the interface also turns off
    ``stgs_requires_extra_struct_type``, so ``Canon_Settings`` — its type, its
    instance and every ``cpg_set_solver_*`` — is absent from the generated code.
    Naming any member here would then fail to compile.
    """
    src = _emit(tmp_path, _qp(), settings=())

    assert "Canon_Settings" not in src
    assert "fom_work.settings" not in src
    assert "No runtime settings" in src


def test_bridge_caps_max_iter_at_the_schedule_tables(tmp_path):
    """Past the tables the accessors clamp, which would freeze the coefficients."""
    src = _emit(tmp_path, _qp(), schedule_len=64)

    assert (
        "fom_work.settings.max_iter = (fom_int)(Canon_Settings.max_iter < 64 "
        "? Canon_Settings.max_iter : 64);" in src
    )
    # The other two stay plain copies: a tolerance and a check interval mean the
    # same thing however far the loop runs.
    assert "fom_work.settings.tol = (fom_float)Canon_Settings.tol;" in src
    assert "fom_work.settings.check_termination = (fom_int)Canon_Settings.check_termination;" in src


def test_bridge_reports_the_recorded_iteration_count_when_unrolled(tmp_path):
    """``solution.iter`` is 1 for a single pass, which is not what the run did."""
    src = _emit(tmp_path, _qp(), settings=(), iter_count=12)

    assert "fom_iter = 12;" in src
    assert "fom_iter = (int)fom_work.solution.iter;" not in src


def test_bridge_reports_the_loop_counter_by_default(tmp_path):
    src = _emit(tmp_path, _qp())
    assert "fom_iter = (int)fom_work.solution.iter;" in src


def test_bridge_forwards_only_the_settings_it_is_given(tmp_path):
    src = _emit(tmp_path, _qp(), settings=("tol",))

    assert "fom_work.settings.tol = (fom_float)Canon_Settings.tol;" in src
    assert "Canon_Settings.max_iter" not in src
    assert "Canon_Settings.check_termination" not in src


def test_bridge_lp_has_no_quadratic_term(tmp_path):
    qp = _qp(with_p=False)
    src = _emit(tmp_path, qp, has_quad_term=False)
    # No P contribution to the dual residual or the objective.
    assert "Canon_Params.P" not in src
    assert "no quadratic term" in src
    # But the dual residual still has q + A^T y.
    assert "bridge_g[i] = Canon_Params.q[i];" in src
    assert "bridge_g[i] += Canon_Params.A->x[j] * sol_y[Canon_Params.A->i[j]];" in src


def test_bridge_unconstrained_skips_primal(tmp_path):
    qp = _qp(n_eq=0, n_ineq=0)
    src = _emit(tmp_path, qp)
    assert "fom_pri_res = 0.0;  /* no constraints */" in src
    assert "bridge_Ax[i] = 0.0" not in src  # no A loops at all
    assert "Canon_Params.A" not in src


def test_bridge_applies_symbol_prefix(tmp_path):
    src = _emit(tmp_path, _qp(), prefix="myp")
    assert "void mypcpg_fom_solve(void)" in src
    assert "mypCanon_Settings.check_termination" in src
    assert "mypCanon_Params.A->x[j]" in src


def test_cmake_solver_sources_embedded_is_all_c(tmp_path):
    target = get_target(CodegenConfig(version=Version.EMBEDDED, backend=Backend.C))
    srcs = cmake_solver_sources(target)
    assert any(s.endswith("src/fom_solver.c") for s in srcs)
    assert any(s.endswith("src/fom_ops.c") for s in srcs)
    assert any(s.endswith("src/fom_linsolve.c") for s in srcs)
    assert any(s.endswith("src/fom_objective.c") for s in srcs)
    assert any(s.endswith("src/cpg_fom_bridge.c") for s in srcs)
    assert not any(s.endswith(".cu") for s in srcs)


def test_cmake_solver_sources_openmp_reuses_c():
    target = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.OPENMP))
    srcs = cmake_solver_sources(target)
    assert any(s.endswith("src/fom_solver.c") for s in srcs)
    assert not any(s.endswith(".cu") for s in srcs)


def test_cmake_solver_sources_cuda_uses_cu():
    target = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.CUDA))
    srcs = cmake_solver_sources(target)
    assert any(s.endswith("src/fom_solver.cu") for s in srcs)
    assert any(s.endswith("src/fom_ops.cu") for s in srcs)
    assert any(s.endswith("src/fom_linsolve.cu") for s in srcs)
    assert any(s.endswith("src/fom_objective.cu") for s in srcs)
    assert len(srcs) == len(set(srcs))
    # Numeric data and the bridge stay host C even on CUDA.
    assert any(s.endswith("src/fom_data.c") for s in srcs)
    assert any(s.endswith("src/cpg_fom_bridge.c") for s in srcs)


def test_workspace_header_exposes_runtime_check_termination():
    from fomtools.codegen.extract import extract_ir
    from tests.codegen.conftest import build_ista

    ex = build_ista()
    ir = extract_ir(ex["verifier"], ex["next_state"], config=ex["config"])
    header = get_target(ex["config"])._workspace_header(ir)
    settings_block = header.split("} FomSettings;")[0]
    assert "fom_int check_termination;" in settings_block


def test_bridge_without_scaling_copies_and_publishes_unscaled(tmp_path):
    src = _emit(tmp_path, _qp())
    assert "bridge_D" not in src and "bridge_E" not in src
    assert "fom_params.p0[i] = (fom_float)((Canon_Params.q)[i]);" in src
    assert "sol_x[i] = (double)fom_work.solution.x[0 + i];" in src


def test_bridge_with_scaling_scales_parameters_in_and_solution_out(tmp_path):
    from fomtools.cvxpygen.scaling import Scaling

    qp = _qp()
    scaling = Scaling(D=np.array([1.0, 2.0, 4.0]), E=np.arange(1.0, 6.0), c=0.5)
    src = _emit(tmp_path, qp, scaling=scaling)

    assert "static const double bridge_D[3] = {1, 2, 4};" in src
    assert "static const double bridge_E[5] = {1, 2, 3, 4, 5};" in src
    assert "static const double bridge_c = 0.5;" in src
    # q in: c D q.  x out: D x~.  y out: E y~ / c.
    assert "bridge_c * bridge_D[i] * (Canon_Params.q)[i]" in src
    assert "sol_x[i] = bridge_D[i] * (double)fom_work.solution.x[0 + i];" in src
    assert "sol_y[i] = bridge_E[i] * (double)fom_work.solution.x[8 + i] / bridge_c;" in src
    # Residuals still read the original canonical data.
    assert "double over = bridge_Ax[i] - Canon_Params.u[i];" in src


def test_bridge_scales_inequality_bounds_but_keeps_infinities(tmp_path):
    from dataclasses import replace

    from fomtools.cvxpygen.scaling import Scaling

    qp = _qp()
    qp = replace(
        qp, param_dependent=frozenset({"u_ineq"}), bounds={"u_ineq": (np.zeros(3), np.ones(3))}
    )
    spec = replace(_spec(qp), param_copy=("u_ineq",))
    scaling = Scaling(D=np.ones(3), E=np.arange(1.0, 6.0), c=1.0)
    emit_bridge(spec, qp, tmp_path, "", has_quad_term=True, scaling=scaling)
    src = (tmp_path / "src" / "cpg_fom_bridge.c").read_text()
    # u_ineq rows start after the n_eq = 2 equality rows.
    assert "bridge_E[2 + i] * (Canon_Params.u + 2)[i]" in src
    assert ">= 1e+30" in src
