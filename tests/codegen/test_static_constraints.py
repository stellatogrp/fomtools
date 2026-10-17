"""Tests for the embedded-C static invariant checks (no compiler needed)."""

import pytest

from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.targets.base import strip_c_comments_and_strings
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget


@pytest.fixture
def target():
    return EmbeddedCTarget(CodegenConfig())


def test_clean_source_passes(target):
    source = """
    #include "fom_solver.h"
    fom_int fom_solve(FomWorkspace *work) {
        int i;
        for (i = 0; i < FOM_STATE_DIM; ++i) {
            work->state[i] = work->state[i] + 1.0;
        }
        return FOM_SOLVED;
    }
    """
    assert target.static_checks(source) == []


def test_malloc_flagged(target):
    source = "void f(void){ double *x = malloc(8); free(x); }"
    violations = target.static_checks(source)
    assert any("malloc" in v for v in violations)
    assert any("free" in v for v in violations)


def test_division_flagged(target):
    source = "fom_float f(fom_float a, fom_float b){ return a / b; }"
    violations = target.static_checks(source)
    assert any("division" in v for v in violations)


def test_division_in_comment_not_flagged(target):
    source = """
    /* this comment has a / and a path/to/file */
    // another // comment with division a / b
    const char *s = "a/b/c";
    fom_int f(void){ return 0; }
    """
    assert target.static_checks(source) == []


def test_strip_comments_helper():
    stripped = strip_c_comments_and_strings('a /* /b */ c // d/e\n "f/g" h')
    assert "/b" not in stripped
    assert "d/e" not in stripped
    assert "f/g" not in stripped
    assert "a" in stripped and "c" in stripped and "h" in stripped


# -- kernel checks: rewrites stay tied to the checked data ---------------------
@pytest.fixture(scope="module")
def mpc_ir():
    from fomtools.codegen.examples import mpc_osqp_example
    from fomtools.codegen.extract import extract_ir

    ex = mpc_osqp_example(nx=2, nu=1, N=3)
    return extract_ir(ex["verifier"], ex["next_state"], config=CodegenConfig(), name="mpc")


def test_float_literals_flagged_in_rewritten_ops(target, mpc_ir):
    source = "void f(const fom_float *x, fom_float *out){ out[0] += 0.37 * x[1] + 1e-3f; }"
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert any("'0.37'" in v for v in violations)
    assert any("'1e-3f'" in v for v in violations)


def test_structural_literals_and_integers_allowed(target, mpc_ir):
    source = """
    /* 3.14 in a comment is fine */
    void f(const fom_float *x, fom_float *out){
        out[0] += 0.0 + 0.5 * x[1] + 1.0 * x[2] - 2.0 * x[3] + x[12] + 2. ;
    }
    """
    assert target.kernel_checks("src/fom_ops.c", source, mpc_ir) == []


def test_solver_may_not_read_matvec_matrix_data(target, mpc_ir):
    source = """
    void g(FomWorkspace *work){
        work->acc[0] += FOM_MAT_t0_state[0] * work->state[0];
        work->acc[1] += FOM_MAT_t0_p_p0_VALS[0];
    }
    """
    violations = target.kernel_checks("src/fom_solver.c", source, mpc_ir)
    assert any("FOM_MAT_t0_state directly" in v for v in violations)
    assert any("FOM_MAT_t0_p_p0 directly" in v for v in violations)


def test_solver_may_read_clip_bounds_and_dimensions(target, mpc_ir):
    source = """
    void g(FomWorkspace *work){
        int i;
        fom_apply_t0_state(work->state, work->acc);
        for (i = 0; i < FOM_MAT_t0_state_ROWS; ++i)
            if (work->t1[i] < FOM_MAT_t1_lo[i]) work->t1[i] = FOM_MAT_t1_lo[i];
    }
    """
    assert target.kernel_checks("src/fom_solver.c", source, mpc_ir) == []


def test_kernel_checks_skip_other_files_and_cuda(target, mpc_ir):
    from fomtools.codegen.config import Backend, Version
    from fomtools.codegen.targets import get_target

    source = "fom_float f(void){ return 0.37 * FOM_MAT_t0_state[0]; }"
    assert target.kernel_checks("src/fom_linsolve.c", source, mpc_ir) == []
    cuda = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.CUDA))
    assert cuda.kernel_checks(cuda.solver_source_path(), source, mpc_ir) == []
