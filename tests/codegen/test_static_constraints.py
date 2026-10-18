"""Tests for the embedded-C static invariant checks (no compiler needed)."""

import pytest

from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.kernel_rules import strip_c_comments_and_strings
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


#: The setup functions every solver defines (the kernel rules look inside them).
SETUP = """
void fom_setup(FomWorkspace *work, const FomSettings *settings) { (void)settings; }
void fom_update_params(FomWorkspace *work, const FomParams *params) { (void)params; }
"""


def _ops(ir, bodies=None) -> str:
    """A fom_ops.c defining every fom_apply_<name>, with ``bodies[name]`` where given."""
    from fomtools.codegen.targets.embedded_c import _matvec_matrix_names

    bodies = bodies or {}
    return "\n".join(
        f"void fom_apply_{n}(const fom_float *x, fom_float *out) {{ {bodies.get(n, '')} }}"
        for n in _matvec_matrix_names(ir)
    )


def test_float_literals_flagged_in_rewritten_ops(target, mpc_ir):
    source = "void f(const fom_float *x, fom_float *out){ out[0] += 0.37 * x[1] + 1e-3f; }"
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert any("'0.37'" in v for v in violations)
    assert any("'1e-3f'" in v for v in violations)


def test_structural_literals_and_integers_allowed(target, mpc_ir):
    source = _ops(
        mpc_ir,
        {
            "t0_state": "/* 3.14 in a comment is fine */ "
            "out[0] += 0.0 + 0.5 * x[1] + 1.0 * x[2] - 2.0 * x[3] + x[12] + 2. "
            "+ (fom_float)1 + (fom_float)0;",
        },
    )
    assert target.kernel_checks("src/fom_ops.c", source, mpc_ir) == []


def test_hex_floats_and_integer_casts_are_flagged(target, mpc_ir):
    source = _ops(mpc_ir, {"t0_state": "out[0] += 0x1.8p0 * x[0] + (fom_float)3 * x[1];"})
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert any("'0x1.8p0'" in v for v in violations)
    assert any("'(fom_float)3'" in v for v in violations)
    # Hex integers (an index) and a structural cast are not coefficients.
    source = _ops(mpc_ir, {"t0_state": "out[0x2] += (fom_float)2 * x[0x1];"})
    assert target.kernel_checks("src/fom_ops.c", source, mpc_ir) == []


def test_integer_arithmetic_is_left_to_the_numerical_checks(target, mpc_ir):
    # The documented limit of a syntactic rule: an integer coefficient, or one
    # computed from allowed constants, gets past it (the probes catch it).
    source = _ops(mpc_ir, {"t0_state": "out[0] += 3 * x[0] + (1.0 + 1.0 + 1.0) * x[1];"})
    assert target.kernel_checks("src/fom_ops.c", source, mpc_ir) == []


def test_ops_routines_read_only_their_own_matrix(target, mpc_ir):
    source = _ops(mpc_ir, {"t0_state": "out[0] += FOM_MAT_su_state_VALS[0] * x[0];"})
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert violations == [
        "fom_apply_t0_state reads FOM_MAT_su_state, the data of another product; each "
        "routine may read only its own matrix"
    ]
    # Its own data, and another matrix's dimensions, are fine.
    source = _ops(
        mpc_ir,
        {"t0_state": "out[0] += FOM_MAT_t0_state_VALS[0] * x[FOM_MAT_su_state_COLS - 1];"},
    )
    assert target.kernel_checks("src/fom_ops.c", source, mpc_ir) == []


def test_a_missing_or_unbalanced_routine_is_a_violation(target, mpc_ir):
    source = _ops(mpc_ir).replace("fom_apply_t0_state", "fom_apply_renamed")
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert any("could not find a definition of fom_apply_t0_state" in v for v in violations)
    unbalanced = _ops(mpc_ir, {"t0_state": "\n#if A\n if (x[0] > 0) {\n#else\n {\n#endif\n"})
    violations = target.kernel_checks("src/fom_ops.c", unbalanced, mpc_ir)
    assert any("could not find a definition of fom_apply_t0_state" in v for v in violations)


def test_ops_literal_exemption_is_per_routine(target, mpc_ir):
    # 0.3 sits in another function of the translated file: it exempts that
    # function, not fom_apply_t0_state.
    baseline = _ops(mpc_ir) + "\nfom_float step(void) { return 0.3; }"
    rewrite = _ops(mpc_ir, {"t0_state": "out[0] += 0.3 * x[0];"})
    rewrite += "\nfom_float step(void) { return 0.3; }"
    violations = target.kernel_checks("src/fom_ops.c", rewrite, mpc_ir, baseline)
    assert violations == [
        "fom_apply_t0_state: floating-point literal '0.3' looks like a hardcoded "
        "coefficient; read coefficients from fom_data.c"
    ]
    # ...while a literal the translated routine itself had stays exempt.
    baseline = _ops(mpc_ir, {"t0_state": "out[0] += 0.3 * x[0];"})
    baseline += "\nfom_float step(void) { return 0.3; }"
    assert target.kernel_checks("src/fom_ops.c", rewrite, mpc_ir, baseline) == []
    # A routine's literal does not exempt a new function outside every routine.
    baseline = _ops(mpc_ir, {"t0_state": "out[0] += 0.3 * x[0];"})
    moved = _ops(mpc_ir) + "\nfom_float step(void) { return 0.3; }"
    violations = target.kernel_checks("src/fom_ops.c", moved, mpc_ir, baseline)
    assert len(violations) == 1 and "'0.3'" in violations[0]


def test_braces_in_character_literals_do_not_break_the_rules(target, mpc_ir):
    source = SETUP.replace("(void)settings;", "char c = '}'; (void)c; (void)settings;")
    source = source.replace("(void)params;", "fom_apply_t0_state(work->state, work->acc);")
    violations = target.kernel_checks("src/fom_solver.c", source, mpc_ir)
    assert violations == [
        "fom_apply_t0_state is called from fom_setup / fom_update_params; apply "
        "matrices in the iteration, not to rebuild them at setup"
    ]


def test_a_solver_without_its_setup_functions_is_a_violation(target, mpc_ir):
    source = "fom_int fom_solve(FomWorkspace *work) { return 0; }"
    violations = target.kernel_checks("src/fom_solver.c", source, mpc_ir)
    assert any("could not find a definition of fom_setup" in v for v in violations)
    assert any("could not find a definition of fom_update_params" in v for v in violations)


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
    source = (
        SETUP
        + """
    void g(FomWorkspace *work){
        int i;
        fom_apply_t0_state(work->state, work->acc);
        for (i = 0; i < FOM_MAT_t0_state_ROWS; ++i)
            if (work->t1[i] < FOM_MAT_t1_lo[i]) work->t1[i] = FOM_MAT_t1_lo[i];
    }
    """
    )
    assert target.kernel_checks("src/fom_solver.c", source, mpc_ir) == []


def test_kernel_checks_skip_other_files_and_cuda(target, mpc_ir):
    from fomtools.codegen.config import Backend, Version
    from fomtools.codegen.targets import get_target

    source = "fom_float f(void){ return 0.37 * FOM_MAT_t0_state[0]; }"
    assert target.kernel_checks("src/fom_linsolve.c", source, mpc_ir) == []
    cuda = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.CUDA))
    assert cuda.kernel_checks(cuda.solver_source_path(), source, mpc_ir) == []


def test_literals_already_in_the_translated_file_are_exempt(target, mpc_ir):
    baseline = SETUP + "fom_float eps(void){ return 1e-12; }"
    rewrite = baseline + "\nfom_float c(void){ return 0.37; }"
    violations = target.kernel_checks("src/fom_solver.c", rewrite, mpc_ir, baseline)
    assert len(violations) == 1 and "'0.37'" in violations[0]


def test_new_fom_apply_calls_at_setup_are_flagged(target, mpc_ir):
    source = """
    void fom_setup(FomWorkspace *work, const FomSettings *settings) {
        int j;
        for (j = 0; j < 3; ++j) { if (1) { fom_apply_t0_state(work->state, work->acc); } }
    }
    void fom_update_params(FomWorkspace *work, const FomParams *params) {
        fom_apply_t0_p_p0(work->params.p0, work->acc);
    }
    fom_int fom_solve(FomWorkspace *work) { fom_apply_su_state(work->state, work->acc); return 0; }
    """
    violations = target.kernel_checks("src/fom_solver.c", source, mpc_ir)
    assert any("fom_apply_t0_state is called from fom_setup" in v for v in violations)
    assert any("fom_apply_t0_p_p0 is called from fom_setup" in v for v in violations)
    assert not any("su_state" in v for v in violations)  # in the iteration: fine
    # Already done by the translated solver: exempt.
    assert target.kernel_checks("src/fom_solver.c", source, mpc_ir, source) == []


def test_ops_matrix_data_is_read_only_inside_its_routine(target, mpc_ir):
    # A helper reading another product's data directly is not hidden from the rule.
    helper = (
        "static void kern(const fom_float *x, fom_float *o) { o[0] += FOM_MAT_su_state_VALS[0]; }"
    )
    source = helper + "\n" + _ops(mpc_ir, {"t0_state": "kern(x, out);"})
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert any("FOM_MAT_su_state is read outside fom_apply_su_state" in v for v in violations)
    # The same helper, handed the owning routine's data, is fine.
    helper = (
        "static void kern(const fom_float *v, const fom_float *x, fom_float *o) { o[0] += v[0]; }"
    )
    source = helper + "\n" + _ops(mpc_ir, {"t0_state": "kern(FOM_MAT_t0_state_VALS, x, out);"})
    assert target.kernel_checks("src/fom_ops.c", source, mpc_ir) == []


def test_ops_routine_defined_twice_is_a_violation(target, mpc_ir):
    other = "void fom_apply_t0_state(const fom_float *x, fom_float *out) { out[0] += x[0]; }"
    source = "#if FAST\n" + other + "\n#else\n" + _ops(mpc_ir) + "\n#endif\n"
    violations = target.kernel_checks("src/fom_ops.c", source, mpc_ir)
    assert any("fom_apply_t0_state is defined more than once" in v for v in violations)
