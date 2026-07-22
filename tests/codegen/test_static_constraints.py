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
