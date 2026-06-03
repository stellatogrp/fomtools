"""Tests for the production variant + optimization report."""

import importlib.util
from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.harness import build_lib
from fomtools.codegen.postprocess import has_profiling_markers, strip_profiling
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"


# --- unit (no compiler) ---------------------------------------------------
def test_strip_profiling_removes_markers_and_include():
    src = (
        '#include "fom_solver.h"\n'
        '#include "fom_profile.h"\n'
        "void f(void){\n"
        "    FOM_PROF_BEGIN(region);\n"
        "    int x = 1;\n"
        "    FOM_PROF_END(region);\n"
        "}\n"
    )
    assert has_profiling_markers(src)
    clean = strip_profiling(src)
    assert "fom_profile.h" not in clean
    assert "FOM_PROF_BEGIN" not in clean
    assert "FOM_PROF_END" not in clean
    assert "int x = 1;" in clean  # real logic preserved
    assert not has_profiling_markers(clean)


def test_strip_profiling_is_noop_without_markers():
    src = '#include "fom_solver.h"\nvoid f(void){ int x = 1; }\n'
    assert strip_profiling(src) == src if src.endswith("\n") else strip_profiling(src)


# --- compiled (production variant builds and is correct) ------------------
def test_production_variant_emitted_and_clean(tmp_path, requires_compiler):
    case = build_ista(max_iters=150)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
    )
    prod = result.production_path
    assert prod is not None and prod.exists()
    text = prod.read_text()
    assert "fom_profile.h" not in text
    assert "FOM_PROF_BEGIN" not in text
    # The instrumented dev source still has the markers.
    assert has_profiling_markers((tmp_path / "src" / "fom_solver.c").read_text())


def test_production_variant_builds_and_matches_oracle(tmp_path, requires_compiler):
    case = build_ista(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
    )
    # `make release` builds libfomsolver from the marker-free source.
    build_lib(tmp_path, make_target="release")
    spec = importlib.util.spec_from_file_location("w", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)
    out = solver.solve(p0=case["data"]["b"])
    expected = oracle.solve(param_values={"p0": case["data"]["b"]})
    assert np.allclose(out["x"], expected.solution, atol=1e-6)


def test_optimization_report_written(tmp_path, requires_compiler):
    case = build_ista(max_iters=150)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
        run_phase_b=True,
        profile=True,
    )
    md = result.optimizations_md
    assert md is not None and md.exists()
    text = md.read_text()
    assert "Optimization report" in text
    assert "Timing" in text
    # The stub makes no source change, so the report should say so.
    assert "No changes" in text or "```diff" in text
    assert result.optimization_diff is not None
