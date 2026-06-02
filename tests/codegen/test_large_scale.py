"""Tests for the large-scale (OpenMP / CUDA) targets.

OpenMP is built serially and numerically verified against the oracle (portable).
CUDA is compile-checked only and skipped without ``nvcc`` (no GPU in CI).
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.config import Backend, CodegenConfig, TerminationSpec, Version
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.targets import get_target

GOLDEN_ROOT = Path(__file__).parent / "golden" / "ista"
OPENMP_GOLDEN = GOLDEN_ROOT / "large_scale_openmp" / "fom_solver.c"
CUDA_GOLDEN = GOLDEN_ROOT / "large_scale_cuda" / "fom_solver.cu"


def _openmp_config(max_iters=200):
    return CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.OPENMP,
        termination=TerminationSpec(max_iters=max_iters),
    )


def _cuda_config(max_iters=200):
    return CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.CUDA,
        termination=TerminationSpec(max_iters=max_iters),
    )


def test_embedded_cuda_combo_rejected():
    with pytest.raises(ValueError):
        CodegenConfig(version=Version.EMBEDDED, backend=Backend.CUDA)


def test_openmp_scaffold(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=_openmp_config(), name="ista")
    target = scaffold(ir, tmp_path, _openmp_config())
    assert target.name == "large_scale_openmp"
    assert "OMPFLAGS" in (tmp_path / "Makefile").read_text()
    assert (tmp_path / "include" / "fom_solver.h").exists()


def test_cuda_scaffold(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=_cuda_config(), name="ista")
    target = scaffold(ir, tmp_path, _cuda_config())
    assert target.name == "large_scale_cuda"
    assert target.solver_source_path().endswith(".cu")
    mk = (tmp_path / "Makefile").read_text()
    assert "nvcc" in mk


def test_openmp_golden_verifies(tmp_path, requires_compiler):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_openmp_config(),
        driver=StubAgentDriver(OPENMP_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_openmp_relaxed_static_checks():
    # Large-scale targets do not enforce the embedded no-malloc/division rules.
    target = get_target(_openmp_config())
    assert (
        target.static_checks("void f(void){ double *p = malloc(8); double y = p[0] / 2.0; }") == []
    )


@pytest.mark.skipif(not shutil.which("nvcc"), reason="no CUDA toolkit (nvcc) available")
def test_cuda_compile_check(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_cuda_config(),
        driver=StubAgentDriver(CUDA_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log


def test_openmp_wrapper_matches_oracle(tmp_path, requires_compiler):
    import importlib.util

    from fomtools.codegen import ReferenceOracle
    from fomtools.codegen.harness import build_lib
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_openmp_config(),
        driver=StubAgentDriver(OPENMP_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    build_lib(tmp_path)
    spec = importlib.util.spec_from_file_location("w_omp", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)
    out = solver.solve(p0=case["data"]["b"])
    expected = oracle.solve(param_values={"p0": case["data"]["b"]})
    assert np.allclose(out["x"], expected.solution, atol=1e-6)
