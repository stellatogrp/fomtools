"""End-to-end pipeline + wrapper tests for Douglas-Rachford (golden, no LLM)."""

import importlib.util
from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.harness import build_lib
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_dr

GOLDEN = Path(__file__).parent / "golden" / "dr" / "embedded_c" / "fom_solver.c"


def test_dr_golden_static_checks():
    target = EmbeddedCTarget(build_dr()["config"])
    assert target.static_checks(GOLDEN.read_text()) == []


def test_dr_stub_pipeline_verifies(tmp_path, requires_compiler):
    case = build_dr(max_iters=300)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="dr",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_dr_wrapper_matches_oracle(tmp_path, requires_compiler):
    case = build_dr(max_iters=300)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="dr",
    )
    assert result.translate.success, result.translate.log

    build_lib(tmp_path)
    spec = importlib.util.spec_from_file_location("w_dr", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)

    n = case["data"]["n"]
    rng = np.random.default_rng(7)
    for _ in range(4):
        q = np.hstack([case["data"]["c"] + rng.uniform(-0.1, 0.1, size=n), case["data"]["b"]])
        out = solver.solve(p0=q)
        expected = oracle.solve(param_values={"p0": q})
        assert np.allclose(out["x"], expected.solution, atol=1e-6)
