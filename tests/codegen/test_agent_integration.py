"""Real-agent integration test.

This exercises the actual Claude Agent SDK driver writing a solver from scratch
and iterating until ``make verify`` passes. It is nondeterministic, needs auth,
and is therefore gated behind ``FOMTOOLS_CODEGEN_AGENT=1`` and the ``agent``
marker; it never runs in CI.

Run locally with:
    FOMTOOLS_CODEGEN_AGENT=1 pytest tests/codegen/test_agent_integration.py -m agent
"""

import os
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.agent

_ENABLED = os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1"
_HAS_CC = bool(shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"))


@pytest.mark.skipif(not _ENABLED, reason="set FOMTOOLS_CODEGEN_AGENT=1 to run the real agent")
@pytest.mark.skipif(not _HAS_CC, reason="no C compiler available")
def test_real_agent_translates_ista(tmp_path):
    from fomtools.codegen import ReferenceOracle, generate_code
    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver
    from fomtools.codegen.harness import build_lib
    from tests.codegen.conftest import build_ista

    case = build_ista(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=ClaudeAgentDriver(),
        name="ista",
    )
    assert result.translate.success, result.translate.log

    # Independent confirmation via the Python wrapper.
    build_lib(tmp_path)
    import importlib.util

    spec = importlib.util.spec_from_file_location("w", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)
    out = solver.solve(p0=case["data"]["b"])
    expected = oracle.solve(param_values={"p0": case["data"]["b"]})
    assert np.allclose(out["x"], expected.solution, atol=1e-6)
