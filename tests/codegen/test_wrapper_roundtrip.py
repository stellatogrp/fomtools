"""Build the generated shared library and drive it from the emitted ctypes wrapper.

Confirms the Python ABI mirror matches the C ABI and that solver output matches
the reference oracle. Requires a C compiler.
"""

import importlib.util
from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.harness import build_lib
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"


def _load_wrapper(out_dir):
    spec = importlib.util.spec_from_file_location(
        "fom_solver_wrapper", out_dir / "python" / "fom_solver.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrapper_matches_oracle(tmp_path, requires_compiler):
    case = build_ista(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log

    build_lib(tmp_path)
    wrapper = _load_wrapper(tmp_path)
    solver = wrapper.Solver()
    oracle = ReferenceOracle(result.ir)

    rng = np.random.default_rng(123)
    for _ in range(5):
        b_val = case["data"]["b"] + rng.uniform(-0.5, 0.5, size=case["data"]["m"])
        out = solver.solve(p0=b_val)
        expected = oracle.solve(param_values={"p0": b_val})
        assert np.allclose(out["x"], expected.solution, atol=1e-6), (out["x"], expected.solution)


def test_wrapper_param_validation(tmp_path, requires_compiler):
    case = build_ista(max_iters=50)
    generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
    )
    build_lib(tmp_path)
    wrapper = _load_wrapper(tmp_path)
    solver = wrapper.Solver()
    import pytest

    with pytest.raises(ValueError):
        solver.solve()  # missing p0
    with pytest.raises(ValueError):
        solver.solve(p0=np.zeros(case["data"]["m"] + 1))  # wrong size
