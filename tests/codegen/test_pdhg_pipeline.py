"""End-to-end pipeline + wrapper tests for plain PDHG-on-LP (golden, no LLM)."""

import importlib.util
from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.harness import build_lib
from fomtools.codegen.oracle import run_faithful
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_pdhg_lp, pdhg_lp_reference

GOLDEN = Path(__file__).parent / "golden" / "pdhg_lp" / "embedded_c" / "fom_solver.c"


def test_pdhg_extracts_two_projection_steps():
    case = build_pdhg_lp()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="pdhg_lp")
    kinds = [s.activation.kind.value for s in ir.body]
    assert kinds == ["saturated_linear", "relu"]  # primal box, then dual y>=0
    assert all(s.solve is None for s in ir.body)  # pure first-order: no linear solve
    assert not ir.factorizations
    assert ir.state.dim == case["data"]["n"] + case["data"]["m"]


def test_pdhg_ir_matches_faithful_and_reference():
    # IR extraction preserves the algorithm: rolled oracle == step-level run ==
    # the independent NumPy PDHG reference.
    case = build_pdhg_lp(max_iters=150)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="pdhg_lp")
    d = case["data"]
    cval = d["c_nom"]
    oracle = ReferenceOracle(ir).solve(param_values={"p0": cval})

    faithful = run_faithful(
        case["verifier"],
        case["next_state"],
        {case["param"]: cval},
        n_iters=ir.termination.max_iters,
    )
    assert np.allclose(oracle.solution, faithful[-1], atol=1e-9)

    ref = pdhg_lp_reference(d, cval, np.zeros(d["n"] + d["m"]), ir.termination.max_iters)[-1]
    assert np.allclose(oracle.solution, ref, atol=1e-9)


def test_pdhg_golden_static_checks():
    # The PDHG golden is allocation- and division-free (embedded invariants).
    target = EmbeddedCTarget(build_pdhg_lp()["config"])
    assert target.static_checks(GOLDEN.read_text()) == []


def test_pdhg_stub_pipeline_verifies(tmp_path, requires_compiler):
    case = build_pdhg_lp(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="pdhg_lp",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_pdhg_wrapper_matches_oracle(tmp_path, requires_compiler):
    case = build_pdhg_lp(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="pdhg_lp",
    )
    assert result.translate.success, result.translate.log

    build_lib(tmp_path)
    spec = importlib.util.spec_from_file_location("w_pdhg", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)

    rng = np.random.default_rng(11)
    for _ in range(4):
        c = case["data"]["c_nom"] + rng.uniform(-0.1, 0.1, size=case["data"]["n"])
        out = solver.solve(p0=c)
        expected = oracle.solve(param_values={"p0": c})
        assert np.allclose(out["x"], expected.solution, atol=1e-6)
