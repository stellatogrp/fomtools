"""Non-quasidefinite affine solve: symmetric-indefinite KKT -> pivoted LU.

Demonstrates that code generation handles linear systems where no-pivot LDL^T is
not the right approach: it falls back to a pivoted LU factorization (never an
inverse) and the generated C still matches the reference oracle.
"""

from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.oracle import run_faithful
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_indefinite_kkt

GOLDEN = Path(__file__).parent / "golden" / "indef_kkt" / "embedded_c" / "fom_solver.c"


def test_indefinite_kkt_lowers_to_lu_not_ldlt():
    case = build_indefinite_kkt()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="indef_kkt")
    fac = ir.factorizations["t0"]
    assert fac.method == "lu"  # symmetric-indefinite, not quasidefinite -> LU
    assert fac.nnz < fac.dim**2  # genuinely sparse factors, not a dense inverse
    # The system really is symmetric and indefinite.
    K = case["data"]["K"]
    assert np.allclose(K, K.T)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() < 0 < eig.max()


def test_indefinite_kkt_oracle_matches_faithful():
    case = build_indefinite_kkt(max_iters=40)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="indef_kkt")
    q = np.full(case["data"]["n"], 0.3)
    res = ReferenceOracle(ir).solve(param_values={"p0": q})
    traj = run_faithful(
        case["verifier"], case["next_state"], {case["param"]: q}, n_iters=ir.termination.max_iters
    )
    assert np.allclose(res.solution, traj[-1], atol=1e-9)


def test_indefinite_kkt_golden_static_checks():
    target = EmbeddedCTarget(build_indefinite_kkt()["config"])
    assert target.static_checks(GOLDEN.read_text()) == []


def test_indefinite_kkt_pipeline_verifies(tmp_path, requires_compiler):
    case = build_indefinite_kkt(max_iters=40)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="indef_kkt",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
