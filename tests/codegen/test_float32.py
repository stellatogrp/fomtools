"""float32 (single-precision) code-generation path.

The dtype is target-independent: the embedded / OpenMP C targets emit
``typedef float fom_float`` just like CUDA would, so the whole float32 pipeline
(including the iterative CG solve the GPU work cares about) is verified here on
the CPU against the reference oracle — no GPU required.

Single precision carries ~1e-7 relative error, so ``tol_verify`` defaults looser
for float32 (1e-4 vs 1e-6) and the CG stopping tolerance is raised to a
single-precision-reachable 1e-6; see ``CodegenConfig`` / ``extract``.
"""

from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir

GOLDEN = Path(__file__).parent / "golden"
ISTA_EMBEDDED_GOLDEN = GOLDEN / "ista" / "embedded_c" / "fom_solver.c"
CG_OPENMP_GOLDEN = GOLDEN / "cgqp" / "large_scale_openmp" / "fom_solver.c"


def test_float32_config_defaults():
    assert CodegenConfig().tol_verify == 1e-6  # float64 default
    assert CodegenConfig(dtype="float32").tol_verify == 1e-4  # looser for f32
    # an explicit tolerance is always honored
    assert CodegenConfig(dtype="float32", tol_verify=1e-5).tol_verify == 1e-5
    with pytest.raises(ValueError):
        CodegenConfig(tol_verify=0)


def test_float32_cg_tol_reachable_in_single_precision():
    # The CG stopping tolerance must be raised for float32 (1e-12 is below single
    # precision); the float64 path keeps the tight default.
    from tests.codegen.conftest import build_spd_cg

    f64 = build_spd_cg(dtype="float64")
    ir64 = extract_ir(f64["verifier"], f64["next_state"], config=f64["config"], name="cgqp")
    assert ir64.factorizations["t0"].cg_tol == 1e-12

    f32 = build_spd_cg(dtype="float32")
    ir32 = extract_ir(f32["verifier"], f32["next_state"], config=f32["config"], name="cgqp")
    assert ir32.factorizations["t0"].cg_tol == 1e-6


def test_float32_emits_float_typedef(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista(dtype="float32")
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="ista")
    scaffold(ir, tmp_path, case["config"])
    header = (tmp_path / "include" / "fom_workspace.h").read_text()
    assert "typedef float fom_float;" in header


def test_float32_ista_embedded_verifies(tmp_path, requires_compiler):
    # End-to-end: build the float32 ISTA solver and check it against the oracle.
    from tests.codegen.conftest import build_ista

    case = build_ista(dtype="float32")
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(ISTA_EMBEDDED_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_float32_cg_openmp_verifies(tmp_path, requires_compiler):
    # End-to-end float32 CG: the iterative solve runs in single precision and
    # still matches the oracle within the (dtype-aware) tolerance.
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(dtype="float32", max_iters=80)
    assert case["config"].dtype == "float32"
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(CG_OPENMP_GOLDEN),
        name="cgqp",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_float32_cg_carries_single_precision_error(tmp_path, requires_compiler):
    # Guard the rationale for a dtype-aware tolerance: the float32 CG solve
    # deviates from the float64 oracle at the single-precision floor (~1e-7), so
    # an over-strict gate rejects it. (1e-9 here is well below that floor; the
    # float32 default of 1e-4 leaves margin for larger / worse-conditioned
    # problems, where single-precision error grows.)
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(dtype="float32", max_iters=80)
    strict = case["config"].model_copy(update={"tol_verify": 1e-9})
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=strict,
        driver=StubAgentDriver(CG_OPENMP_GOLDEN),
        name="cgqp",
    )
    assert not result.translate.success
    assert "VERIFY FAILED" in result.translate.log
