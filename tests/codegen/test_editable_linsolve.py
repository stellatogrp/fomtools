"""The code optimization phase may rewrite the linear solve (``src/fom_linsolve.c``).

A correct rewrite is kept, re-verified, diffed and shipped in ``release/``; a
wrong one is caught by verify's direct check of ``fom_lin_solve_<name>`` and
reverted together with the solver; the translation phase still may not touch it.
Uses the pivoted-LU fixture, whose iteration is one linear solve.
"""

from __future__ import annotations

from pathlib import Path

from fomtools.codegen.agent.prompts import code_optimization_instructions
from fomtools.codegen.agent.stub import StubAgentDriver
from fomtools.codegen.config import Backend, CodegenConfig, Version
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.generate import generate_code
from fomtools.codegen.targets import get_target
from tests.codegen.conftest import build_indefinite_kkt, gate

GOLDEN = Path(__file__).parent / "golden" / "indef_kkt" / "embedded_c" / "fom_solver.c"
LINSOLVE = "src/fom_linsolve.c"
MARK = "/* rewritten by the code optimization phase */"


def _same_solve(text: str) -> str:
    """A rewrite that changes the text but not the solve."""
    return MARK + "\n" + text


def _wrong_solve(text: str) -> str:
    """A rewrite that flips the sign of the first solution entry."""
    head, tail = text.rsplit("}", 1)
    return MARK + "\n" + head + "    x[0] = -x[0];\n}" + tail


def _generate(tmp_path, linsolve_rewrite, *, code_optimization=True):
    case = build_indefinite_kkt(max_iters=40)
    driver = StubAgentDriver(GOLDEN, extra_sources={LINSOLVE: linsolve_rewrite})
    return generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=driver,
        name="indef_kkt",
        code_optimization=code_optimization,
        **gate(case),
    )


def test_a_correct_linear_solve_rewrite_is_kept(tmp_path, requires_compiler):
    result = _generate(tmp_path, _same_solve)
    assert result.success
    assert result.optimization_kept, result.optimize.log
    assert MARK in (tmp_path / LINSOLVE).read_text()
    assert LINSOLVE in result.optimization_diff
    assert tmp_path / LINSOLVE in result.optimize.artifacts
    # The release build compiles the shipped (marker-free) copy of the solve.
    assert MARK in (tmp_path / "release" / "fom_linsolve.c").read_text()
    assert "release/fom_linsolve.c" in (tmp_path / "Makefile").read_text()


def test_a_wrong_linear_solve_is_caught_directly_and_reverted(tmp_path, requires_compiler):
    result = _generate(tmp_path, _wrong_solve)
    assert result.success  # the translated solver still ships
    assert not result.optimization_kept
    assert "fom_lin_solve_t0 differs from its reference solve" in result.optimize.log
    assert MARK not in (tmp_path / LINSOLVE).read_text()


def test_the_translation_phase_may_not_rewrite_the_solve(tmp_path, requires_compiler):
    result = _generate(tmp_path, _same_solve, code_optimization=False)
    assert result.translate.success
    assert MARK not in (tmp_path / LINSOLVE).read_text()


def test_a_division_in_a_rewritten_embedded_solve_is_rejected(tmp_path, requires_compiler):
    def with_division(text):
        head, tail = text.rsplit("}", 1)
        return MARK + "\n" + head + "    x[0] = x[0] / 1.0;\n}" + tail

    result = _generate(tmp_path, with_division)
    assert not result.optimization_kept
    assert "division" in result.optimize.log
    assert MARK not in (tmp_path / LINSOLVE).read_text()


def test_verify_checks_the_solve_only_where_it_is_editable():
    case = build_indefinite_kkt()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="t")
    embedded = get_target(case["config"])
    assert embedded.optimization_editable_paths() == ("src/fom_solver.c", LINSOLVE, "src/fom_ops.c")
    assert "fom_lin_solve_t0(FOM_VERIFY_LS_T0_B[r], x)" in embedded._verify_main(ir)
    assert f"You may edit `src/fom_solver.c` and `{LINSOLVE}`" in code_optimization_instructions(
        ir, embedded
    )

    cuda = get_target(
        CodegenConfig(
            version=Version.LARGE_SCALE,
            backend=Backend.CUDA,
            termination=case["config"].termination,
        )
    )
    assert cuda.optimization_editable_paths() == (cuda.solver_source_path(),)
    assert "FOM_VERIFY_LS_" not in cuda._verify_main(ir)
    assert "Edit ONLY" in code_optimization_instructions(ir, cuda)
