"""The code optimization phase may rewrite the matrix-vector products (``src/fom_ops.c``).

Each ``fom_apply_<name>`` is checked directly by ``make verify`` against a
package-written reference over the same ``fom_data.c`` arrays: on every unit
vector (which recovers the routine's matrix column by column) when that fits the
work budget, and on random vectors. A correct rewrite is kept and shipped in
``release/``; a wrong coefficient is reported by entry and reverted, including
one the sampled end-to-end check alone does not catch. Uses the OSQP MPC
example, whose sparse blocks are emitted as CSR and applied via ``fom_apply_*``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fomtools.codegen.agent.prompts import code_optimization_instructions
from fomtools.codegen.agent.stub import StubAgentDriver
from fomtools.codegen.config import Backend, CodegenConfig, TerminationSpec, Version
from fomtools.codegen.emit import scaffold
from fomtools.codegen.examples import mpc_osqp_example
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.generate import generate_code
from fomtools.codegen.targets import get_target
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget

GOLDEN = Path(__file__).parent / "golden" / "mpc" / "embedded_c" / "fom_solver.c"
OPS = "src/fom_ops.c"
MARK = "/* rewritten by the code optimization phase */"
CONFIG = CodegenConfig(termination=TerminationSpec(max_iters=400))

# A coefficient of t0_state that the 8 sampled end-to-end solves never excite:
# off by 1e-2, the whole-solver check still passes (see the gap test below).
RARE_MATRIX, RARE_COL = "t0_state", 56


@pytest.fixture(scope="module")
def mpc():
    ex = mpc_osqp_example(nx=4, nu=2, N=8)
    return ex, extract_ir(ex["verifier"], ex["next_state"], config=CONFIG, name="mpc")


def _same_ops(text: str) -> str:
    """A rewrite that changes the text but not the products."""
    return MARK + "\n" + text


def _perturbed_ops(name: str, col: int, delta: float = 1e-2):
    """A rewrite of ``fom_apply_<name>`` with entry ``(0, col)`` off by ``delta``."""

    def rewrite(text: str) -> str:
        sig = f"void fom_apply_{name}(const fom_float *x, fom_float *out) {{"
        head, rest = text.split(sig, 1)
        body, tail = rest.split("\n}\n", 1)
        return (
            MARK + "\n" + head + sig + body + f"\n    out[0] += {delta!r} * x[{col}];\n}}\n" + tail
        )

    return rewrite


def _overwriting_ops(name: str):
    """A rewrite of ``fom_apply_<name>`` that assigns ``out`` instead of accumulating."""

    def rewrite(text: str) -> str:
        sig = f"void fom_apply_{name}(const fom_float *x, fom_float *out) {{"
        head, rest = text.split(sig, 1)
        body, tail = rest.split("\n}\n", 1)
        return (
            MARK + "\n" + head + sig + body.replace("out[i] += s;", "out[i] = s;") + "\n}\n" + tail
        )

    return rewrite


def _generate(tmp_path, mpc, ops_rewrite, *, code_optimization=True):
    ex, _ = mpc
    driver = StubAgentDriver(GOLDEN, extra_sources={OPS: ops_rewrite})
    return generate_code(
        ex["verifier"],
        ex["next_state"],
        tmp_path,
        config=CONFIG,
        driver=driver,
        name="mpc",
        code_optimization=code_optimization,
    )


def _verify(out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "verify"], cwd=out_dir, capture_output=True, text=True)


def test_a_correct_ops_rewrite_is_kept(tmp_path, mpc, requires_compiler):
    result = _generate(tmp_path, mpc, _same_ops)
    assert result.success
    assert result.optimization_kept, result.optimize.log
    assert MARK in (tmp_path / OPS).read_text()
    assert OPS in result.optimization_diff
    assert MARK in (tmp_path / "release" / "fom_ops.c").read_text()
    assert "release/fom_ops.c" in (tmp_path / "Makefile").read_text()
    # Every routine was probed on all of its unit vectors (all fit the budget).
    for name in ("t0_state", "su_t_t1"):
        assert f"OPSCHECK name={name} mode=basis" in result.optimize.log


def test_a_wrong_coefficient_is_reported_by_entry_and_reverted(tmp_path, mpc, requires_compiler):
    result = _generate(tmp_path, mpc, _perturbed_ops(RARE_MATRIX, RARE_COL))
    assert result.success  # the translated solver still ships
    assert not result.optimization_kept
    assert f"fom_apply_{RARE_MATRIX} entry (0, {RARE_COL}) is off by" in result.optimize.log
    assert MARK not in (tmp_path / OPS).read_text()


def test_overwriting_instead_of_accumulating_is_caught(tmp_path, mpc, requires_compiler):
    result = _generate(tmp_path, mpc, _overwriting_ops("su_t_t0"))
    assert not result.optimization_kept
    assert "fom_apply_su_t_t0 entry" in result.optimize.log


def test_the_end_to_end_check_alone_misses_the_rare_coefficient(
    tmp_path, mpc, requires_compiler, monkeypatch
):
    # Documents the gap the direct check closes: without it, the same wrong
    # coefficient passes the sampled whole-solver comparison.
    _, ir = mpc
    monkeypatch.setattr(EmbeddedCTarget, "_checked_ops", lambda self, ir: [])
    scaffold(ir, tmp_path, CONFIG)
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())
    ops = tmp_path / OPS
    ops.write_text(_perturbed_ops(RARE_MATRIX, RARE_COL)(ops.read_text()))
    proc = _verify(tmp_path)
    assert proc.returncode == 0 and "VERIFY OK" in proc.stdout


def test_random_probes_check_a_matrix_over_the_budget(
    tmp_path, mpc, requires_compiler, monkeypatch
):
    _, ir = mpc
    monkeypatch.setattr(EmbeddedCTarget, "_OPS_PROBE_BUDGET", 0)
    scaffold(ir, tmp_path, CONFIG)
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())
    proc = _verify(tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert "OPSCHECK name=t0_state mode=random probes=2" in proc.stdout
    ops = tmp_path / OPS
    ops.write_text(_perturbed_ops("su_t_t1", 3)(ops.read_text()))
    proc = _verify(tmp_path)
    assert "fom_apply_su_t_t1 differs from its matrix" in proc.stdout


def test_the_check_stores_no_matrix_sized_data(tmp_path, mpc):
    # Only O(cols) random probes per matrix go into verify_data.h; the reference
    # is computed in C from fom_data.c, so no outputs or columns are stored.
    _, ir = mpc
    scaffold(ir, tmp_path, CONFIG)
    data = (tmp_path / "test" / "verify_data.h").read_text()
    md = ir.matrices["su_state"]
    assert f"FOM_VERIFY_OP_SU_STATE_X[2][{md.shape[1]}]" in data
    ref = (tmp_path / "test" / "ops_ref.c").read_text()
    assert "static const" not in ref and "fom_ref_apply_su_state" in ref


def test_the_translation_phase_may_not_rewrite_ops(tmp_path, mpc, requires_compiler):
    result = _generate(tmp_path, mpc, _same_ops, code_optimization=False)
    assert result.translate.success
    assert MARK not in (tmp_path / OPS).read_text()


def test_ops_are_checked_and_editable_only_off_cuda(mpc):
    _, ir = mpc
    embedded = get_target(CONFIG)
    assert OPS in embedded.optimization_editable_paths()
    assert "fom_ref_apply_t0_state(x, ref)" in embedded._verify_main(ir)
    assert f"`{OPS}` implements each `fom_apply_<name>(x, out)`" in (
        code_optimization_instructions(ir, embedded)
    )

    openmp = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.OPENMP))
    assert OPS in openmp.optimization_editable_paths()
    assert "$(VERIFY_SRC)" in openmp._makefile()

    cuda = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.CUDA))
    assert cuda.optimization_editable_paths() == (cuda.solver_source_path(),)
    assert "fom_ref_apply_" not in cuda._verify_main(ir)
    assert "Edit ONLY" in code_optimization_instructions(ir, cuda)
