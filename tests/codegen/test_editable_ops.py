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
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget, _matvec_matrix_names

GOLDEN = Path(__file__).parent / "golden" / "mpc" / "embedded_c" / "fom_solver.c"
OPS = "src/fom_ops.c"
MARK = "/* rewritten by the code optimization phase */"
CONFIG = CodegenConfig(termination=TerminationSpec(max_iters=400))

# A coefficient of t0_state that the 8 sampled end-to-end solves never excite:
# off by 1, the whole-solver check still passes (see the gap test below).
RARE_MATRIX, RARE_COL = "t0_state", 56


@pytest.fixture(scope="module")
def mpc():
    ex = mpc_osqp_example(nx=4, nu=2, N=8)
    return ex, extract_ir(ex["verifier"], ex["next_state"], config=CONFIG, name="mpc")


def _same_ops(text: str) -> str:
    """A rewrite that changes the text but not the products."""
    return MARK + "\n" + text


def _perturbed_ops(name: str, col: int):
    """A rewrite of ``fom_apply_<name>`` with entry ``(0, col)`` off by one.

    Written without a float literal, so it gets past the kernel rules and only
    the numerical checks can catch it."""

    def rewrite(text: str) -> str:
        sig = f"void fom_apply_{name}(const fom_float *x, fom_float *out) {{"
        head, rest = text.split(sig, 1)
        body, tail = rest.split("\n}\n", 1)
        return MARK + "\n" + head + sig + body + f"\n    out[0] += x[{col}];\n}}\n" + tail

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


def test_the_tolerance_scales_with_the_entry_not_the_starting_out(tmp_path, mpc, requires_compiler):
    # Row 6 of `out` starts at 3 and entry (6, 0) of t0_state is zero. An error
    # of 2.5 * tol there must fail: scaled by the starting value it would pass.
    _, ir = mpc
    scaffold(ir, tmp_path, CONFIG)
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())
    sig = "void fom_apply_t0_state(const fom_float *x, fom_float *out) {"
    ops = tmp_path / OPS
    off = 2.5 * CONFIG.tol_verify
    ops.write_text(ops.read_text().replace(sig, sig + f"\n    out[6] += {off!r} * x[0];"))
    proc = _verify(tmp_path)
    assert "fom_apply_t0_state entry (6, 0) is off by" in proc.stdout, proc.stdout


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


def test_a_hardcoded_coefficient_in_ops_is_rejected(tmp_path, mpc, requires_compiler):
    # Numerically right (adds 0.37 * 0), but a literal coefficient is untraceable.
    def with_literal(text: str) -> str:
        return MARK + "\n" + text + "\nfom_float fom_unused_scale(void) { return 0.37 * 0; }\n"

    result = _generate(tmp_path, mpc, with_literal)
    assert not result.optimization_kept
    assert "floating-point literal '0.37'" in result.optimize.log
    assert MARK not in (tmp_path / OPS).read_text()


def test_an_optimized_solver_with_its_own_matvec_is_rejected(tmp_path, mpc, requires_compiler):
    # Same numbers as the golden, but t0_state applied by an inline loop over
    # its CSR data instead of the checked fom_apply_t0_state.
    inline = (
        "{ fom_int i, p; for (i = 0; i < FOM_MAT_t0_state_ROWS; ++i) "
        "for (p = FOM_MAT_t0_state_ROWPTR[i]; p < FOM_MAT_t0_state_ROWPTR[i + 1]; ++p) "
        "work->acc[i] += FOM_MAT_t0_state_VALS[p] * work->state[FOM_MAT_t0_state_COLIDX[p]]; }"
    )
    golden = GOLDEN.read_text()
    call = "fom_apply_t0_state(work->state, work->acc);"
    assert golden.count(call) == 1
    ex, _ = mpc

    class TwoPhaseStub(StubAgentDriver):
        def run(self, task):
            if task.phase == "code_optimization":
                self._source = golden.replace(call, inline)
            return super().run(task)

    result = generate_code(
        ex["verifier"],
        ex["next_state"],
        tmp_path,
        config=CONFIG,
        driver=TwoPhaseStub(GOLDEN),
        name="mpc",
        code_optimization=True,
    )
    assert result.success
    # Numerically the same products, but outside the checked routine: `make verify`
    # flags it while the agent works, and the collection gate rejects it.
    assert "VERIFY FAILED: kernel rule: src/fom_solver.c" in result.optimize.log
    assert not result.optimization_kept
    assert "the solver reads FOM_MAT_t0_state directly" in result.optimize.log
    assert (tmp_path / "src" / "fom_solver.c").read_text() == golden


def test_make_verify_applies_the_kernel_rules_while_configured(tmp_path, mpc, requires_compiler):
    import fomtools.codegen.kernel_rules as kernel_rules

    _, ir = mpc
    target = get_target(CONFIG)
    scaffold(ir, tmp_path, CONFIG)
    script = tmp_path / "test" / "kernel_check.py"
    assert script.read_text() == Path(kernel_rules.__file__).read_text()
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())
    ops = tmp_path / OPS
    ops.write_text(ops.read_text() + "\nfom_float fom_unused(void) { return 0.37; }\n")

    # No config (translation phase, shipped directory): verify ignores the rules.
    assert "VERIFY OK" in _verify(tmp_path).stdout
    # Configured (code optimization phase): the agent's verify reports the rule.
    target.write_kernel_check_config(tmp_path, ir, {OPS: ops.read_text().replace("0.37", "0.0")})
    proc = _verify(tmp_path)
    assert proc.returncode != 0
    assert "VERIFY FAILED: kernel rule: src/fom_ops.c: floating-point literal '0.37'" in proc.stdout
    target.remove_kernel_check_config(tmp_path)
    assert not (tmp_path / "test" / "kernel_check.json").exists()


def test_the_pipeline_gate_decides_even_without_the_verify_step(tmp_path, mpc, requires_compiler):
    # An agent that deletes the config gets past its own verify, but the
    # collection gate applies the same rules and reverts the rewrite.
    ex, _ = mpc

    class DeletesConfig(StubAgentDriver):
        def run(self, task):
            (task.workdir / "test" / "kernel_check.json").unlink(missing_ok=True)
            return super().run(task)

    def with_literal(text: str) -> str:
        return MARK + "\n" + text + "\nfom_float fom_unused(void) { return 0.37; }\n"

    result = generate_code(
        ex["verifier"],
        ex["next_state"],
        tmp_path,
        config=CONFIG,
        driver=DeletesConfig(GOLDEN, extra_sources={OPS: with_literal}),
        name="mpc",
        code_optimization=True,
    )
    assert not result.optimization_kept
    assert "[optimization not kept: src/fom_ops.c: floating-point literal '0.37'" in (
        result.optimize.log
    )
    assert MARK not in (tmp_path / OPS).read_text()
    assert not (tmp_path / "test" / "kernel_check.json").exists()  # nothing shipped
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "### Rule violations" in report
    assert "- [optimization not kept: src/fom_ops.c: floating-point literal '0.37'" in report


# -- the report ----------------------------------------------------------------
def _rewrite_one_routine(name: str):
    """Change the text, not the result, of ``fom_apply_<name>`` only."""

    def rewrite(text: str) -> str:
        sig = f"void fom_apply_{name}(const fom_float *x, fom_float *out) {{"
        head, rest = text.split(sig, 1)
        body, tail = rest.split("\n}\n", 1)
        body = body.replace("fom_float s = 0.0;", "fom_float s = 0;")
        return MARK + "\n" + head + sig + body + "\n}\n" + tail

    return rewrite


def test_parse_ops_checks():
    from fomtools.codegen.postprocess import parse_ops_checks

    out = "noise\nOPSCHECK name=a_b mode=basis probes=7 max_err=0.000e+00\nOPSCHECK name=c mode=random probes=2 max_err=1.5e-16\n"
    assert parse_ops_checks(out) == [
        {"name": "a_b", "mode": "basis", "probes": 7, "max_err": 0.0},
        {"name": "c", "mode": "random", "probes": 2, "max_err": 1.5e-16},
    ]


def test_report_tables_each_routine_and_marks_the_rewritten_one(tmp_path, mpc, requires_compiler):
    _, ir = mpc
    result = _generate(tmp_path, mpc, _rewrite_one_routine("su_state"))
    assert result.optimization_kept, result.optimize.log
    rows = {c["name"]: c for c in result.kernel_checks}
    assert list(rows) == _matvec_matrix_names(ir)
    assert rows["su_state"]["rewritten"] and not rows["t0_state"]["rewritten"]
    md = ir.matrices["su_state"]
    assert rows["su_state"]["shape"] == tuple(md.shape)
    assert rows["su_state"]["nnz"] == md.nnz()
    assert rows["su_state"]["mode"] == "basis"
    assert rows["su_state"]["probes"] == md.shape[1] + 2
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "## Kernel checks" in report
    shape = "x".join(map(str, md.shape))
    assert f"| `su_state` | {shape} | {md.nnz()} | yes | basis | {md.shape[1] + 2} |" in report
    assert "| `t0_state` |" in report
    assert "_None: the collected rewrite passed the rules and was kept._" in report


def test_report_says_when_no_rewrite_was_collected(tmp_path, mpc, requires_compiler):
    result = _generate(tmp_path, mpc, lambda text: text)  # the agent changes nothing
    assert not result.optimization_kept
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "_Not checked: the code optimization phase returned no changed" in report


def test_parse_ops_checks_round_trips_real_verify_output(tmp_path, mpc, requires_compiler):
    # Pins the OPSCHECK format the C harness prints to the parser that reads it.
    from fomtools.codegen.postprocess import parse_ops_checks

    _, ir = mpc
    scaffold(ir, tmp_path, CONFIG)
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())
    proc = _verify(tmp_path)
    assert proc.returncode == 0, proc.stdout
    checks = parse_ops_checks(proc.stdout)
    assert [c["name"] for c in checks] == _matvec_matrix_names(ir)
    for c in checks:
        cols = ir.matrices[c["name"]].shape[1]
        assert c["mode"] == "basis" and c["probes"] == cols + 2 and c["max_err"] == 0.0


def test_a_routine_that_returns_nan_fails_verify(tmp_path, mpc, requires_compiler, monkeypatch):
    # A NaN difference compares false with anything: each check must treat it as
    # a failure, not as an error of zero. Both the direct check and, without it,
    # the end-to-end comparison.
    _, ir = mpc
    scaffold(ir, tmp_path, CONFIG)
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())
    ops = tmp_path / OPS
    nan_ops = ops.read_text().replace("out[i] += s;", "out[i] += s * (s - s) / (s - s);", 1)
    ops.write_text(nan_ops)
    proc = _verify(tmp_path)
    assert proc.returncode != 0 and "VERIFY FAILED: fom_apply_" in proc.stdout

    monkeypatch.setattr(EmbeddedCTarget, "_checked_ops", lambda self, ir: [])
    scaffold(ir, tmp_path, CONFIG)
    ops.write_text(nan_ops)
    proc = _verify(tmp_path)
    assert proc.returncode != 0 and "VERIFY OK" not in proc.stdout
    # printf spells NaN "nan" or "-nan" depending on the libc; the verdict is what counts.
    assert "VERIFY FAILED: 8/8 samples exceed tol" in proc.stdout


def test_report_checks_the_shipped_sources_when_a_rewrite_is_reverted(
    tmp_path, mpc, requires_compiler, monkeypatch
):
    import fomtools.codegen.generate as generate

    runs = []
    real = generate.build_and_verify
    monkeypatch.setattr(generate, "build_and_verify", lambda *a: runs.append(a) or real(*a))
    result = _generate(tmp_path, mpc, _perturbed_ops(RARE_MATRIX, RARE_COL))
    assert not result.optimization_kept
    # The translated (generated) routines ship, and that is what the table shows,
    # read from the translation phase's verify: no extra verify after the phase.
    assert result.kernel_checks and not any(c["rewritten"] for c in result.kernel_checks)
    assert all(c["max_err"] == 0.0 for c in result.kernel_checks)
    assert runs == []


def test_report_has_no_kernel_section_where_ops_are_not_editable(tmp_path):
    from fomtools.codegen.postprocess import write_optimization_report

    path = write_optimization_report(tmp_path, "", "", "", optimize_succeeded=False)
    assert "Kernel checks" not in path.read_text()
