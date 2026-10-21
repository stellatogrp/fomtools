"""Fused products and selections: cross-operator structure the code optimization
phase may exploit, under checks that stay exact.

Each output block with two or more products by per-iteration sources gets a
``fom_fused_<block>`` in ``fom_ops.c``. ``make verify`` checks it against the sum
of its members' references on every unit vector of every source, with every
schedule value at 0 and with each alone at 1 (which recovers each member
exactly), and on random sources and schedule values. A 0/1 selection matrix gets
its index map as package data, and a solver may apply it itself.

Uses the scheduled accelerated lasso, whose slope ``t0_state__d_gamma`` is
``[-A1, A1]`` for ``t0_state = [0, A1]``: the relation the paper's FISTA rewrite
exploited, which no single ``fom_apply_*`` can see.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.codegen.agent.prompts import code_optimization_instructions, translation_instructions
from fomtools.codegen.agent.stub import StubAgentDriver
from fomtools.codegen.emit import scaffold
from fomtools.codegen.generate import extract_fom_ir, generate_code
from fomtools.codegen.ir import MatrixData, selection_description, selection_indices
from fomtools.codegen.targets import get_target
from fomtools.codegen.targets.embedded_c import _fused_blocks
from fomtools.codegen.targets.large_scale_cuda import CudaTarget
from tests.codegen.conftest import build_scheduled

GOLDEN = Path(__file__).parent / "golden" / "scheduled" / "embedded_c" / "fom_solver.c"
OPS = "src/fom_ops.c"
SOLVER = "src/fom_solver.c"

#: The golden t0 assembly, and the same block through the fused routine.
T0_SEPARATE = """\
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] = work->t0[i] = 0.0;
        fom_apply_t0_state(work->state, work->acc);
        fom_apply_t0_state__d_gamma(work->state, work->t0);
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] += gamma * work->t0[i];
"""
T0_FUSED = """\
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] = 0.0;
        fom_fused_t0(work->state, gamma, work->acc);
"""
#: The golden state update, and the same copies written from the selection maps.
SU_SEPARATE = """\
        fom_apply_su_state(work->state, work->new_state);
        fom_apply_su_t_t0(work->t0, work->new_state);
"""
SU_SELECT = """\
        for (i = 0; i < FOM_SEL_su_state_ROWS; ++i)
            if (FOM_SEL_su_state_SRC[i] >= 0) work->new_state[i] += work->state[FOM_SEL_su_state_SRC[i]];
        for (i = 0; i < FOM_SEL_su_t_t0_ROWS; ++i)
            if (FOM_SEL_su_t_t0_SRC[i] >= 0) work->new_state[i] += work->t0[FOM_SEL_su_t_t0_SRC[i]];
"""

#: ``t0_state x + gamma D x`` with ``D = [-A1, A1]`` and ``t0_state = [0, A1]``:
#: one 2x2 product ``A1 (s1 + gamma (s1 - s0))``, reading only ``t0_state``.
FUSED_T0_STRUCTURED = """\
void fom_fused_t0(const fom_float *state, fom_float sc_gamma, fom_float *out) {
    fom_int i, j;
    for (i = 0; i < 2; ++i) {
        fom_float s = 0.0;
        for (j = 0; j < 2; ++j)
            s += FOM_MAT_t0_state[i * 4 + 2 + j] * (state[2 + j] + sc_gamma * (state[2 + j] - state[j]));
        out[i] += s;
    }
}
"""


@pytest.fixture(scope="module")
def case():
    return build_scheduled()


@pytest.fixture(scope="module")
def ir(case):
    return extract_fom_ir(
        case["build"], None, config=case["config"], name="scheduled", schedules=case["schedules"]
    )


def _fused_solver() -> str:
    text = GOLDEN.read_text()
    assert T0_SEPARATE in text and SU_SEPARATE in text
    return text.replace(T0_SEPARATE, T0_FUSED).replace(SU_SEPARATE, SU_SELECT)


def _replace_fused(ops: str, block: str, routine: str) -> str:
    """``ops`` with ``fom_fused_<block>`` replaced by ``routine``."""
    head, rest = ops.split(f"void fom_fused_{block}(", 1)
    _, tail = rest.split("\n}\n", 1)
    return head + routine + tail


def _edit_fused(ops: str, block: str, line: str) -> str:
    """``ops`` with ``line`` added at the end of ``fom_fused_<block>``."""
    head, rest = ops.split(f"void fom_fused_{block}(", 1)
    body, tail = rest.split("\n}\n", 1)
    return head + f"void fom_fused_{block}(" + body + f"\n    {line}\n}}\n" + tail


def _scaffold(tmp_path, ir, case, solver: str | None = None) -> Path:
    scaffold(ir, tmp_path, case["config"])
    (tmp_path / SOLVER).write_text(solver or GOLDEN.read_text())
    return tmp_path


def _verify(out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "verify"], cwd=out_dir, capture_output=True, text=True)


# -- emission -----------------------------------------------------------------
def test_one_fused_routine_per_block_without_parameter_terms(ir):
    blocks = {b.name: b for b in _fused_blocks(ir)}
    assert set(blocks) == {"t0", "su"}
    assert blocks["t0"].members == ["t0_state", "t0_state__d_gamma"]  # not t0_p_p0
    assert blocks["t0"].sources == ("state",) and blocks["t0"].scales == ("gamma",)
    assert blocks["su"].sources == ("state", "tmp:t0") and blocks["su"].scales == ()


def test_the_generated_fused_routines_are_exact_and_follow_the_rules(
    tmp_path, ir, case, requires_compiler
):
    out = _scaffold(tmp_path, ir, case)
    header = (out / "include" / "fom_ops.h").read_text()
    assert (
        "void fom_fused_t0(const fom_float *state, fom_float sc_gamma, fom_float *out);" in header
    )
    assert "out += t0_state @ state + sc_gamma * (t0_state__d_gamma @ state)" in header
    proc = _verify(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # 4 unit vectors x 2 schedule settings + 2 random probes; bitwise equal to the reference.
    assert "OPSCHECK name=fused_t0 mode=basis probes=10 max_err=0.000e+00" in proc.stdout
    assert "OPSCHECK name=fused_su mode=basis probes=8 max_err=0.000e+00" in proc.stdout
    target = get_target(case["config"])
    ops = (out / OPS).read_text()
    assert target.kernel_checks(OPS, ops, ir, ops) == []


def test_a_solver_on_the_fused_routine_and_the_selections_verifies(
    tmp_path, ir, case, requires_compiler
):
    out = _scaffold(tmp_path, ir, case, _fused_solver())
    proc = _verify(out)
    assert proc.returncode == 0 and "VERIFY OK" in proc.stdout, proc.stdout + proc.stderr
    target = get_target(case["config"])
    assert target.kernel_checks(SOLVER, _fused_solver(), ir, GOLDEN.read_text()) == []


def test_a_structured_fused_rewrite_is_kept(tmp_path, case, requires_compiler):
    """The cross-operator rewrite (one 2x2 product for two 2x4 ones) passes the
    exact check and is shipped."""

    def rewrite(ops: str) -> str:
        return _replace_fused(ops, "t0", FUSED_T0_STRUCTURED)

    result = generate_code(
        case["build"],
        None,
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(_fused_solver(), extra_sources={OPS: rewrite}),
        name="scheduled",
        schedules=case["schedules"],
        code_optimization=True,
    )
    assert result.optimization_kept, result.optimize.log
    assert "state[2 + j] - state[j]" in (tmp_path / "release" / "fom_ops.c").read_text()
    rows = {c["name"]: c for c in result.kernel_checks}
    assert rows["fused_t0"]["rewritten"] and rows["fused_t0"]["mode"] == "basis"
    assert not rows["fused_su"]["rewritten"]
    assert rows["fused_t0"]["shape"] == (2, 4)
    assert "| `fused_t0` | 2x4 |" in (tmp_path / "OPTIMIZATIONS.md").read_text()


# -- what the direct check catches ------------------------------------------------
@pytest.mark.parametrize(
    "line, expected",
    [
        # A coefficient off, without a float literal.
        ("out[0] += state[1];", "fom_fused_t0 entry (0, 1) of state is off by"),
        # Right at every schedule value the basis probes use, wrong elsewhere.
        (
            "if (sc_gamma != 0.0 && sc_gamma != 1.0) out[1] += state[0];",
            "fom_fused_t0 differs from its matrices",
        ),
        # Right only while the schedule value is 0 (drops the slope's scaling).
        ("out[0] += sc_gamma * state[0];", "of state is off by"),
    ],
)
def test_a_wrong_fused_routine_fails_verify(tmp_path, ir, case, requires_compiler, line, expected):
    out = _scaffold(tmp_path, ir, case)
    ops = out / OPS
    ops.write_text(_edit_fused(ops.read_text(), "t0", line))
    proc = _verify(out)
    assert proc.returncode != 0 and expected in proc.stdout, proc.stdout


def test_a_structured_rewrite_with_the_wrong_relation_fails_verify(
    tmp_path, ir, case, requires_compiler
):
    out = _scaffold(tmp_path, ir, case)
    ops = out / OPS
    wrong = FUSED_T0_STRUCTURED.replace("state[2 + j] - state[j]", "state[2 + j] + state[j]")
    ops.write_text(_replace_fused(ops.read_text(), "t0", wrong))
    proc = _verify(out)
    assert proc.returncode != 0
    assert "fom_fused_t0 entry" in proc.stdout and "sc_gamma = 1" in proc.stdout


def test_a_wrong_selection_copy_fails_the_end_to_end_check(tmp_path, ir, case, requires_compiler):
    # Selections are checked end to end only: a wrong index still has to fail there.
    solver = _fused_solver().replace(
        "work->state[FOM_SEL_su_state_SRC[i]]", "work->state[FOM_SEL_su_state_SRC[i] - 1]"
    )
    proc = _verify(_scaffold(tmp_path, ir, case, solver))
    assert proc.returncode != 0 and "VERIFY OK" not in proc.stdout


# -- what the kernel rules catch ----------------------------------------------------
@pytest.fixture()
def generated_ops(tmp_path, ir, case):
    scaffold(ir, tmp_path, case["config"])
    return (tmp_path / OPS).read_text()


def _rules(case, ir, rel, source, baseline):
    return get_target(case["config"]).kernel_checks(rel, source, ir, baseline)


def test_a_fused_routine_may_read_only_its_members(case, ir, generated_ops):
    bad = _edit_fused(generated_ops, "t0", "out[0] += FOM_MAT_su_state_VALS[0] * state[0];")
    assert _rules(case, ir, OPS, bad, generated_ops) == [
        "fom_fused_t0 reads FOM_MAT_su_state, the data of another product; each routine "
        "(and every helper it calls) may read only its own matrix (a fused routine: its "
        "members')"
    ]


@pytest.mark.parametrize("read", ["fom_sched_gamma(0)", "FOM_SCHED_gamma[1]"])
def test_ops_may_not_read_the_schedule(case, ir, generated_ops, read):
    bad = _edit_fused(generated_ops, "t0", f"if (sc_gamma == {read}) out[0] += state[0];")
    violations = _rules(case, ir, OPS, bad, generated_ops)
    assert any("fom_ops.c reads the schedule" in v for v in violations), violations


def test_a_fused_routine_may_not_add_coefficients(case, ir, generated_ops):
    bad = _edit_fused(generated_ops, "t0", "out[0] += 0.20096742 * state[2];")
    violations = _rules(case, ir, OPS, bad, generated_ops)
    assert any(
        v.startswith("fom_fused_t0: floating-point literal '0.20096742'") for v in violations
    )


def test_the_fused_routine_must_stay_findable(case, ir, generated_ops):
    gone = generated_ops.replace("void fom_fused_su(", "static void renamed_su(")
    violations = _rules(case, ir, OPS, gone, generated_ops)
    assert any("could not find a definition of fom_fused_su" in v for v in violations)


def test_setup_may_not_call_a_fused_routine(case, ir):
    golden = GOLDEN.read_text()
    probing = golden.replace(
        "void fom_setup(", "static fom_float probe[2];\nvoid fom_setup(", 1
    ).replace(
        "work->settings = *settings;",
        "work->settings = *settings;\n    fom_fused_t0(work->state, 1.0, probe);",
        1,
    )
    assert probing != golden
    violations = _rules(case, ir, SOLVER, probing, golden)
    assert any("fom_fused_t0 is called from fom_setup" in v for v in violations), violations


def test_the_solver_may_read_selection_maps_but_not_matrix_data(case, ir):
    golden = GOLDEN.read_text()
    assert _rules(case, ir, SOLVER, _fused_solver(), golden) == []
    reads = _fused_solver().replace(
        "FOM_SEL_su_state_SRC[i] >= 0)",
        "FOM_SEL_su_state_SRC[i] >= 0 && FOM_MAT_su_state_VALS[0])",
        1,
    )
    assert any(
        "reads FOM_MAT_su_state directly" in v for v in _rules(case, ir, SOLVER, reads, golden)
    )


# -- selections -------------------------------------------------------------------
def _md(dense) -> MatrixData:
    return MatrixData.from_coeff("m", spa.csr_matrix(np.asarray(dense, dtype=float)))


@pytest.mark.parametrize(
    "dense, src, description",
    [
        (np.eye(3), [0, 1, 2], "identity"),
        (
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [2, 3, -1, -1],
            "shift: out[i] += x[i + 2] for rows 0..1, 2 empty rows (left unchanged)",
        ),
        ([[0, 1, 0], [1, 0, 0], [0, 0, 1]], [1, 0, 2], "gather: out[i] += x[src[i]] on 3 rows"),
    ],
)
def test_selections_are_certified(dense, src, description):
    found = selection_indices(_md(dense))
    assert found is not None and found.tolist() == src
    assert selection_description(found, np.asarray(dense).shape[1]) == description


@pytest.mark.parametrize(
    "dense",
    [
        [[2.0, 0.0], [0.0, 1.0]],  # a scaled entry
        [[1.0, 1.0], [0.0, 1.0]],  # two entries in a row
        [[1.0 + 1e-12, 0.0], [0.0, 1.0]],  # not exactly one
        [[-1.0, 0.0], [0.0, 1.0]],  # a sign
    ],
)
def test_non_selections_are_not_certified(dense):
    assert selection_indices(_md(dense)) is None


def test_selection_maps_are_package_data_and_documented(tmp_path, ir, case):
    scaffold(ir, tmp_path, case["config"])
    data = (tmp_path / "src" / "fom_data.c").read_text()
    assert "const fom_int FOM_SEL_su_state_SRC[4] = {2, 3, -1, -1};" in data
    assert "const fom_int FOM_SEL_su_t_t0_SRC[4] = {-1, -1, 0, 1};" in data
    assert "FOM_SEL_t0_state" not in data  # a dense coefficient block is not a selection
    spec_md = (tmp_path / "SPEC.md").read_text()
    assert "## Selection matrices" in spec_md and "`su_state`: shift" in spec_md


# -- prompts and targets ------------------------------------------------------------
def test_the_optimization_prompt_explains_fused_routines_and_selections(ir, case):
    target = get_target(case["config"])
    prompt = code_optimization_instructions(ir, target)
    assert "`fom_fused_t0`, `fom_fused_su`" in prompt
    assert "must be exact for\nANY schedule value" in prompt
    assert "`su_state`: shift" in prompt and "FOM_SEL_<name>_SRC" in prompt
    assert "or `fom_fused_*`" in prompt
    assert "fom_fused" not in translation_instructions(ir, target)


def test_cuda_has_no_fused_routines_or_selection_data(tmp_path, ir, case, monkeypatch):
    for gpu in (False, True):
        monkeypatch.setattr(CudaTarget, "_can_run_on_gpu", staticmethod(lambda gpu=gpu: gpu))
        target = CudaTarget(case["config"])
        assert target._fused(ir) == [] and target._selection_data(ir) == {}
        assert target.fused_routines(ir) == {} and target.selection_matrices(ir) == {}
        out = tmp_path / str(gpu)
        target.scaffold(ir, out)
        assert "fom_fused" not in (out / "include" / "fom_ops.h").read_text()
        assert "FOM_SEL_" not in (out / "include" / "fom_data.h").read_text()
        assert "fom_fused" not in code_optimization_instructions(ir, target)
