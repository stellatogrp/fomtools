"""Step and iterate routines, and selection rows: whole blocks and whole
iterations the code optimization phase may rewrite, under checks that stay exact.

``fom_step_<block>`` computes ``out = act(bias + sum_k s_k M_k src_k)`` and
overwrites ``out``; ``make verify`` switches the activation off (which recovers
every member, unit vector by unit vector) and then checks it at the package
parameters and at random ones. ``fom_iterate`` chains the steps of one
iteration and is checked on every temporary and the new state. Rows of a block
that copy one source entry are published as ``FOM_SELROWS_<block>_*``; the
block's ``__rest`` routine computes the other rows and must leave those alone.

Uses PDHG for an LP (saturated-linear and relu steps), the scheduled lasso
(soft threshold, a schedule) and Douglas-Rachford (a state update whose first
rows are copies and whose last are not).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fomtools.codegen import Backend, CodegenConfig, Version
from fomtools.codegen.agent.prompts import code_optimization_instructions, translation_instructions
from fomtools.codegen.agent.stub import StubAgentDriver
from fomtools.codegen.emit import scaffold
from fomtools.codegen.generate import extract_fom_ir, generate_code
from fomtools.codegen.targets import get_target
from fomtools.codegen.targets.embedded_c import _selection_rows, _step_blocks
from tests.codegen.conftest import build_dr, build_pdhg_lp, build_scheduled

GOLDEN = Path(__file__).parent / "golden"
OPS = "src/fom_ops.c"
SOLVER = "src/fom_solver.c"

CASES = {
    "pdhg": (build_pdhg_lp, "pdhg_lp"),
    "scheduled": (build_scheduled, "scheduled"),
    "dr": (build_dr, "dr"),
}


@pytest.fixture(scope="module")
def cases():
    out = {}
    for key, (build, gold) in CASES.items():
        case = build()
        kw = dict(schedules=case["schedules"]) if "schedules" in case else {}
        source = case.get("build", case.get("verifier"))
        ir = extract_fom_ir(source, case.get("next_state"), config=case["config"], name=gold, **kw)
        out[key] = (case, ir, (GOLDEN / gold / "embedded_c" / "fom_solver.c").read_text())
    return out


def _scaffold(tmp_path, cases, key, solver: str | None = None) -> Path:
    case, ir, golden = cases[key]
    scaffold(ir, tmp_path, case["config"])
    (tmp_path / SOLVER).write_text(solver or golden)
    return tmp_path


def _verify(out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "verify"], cwd=out_dir, capture_output=True, text=True)


def _edit(ops: str, fn: str, old: str, new: str) -> str:
    """``ops`` with ``old`` replaced by ``new`` inside the definition of ``fn``."""
    head, rest = ops.split(f"void {fn}(", 1)
    body, tail = rest.split("\n}\n", 1)
    assert old in body, (fn, old)
    return head + f"void {fn}(" + body.replace(old, new, 1) + "\n}\n" + tail


def _rules(cases, key, rel, source, baseline):
    case, ir, _ = cases[key]
    return get_target(case["config"]).kernel_checks(rel, source, ir, baseline)


#: PDHG with the whole iteration in one call: the parameter term is the primal
#: step's bias (computed once per iteration here, still hoistable), the dual
#: step's bias is its constant, and the state update has none.
PDHG_ITERATE = """\
        FOM_PROF_BEGIN(iterate);
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] = 0.0;
        fom_apply_t0_p_p0(work->params.p0, work->acc);
        fom_iterate(work->state, work->acc, FOM_MAT_t1_c, FOM_ZERO_SU, &FOM_ACT_t0, &FOM_ACT_t1,
                    work->t0, work->t1, work->new_state);
        FOM_PROF_END(iterate);
"""

#: Douglas-Rachford's state update: rows 0..3 copy t0 (written by the solver from
#: the selection rows), the rest through the block's __rest routine.
DR_REST = """\
        FOM_PROF_BEGIN(state_update);
        for (i = 0; i < FOM_SELROWS_su_N; ++i) {
            const fom_float *src = FOM_SELROWS_su_SRC[i] == 0 ? work->state
                                 : FOM_SELROWS_su_SRC[i] == 1 ? work->t0 : work->t1;
            work->new_state[FOM_SELROWS_su_ROW[i]] = src[FOM_SELROWS_su_COL[i]OFFSET];
        }
        fom_step_su__rest(work->state, work->t0, work->t1, FOM_ZERO_SU, work->new_state);
        FOM_PROF_END(state_update);
"""


def _pdhg_iterate_solver(golden: str) -> str:
    start = golden.index("        /* primal step")
    end = golden.index("        FOM_PROF_END(state_update);\n") + len(
        "        FOM_PROF_END(state_update);\n"
    )
    solver = golden[:start] + PDHG_ITERATE + golden[end:]
    return solver.replace(
        '#include "fom_solver.h"',
        '#include "fom_solver.h"\n\nstatic const fom_float FOM_ZERO_SU[FOM_STATE_DIM];',
        1,
    )


def _dr_rest_solver(golden: str, offset: str = "") -> str:
    start = golden.index("        /* state update */\n")
    end = golden.index("        FOM_PROF_END(state_update);\n") + len(
        "        FOM_PROF_END(state_update);\n"
    )
    solver = golden[:start] + DR_REST.replace("OFFSET", offset) + golden[end:]
    return solver.replace(
        '#include "fom_solver.h"',
        '#include "fom_solver.h"\n\nstatic const fom_float FOM_ZERO_SU[FOM_STATE_DIM];',
        1,
    )


# -- emission -----------------------------------------------------------------
def test_step_blocks_carry_the_activation(cases):
    _, ir, _ = cases["pdhg"]
    blocks = {b.name: b for b in _step_blocks(ir)}
    assert set(blocks) == {"t0", "t1", "su"}
    assert blocks["t0"].act.kind.value == "saturated_linear" and blocks["t0"].members == [
        "t0_state"
    ]
    assert blocks["t1"].act.kind.value == "relu" and blocks["t1"].sources == ("state", "tmp:t0")
    assert blocks["su"].act is None


def test_step_and_iterate_routines_are_declared(tmp_path, cases):
    out = _scaffold(tmp_path, cases, "pdhg")
    header = (out / "include" / "fom_ops.h").read_text()
    assert (
        "void fom_step_t0(const fom_float *state, const fom_float *bias, "
        "const FomAct_t0 *act, fom_float *out);" in header
    )
    assert "typedef struct { const fom_float *lo; const fom_float *hi; } FomAct_t0;" in header
    assert "void fom_iterate(const fom_float *state, const fom_float *bias_t0," in header
    data = (out / "src" / "fom_data.c").read_text()
    assert "const FomAct_t0 FOM_ACT_t0 = {FOM_MAT_t0_lo, FOM_MAT_t0_hi};" in data
    spec = (out / "SPEC.md").read_text()
    assert "## Selection rows" in spec and "rows 0..2 copy t_t0[0..2]" in spec


@pytest.mark.parametrize("key", list(CASES))
def test_the_generated_routines_are_exact_and_follow_the_rules(
    tmp_path, cases, key, requires_compiler
):
    out = _scaffold(tmp_path, cases, key)
    proc = _verify(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    case, ir, _ = cases[key]
    target = get_target(case["config"])
    for fn in [*target.step_routines(ir), *target.rest_routines(ir)]:
        assert f"OPSCHECK name={fn[4:]} mode=basis" in proc.stdout
    if target.iterate_routine(ir):
        assert "OPSCHECK name=iterate mode=basis" in proc.stdout
    for line in proc.stdout.splitlines():
        if line.startswith(("OPSCHECK name=step_", "OPSCHECK name=iterate")):
            assert line.endswith("max_err=0.000e+00"), line
    ops = (out / OPS).read_text()
    assert target.kernel_checks(OPS, ops, ir, ops) == []


def test_a_solver_on_fom_iterate_verifies(tmp_path, cases, requires_compiler):
    _, _, golden = cases["pdhg"]
    solver = _pdhg_iterate_solver(golden)
    out = _scaffold(tmp_path, cases, "pdhg", solver)
    proc = _verify(out)
    assert proc.returncode == 0 and "VERIFY OK" in proc.stdout, proc.stdout + proc.stderr
    assert _rules(cases, "pdhg", SOLVER, solver, golden) == []


def test_a_rewritten_step_with_the_activation_folded_in_is_kept(tmp_path, cases, requires_compiler):
    case, _, golden = cases["pdhg"]

    def rewrite(ops: str) -> str:
        return _edit(
            ops,
            "fom_step_t1",
            "if (v < act->lo[i]) v = act->lo[i];",
            "v = (v < act->lo[i]) ? act->lo[i] : v;",
        )

    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(_pdhg_iterate_solver(golden), extra_sources={OPS: rewrite}),
        name="pdhg_lp",
        code_optimization=True,
    )
    assert result.optimization_kept, result.optimize.log
    rows = {c["name"]: c for c in result.kernel_checks}
    assert rows["step_t1"]["rewritten"] and rows["step_t1"]["mode"] == "basis"
    assert not rows["step_t0"]["rewritten"]
    assert rows["iterate"]["shape"] == (5, 5)
    assert "| `step_t1` |" in (tmp_path / "OPTIMIZATIONS.md").read_text()


def test_a_solver_writing_its_selection_rows_verifies(tmp_path, cases, requires_compiler):
    _, ir, golden = cases["dr"]
    assert [
        r for r, _, _ in _selection_rows(ir, next(b for b in _step_blocks(ir) if b.name == "su"))
    ] == [
        0,
        1,
        2,
        3,
    ]
    solver = _dr_rest_solver(golden)
    out = _scaffold(tmp_path, cases, "dr", solver)
    proc = _verify(out)
    assert proc.returncode == 0 and "VERIFY OK" in proc.stdout, proc.stdout + proc.stderr
    assert "OPSCHECK name=step_su__rest mode=basis" in proc.stdout
    assert _rules(cases, "dr", SOLVER, solver, golden) == []


def test_a_wrong_selection_row_copy_is_caught_end_to_end(tmp_path, cases, requires_compiler):
    _, _, golden = cases["dr"]
    out = _scaffold(tmp_path, cases, "dr", _dr_rest_solver(golden, " + 1"))
    proc = _verify(out)
    assert proc.returncode != 0 and "samples exceed tol" in proc.stdout, proc.stdout


# -- what the direct checks catch ----------------------------------------------
CHEATS = [
    # The box bounds of this LP are +-1: hardcoded, the activation never switches off.
    (
        "pdhg",
        "fom_step_t0",
        "if (v < act->lo[i]) v = act->lo[i];\n        if (v > act->hi[i]) v = act->hi[i];",
        "if (v < -1.0) v = -1.0;\n        if (v > 1.0) v = 1.0;",
        "VERIFY FAILED: fom_step_t0 entry",
    ),
    # Right while the activation is off; wrong where it clips.
    (
        "pdhg",
        "fom_step_t0",
        "if (v > act->hi[i]) v = act->hi[i];",
        "if (v > act->hi[i]) v = 0.5 * act->hi[i];",
        "VERIFY FAILED: fom_step_t0 differs from its reference",
    ),
    # Recognizes the activation-off probes by their huge bounds (literals are
    # allowed here, the routine being exactly probed) and is wrong otherwise.
    (
        "pdhg",
        "fom_step_t1",
        "out[i] = v;",
        "if (act->lo[0] > -1e299) v = 0.5 * v;\n        out[i] = v;",
        "VERIFY FAILED: fom_step_t1 differs from its reference",
    ),
    # Accumulates instead of overwriting.
    ("pdhg", "fom_step_t1", "out[i] = v;", "out[i] += v;", "VERIFY FAILED: fom_step_t1"),
    # Drops the bias.
    ("pdhg", "fom_step_t0", "v = bias[i];", "v = 0.0;", "VERIFY FAILED: fom_step_t0 entry"),
    # The threshold from the package macro rather than `act`.
    (
        "scheduled",
        "fom_step_t0",
        "v = (v > act->lambd) ? (v - act->lambd) : ((v < -act->lambd) ? (v + act->lambd) : 0.0);",
        "v = (v > FOM_T0_LAMBDA) ? (v - FOM_T0_LAMBDA) : ((v < -FOM_T0_LAMBDA) ? (v + FOM_T0_LAMBDA) : 0.0);",
        "VERIFY FAILED: fom_step_t0",
    ),
    # A wrong temporary with the right new state.
    (
        "pdhg",
        "fom_iterate",
        "    fom_step_su(t_t0, t_t1, bias_su, new_state);",
        "    fom_step_su(t_t0, t_t1, bias_su, new_state);\n    t_t1[0] += state[0];",
        "of t_t1 is off by",
    ),
    # Right at schedule values 0 and 1, wrong in between (the random setting).
    (
        "scheduled",
        "fom_iterate",
        "    fom_step_su(state, t_t0, bias_su, new_state);",
        "    fom_step_su(state, t_t0, bias_su, new_state);\n"
        "    if (sc_gamma != 0.0 && sc_gamma != 1.0) new_state[0] += state[0];",
        "(random schedule values)",
    ),
    # A __rest routine that also writes a selection row.
    (
        "dr",
        "fom_step_su__rest",
        "        out[i] = v;\n    }",
        "        out[i] = v;\n    }\n    out[0] = bias[0];",
        "fom_step_su__rest changed selection row 0",
    ),
]


@pytest.mark.parametrize("key, fn, old, new, expected", CHEATS)
def test_a_cheating_routine_fails_verify(
    tmp_path, cases, requires_compiler, key, fn, old, new, expected
):
    out = _scaffold(tmp_path, cases, key)
    ops = out / OPS
    ops.write_text(_edit(ops.read_text(), fn, old, new))
    proc = _verify(out)
    assert proc.returncode != 0 and expected in proc.stdout, proc.stdout + proc.stderr


def test_the_huge_bound_trick_passes_the_rules(tmp_path, cases):
    # Literals are allowed in an exactly probed routine, so only the probes stop it.
    out = _scaffold(tmp_path, cases, "pdhg")
    base = (out / OPS).read_text()
    bad = _edit(
        base,
        "fom_step_t1",
        "out[i] = v;",
        "if (act->lo[0] > -1e299) v = 0.5 * v;\n        out[i] = v;",
    )
    assert _rules(cases, "pdhg", OPS, bad, base) == []


@pytest.mark.parametrize(
    "key, fn, line, expected",
    [
        (
            "pdhg",
            "fom_step_t0",
            "v += 0.0 * FOM_MAT_t1_state[0];",
            "fom_step_t0 reads FOM_MAT_t1_state",
        ),
        ("scheduled", "fom_step_t0", "v += 0.0 * fom_sched_gamma(0);", "reads the schedule"),
        ("pdhg", "fom_step_t0", "v += 0.0 * FOM_ACT_t0.lo[i];", "reads FOM_ACT_t0"),
        ("pdhg", "fom_step_t0", "v += 0.0 * FOM_MAT_t0_lo[i];", "reads FOM_MAT_t0_lo"),
        ("pdhg", "fom_step_t1", "v += 0.0 * FOM_MAT_t1_c[i];", "reads FOM_MAT_t1_c"),
    ],
)
def test_the_rules_hold_step_routines_to_their_arguments(tmp_path, cases, key, fn, line, expected):
    out = _scaffold(tmp_path, cases, key)
    base = (out / OPS).read_text()
    bad = _edit(base, fn, "        out[i] = v;", f"        {line}\n        out[i] = v;")
    assert any(expected in v for v in _rules(cases, key, OPS, bad, base))


def test_setup_may_not_call_the_iteration(cases):
    _, _, golden = cases["pdhg"]
    probing = golden.replace(
        "work->settings = *settings;",
        "work->settings = *settings;\n    fom_iterate(0, 0, 0, 0, 0, 0, 0, 0, 0);",
        1,
    )
    assert probing != golden
    assert any(
        "fom_iterate is called from fom_setup" in v
        for v in _rules(cases, "pdhg", SOLVER, probing, golden)
    )


# -- prompts and other targets --------------------------------------------------
def test_the_optimization_prompt_describes_steps_iterate_and_selection_rows(cases):
    case, ir, _ = cases["dr"]
    target = get_target(case["config"])
    text = code_optimization_instructions(ir, target)
    assert "fom_step_<block>" in text and "OVERWRITES" in text
    assert "`fom_step_su__rest`" in text and "rows 0..3 copy t_t0[0..3]" in text
    case, ir, _ = cases["pdhg"]
    text = code_optimization_instructions(ir, get_target(case["config"]))
    assert "fom_iterate(state" in text
    for key in CASES:
        case, ir, _ = cases[key]
        trans = translation_instructions(ir, get_target(case["config"]))
        assert "fom_step" not in trans and "fom_iterate" not in trans and "SELROWS" not in trans


def test_cuda_has_no_step_routines(cases, monkeypatch):
    from fomtools.codegen.targets.large_scale_cuda import CudaTarget

    monkeypatch.setattr(CudaTarget, "_can_run_on_gpu", lambda self: True)
    _, ir, _ = cases["pdhg"]
    cuda = get_target(CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.CUDA))
    assert cuda.step_routines(ir) == {} and not cuda.iterate_routine(ir)
    assert cuda.rest_routines(ir) == [] and cuda._step_vectors(ir) == []
