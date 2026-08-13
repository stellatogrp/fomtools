"""What an unrolled IR emits, and how the harness pins it down.

An unrolled body is the whole recorded run, so it must be executed exactly once.
That is a *structural* requirement the numeric checks cannot see on their own: a
body wrapped in the usual ``for (it = 0; it < settings.max_iter; ++it)`` computes
the right answer under the generated default budget of 1, and silently runs the
algorithm several times over for any caller who raises it. The emitted harness
therefore raises the budget itself and requires the answer not to move.
"""

from pathlib import Path

from fomtools.codegen.agent.prompts import phase_a_instructions
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.extract_unrolled import extract_unrolled_ir
from fomtools.codegen.harness import build_and_verify
from fomtools.codegen.spec import render_markdown
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_halpern_loop, build_ista

GOLDEN = Path(__file__).parent / "golden" / "halpern" / "embedded_c" / "fom_solver.c"

#: Matches the golden solver body, which names one block per recorded iteration.
HALPERN_STEPS = 5


def _halpern_ir():
    v, iterates, x, d = build_halpern_loop(steps=HALPERN_STEPS)
    ir = extract_unrolled_ir(v, iterates, name="halpern")
    assert ir.unrolled is True
    return ir


def _rolled_ir():
    case = build_ista()
    return extract_ir(case["verifier"], case["next_state"], config=case["config"], name="ista")


# --- macros -----------------------------------------------------------------


def test_unrolled_macro_is_set(tmp_path):
    scaffold(_halpern_ir(), tmp_path)
    header = (tmp_path / "include" / "fom_workspace.h").read_text()
    assert "#define FOM_UNROLLED 1" in header
    assert "#define FOM_MAX_ITER 1" in header


def test_unrolled_macro_is_clear_for_a_rolled_ir(tmp_path):
    scaffold(_rolled_ir(), tmp_path)
    header = (tmp_path / "include" / "fom_workspace.h").read_text()
    assert "#define FOM_UNROLLED 0" in header


# --- the agent-facing contract ---------------------------------------------


def test_spec_tells_the_agent_not_to_loop():
    md = render_markdown(_halpern_ir())
    assert "to be implemented as a single pass" in md
    assert "Do **not** write a loop over iterations" in md
    assert "Ignore `work->settings.max_iter`" in md
    assert "## Final state (the answer)" in md
    # And it does not describe itself as one iteration.
    assert "One iteration of a first-order method" not in md


def test_spec_for_a_rolled_ir_is_unchanged():
    md = render_markdown(_rolled_ir())
    assert "One iteration of a first-order method, to be implemented as a fixed loop." in md
    assert "## Iteration body" in md
    assert "## State update" in md
    assert "Do **not** write a loop" not in md


def test_spec_permits_rolling_an_identical_run_of_blocks():
    """Implementation freedom, not a change to the specification."""
    md = render_markdown(_halpern_ir())
    assert "implement them with a loop if that is faster" in md


def test_phase_a_prompt_states_the_single_pass_requirement():
    target = EmbeddedCTarget(build_ista()["config"])
    prompt = phase_a_instructions(_halpern_ir(), target)
    assert "UNROLLED" in prompt
    assert "Do not write a loop over iterations" in prompt

    rolled = phase_a_instructions(_rolled_ir(), target)
    assert "UNROLLED" not in rolled


def test_readme_records_the_single_pass(tmp_path):
    scaffold(_halpern_ir(), tmp_path)
    readme = (tmp_path / "README.md").read_text()
    assert "single pass" in readme
    assert "`settings.max_iter` does not apply" in readme


# --- the harness ------------------------------------------------------------


def test_harness_checks_the_body_ignores_max_iter(tmp_path):
    scaffold(_halpern_ir(), tmp_path)
    verify = (tmp_path / "test" / "verify_main.c").read_text()
    assert "settings.max_iter = FOM_MAX_ITER + 3;" in verify
    assert "an unrolled body must execute exactly once" in verify


def test_harness_omits_the_check_for_a_rolled_ir(tmp_path):
    scaffold(_rolled_ir(), tmp_path)
    verify = (tmp_path / "test" / "verify_main.c").read_text()
    assert "FOM_MAX_ITER + 3" not in verify
    assert "must execute exactly once" not in verify


# --- end to end -------------------------------------------------------------


def test_golden_passes_static_checks():
    target = EmbeddedCTarget(build_ista()["config"])
    assert target.static_checks(GOLDEN.read_text()) == []


def test_unrolled_halpern_verifies_against_the_oracle(tmp_path, requires_compiler):
    """A method no rolled path can express, emitted and machine-checked."""
    target = scaffold(_halpern_ir(), tmp_path)
    (tmp_path / "src" / "fom_solver.c").write_text(GOLDEN.read_text())

    outcome = build_and_verify(tmp_path, target)
    assert outcome.success, outcome.output
    assert "VERIFY OK" in outcome.output


def test_a_looping_body_is_rejected(tmp_path, requires_compiler):
    """Proves the single-pass check is not vacuous.

    The looping body is numerically correct on every emitted sample, because the
    generated default budget is 1. Only raising it exposes the mistake.
    """
    ir = _halpern_ir()
    target = scaffold(ir, tmp_path)

    source = GOLDEN.read_text()
    looped = source.replace(
        "    fom_int i;\n    fom_float res = (fom_float)0.0;\n\n    zero(work->acc, FOM_T0_DIM);",
        "    fom_int i, it;\n    fom_float res = (fom_float)0.0;\n\n"
        "    for (it = 0; it < work->settings.max_iter; ++it) {\n"
        "    zero(work->acc, FOM_T0_DIM);",
    ).replace(
        "        work->state[i] = work->new_state[i];\n    }\n\n"
        "    for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i]",
        "        work->state[i] = work->new_state[i];\n    }\n    }\n\n"
        "    for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i]",
    )
    assert looped != source, "the golden body changed; update this rewrite"

    (tmp_path / "src" / "fom_solver.c").write_text(looped)
    outcome = build_and_verify(tmp_path, target)
    assert not outcome.success
    assert "must execute exactly once" in outcome.output
