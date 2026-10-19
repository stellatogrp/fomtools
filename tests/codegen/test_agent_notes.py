"""Capturing the coding agent's own summary of what it optimized.

`OPTIMIZATIONS.md` is the durable record of what the code optimization phase did and
why, so the extraction that fills it has to be trustworthy. It was not: when an
agent attempt hit its wall-clock budget, the raw `TimeoutExpired` was
stringified into the log, and `str()` on a subprocess exception interpolates the
whole argv — including the prompt. Because the prompt is what *asks* for a
summary, the marker search then returned the request instead of the reply, and
the report presented it as the agent's notes.

These tests pin down each link: the redaction, the partial-output recovery, the
extraction, and the honesty of the report when nothing was recovered.
"""

import subprocess
import sys

import pytest

from fomtools.codegen.agent.driver import (
    _decode_stream,
    _redact_exception,
    run_verify_loop,
)
from fomtools.codegen.agent.prompts import SUMMARY_BEGIN, SUMMARY_END
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.postprocess import extract_agent_notes, write_optimization_report

# The prompt fragment that made the original bug invisible: it contains the very
# phrase the extractor searched for.
PROMPT_TAIL = (
    "When done, print a summary as the last thing you output...\n"
    "FINAL SUMMARY containing:\\n- the initial vs. final whole-solve `mean_us`"
)

# A faithful reconstruction of the corrupted log: prompt echoed inside an argv
# repr, the timeout message, then the harness's own build output.
POISONED_LOG = (
    "[agent invocation raised: Command '['claude', '-p', '" + PROMPT_TAIL + "', "
    "'--add-dir', '/tmp/out/ista', '--allowedTools', 'Read', 'Write', 'Edit', "
    "'--permission-mode', 'bypassPermissions', '--append-system-prompt', "
    "'You are an expert in high-performance numerical C.', '--output-format', "
    "'text']' timed out after 600 seconds]"
    "cc -O2 -Wall -Iinclude -Itest src/fom_solver.c -o verify_bin -lm\n"
    "worst max-abs error over 8 samples: 5.440e-15 (tol 1.000e-06)\nVERIFY OK\n"
)

REAL_SUMMARY = (
    "Initial 258.960 us -> final 44.241 us (5.85x).\n"
    "Hottest: matvec_t0_state 82.08%, threshold_update 17.92%.\n"
    "Optimizations: hoisted the iteration-invariant parameter term out of the "
    "loop; skipped the zero entries the soft threshold produces."
)


# --- the regression: a prompt echo must never be reported as agent notes ----
def test_prompt_echo_is_not_returned_as_notes():
    notes = extract_agent_notes(POISONED_LOG)
    for leak in (
        "--append-system-prompt",
        "--add-dir",
        "--allowedTools",
        "--permission-mode",
        "timed out after",
        "You are an expert in high-performance numerical C",
    ):
        assert leak not in notes, f"leaked {leak!r} into the report: {notes[:200]!r}"


def test_prompt_echo_yields_empty_rather_than_misleading_text():
    """Silence is honest here; the prompt masquerading as a summary is not."""
    assert extract_agent_notes(POISONED_LOG).strip() == ""


# --- the happy path -------------------------------------------------------
def test_sentinel_block_is_extracted_exactly():
    log = (
        "thinking out loud, editing files, running make...\n"
        f"{SUMMARY_BEGIN}\n{REAL_SUMMARY}\n{SUMMARY_END}\n"
        "trailing chatter after the block\n"
    )
    notes = extract_agent_notes(log)
    assert notes == REAL_SUMMARY
    assert "trailing chatter" not in notes
    assert "thinking out loud" not in notes
    assert SUMMARY_BEGIN not in notes and SUMMARY_END not in notes


def test_long_sentinel_block_is_not_truncated():
    """A delimited block has known extent, so it is reported whole.

    The first fixed run lost the tail of a real summary mid-sentence to the
    generic character cap.
    """
    long_summary = "\n".join(f"{i}. an optimization and the structure it uses" for i in range(400))
    log = f"chatter\n{SUMMARY_BEGIN}\n{long_summary}\n{SUMMARY_END}\n"
    notes = extract_agent_notes(log)
    assert len(notes) > 4000
    assert notes.endswith("399. an optimization and the structure it uses")


def test_real_summary_survives_a_prompt_echo_in_the_same_log():
    """A retried attempt puts the prompt in the log too; the reply must still win."""
    log = POISONED_LOG + f"\n{SUMMARY_BEGIN}\n{REAL_SUMMARY}\n{SUMMARY_END}\n"
    assert extract_agent_notes(log) == REAL_SUMMARY


def test_heading_fallback_for_a_backend_that_ignores_the_markers():
    log = "ran some things\n\n## Summary\nmade it 3x faster by fusing the matvec\n"
    notes = extract_agent_notes(log)
    assert "3x faster" in notes


def test_unmarked_tail_is_still_returned():
    log = "no markers at all, but this is clearly the agent talking\n"
    assert "clearly the agent talking" in extract_agent_notes(log)


def test_empty_log_gives_empty_notes():
    assert extract_agent_notes("") == ""


def test_harness_diagnostic_is_not_reported_as_notes():
    """A message *about* the agent is not the agent talking.

    A turn-capped run left only this line in the log, and it was rendered under
    an "Agent notes" heading as though the agent had written it.
    """
    log = (
        "[agent invocation raised: Exception: Claude Code returned an error "
        "result: Reached maximum number of turns (60)]"
    )
    assert extract_agent_notes(log) == ""


def test_partial_transcript_survives_its_truncation_marker():
    """Real prose plus a trailing harness marker still yields the prose."""
    log = (
        f"{SUMMARY_BEGIN}\n{REAL_SUMMARY}\n{SUMMARY_END}\n"
        "\n[fomtools: the Claude agent stopped early (Exception: ...); "
        "any output above is partial.]\n"
    )
    assert extract_agent_notes(log) == REAL_SUMMARY


# --- redaction -------------------------------------------------------------
def test_redact_exception_hides_the_argv_and_prompt():
    exc = subprocess.TimeoutExpired(
        cmd=["claude", "-p", "SECRET PROMPT with FINAL SUMMARY inside"], timeout=600
    )
    message = _redact_exception(exc)
    assert "SECRET PROMPT" not in message
    assert "FINAL SUMMARY" not in message
    assert "claude" not in message
    assert "600" in message and "TimeoutExpired" in message


def test_redact_exception_hides_a_called_process_argv():
    exc = subprocess.CalledProcessError(returncode=2, cmd=["claude", "-p", "SECRET"])
    message = _redact_exception(exc)
    assert "SECRET" not in message
    assert "2" in message


def test_redact_exception_keeps_ordinary_messages():
    message = _redact_exception(ValueError("something ordinary went wrong"))
    assert "something ordinary went wrong" in message
    assert "ValueError" in message


# --- partial-output recovery ----------------------------------------------
def test_decode_stream_handles_bytes_str_and_none():
    # TimeoutExpired.stdout arrives as bytes even under text=True.
    assert _decode_stream(b"partial\n") == "partial\n"
    assert _decode_stream("partial\n") == "partial\n"
    assert _decode_stream(None) == ""


def test_timeout_keeps_partial_output_and_leaks_no_argv():
    """The killed agent's output is recoverable; the plumbing must not drop it."""
    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

    driver = ClaudeAgentDriver()
    argv = [
        sys.executable,
        "-c",
        "import sys,time;print('AGENT PARTIAL OUTPUT');sys.stdout.flush();time.sleep(30)",
    ]
    out = driver._invoke_cli(argv, workdir=".", timeout=2)

    assert "AGENT PARTIAL OUTPUT" in out
    assert "exceeded its 2s budget" in out
    assert sys.executable not in out  # no argv anywhere
    assert "time.sleep" not in out
    assert driver._truncated is True


# --- truncation reaches the report ---------------------------------------
def test_truncation_is_recorded_on_the_result():
    task = _noop_task()
    result = run_verify_loop(
        task, lambda prompt, turns: "agent said things", truncated_probe=lambda: True
    )
    assert result.success is True
    assert result.truncated is True


def test_untruncated_run_reports_false():
    result = run_verify_loop(_noop_task(), lambda prompt, turns: "fine")
    assert result.truncated is False


def test_agent_log_excludes_verify_output():
    result = run_verify_loop(_noop_task(), lambda prompt, turns: "AGENT WORDS")
    assert "AGENT WORDS" in result.agent_log
    # The combined log keeps carrying verify output (tests and the notebooks read it).
    assert "VERIFY-MARKER" in result.log
    assert "VERIFY-MARKER" not in result.agent_log


def _noop_task():
    """An AgentTask whose verify command trivially succeeds."""
    from pathlib import Path

    from fomtools.codegen.agent.driver import AgentTask

    return AgentTask(
        workdir=Path("."),
        solver_rel_path="src/fom_solver.c",
        verify_cmd=[sys.executable, "-c", "print('VERIFY-MARKER')"],
    )


# --- the report says so ---------------------------------------------------
def test_report_renders_caveats(tmp_path):
    path = write_optimization_report(
        tmp_path,
        timing_summary="t",
        diff="--- a\n+++ b\n",
        agent_notes="",
        optimize_succeeded=True,
        caveats=["The agent was terminated at its wall-clock budget."],
    )
    text = path.read_text()
    assert "## Caveats" in text
    assert "terminated at its wall-clock budget" in text


def test_report_without_caveats_is_unchanged(tmp_path):
    """Adding the parameter must not alter output for an ordinary run."""
    kwargs = dict(
        timing_summary="t", diff="--- a\n+++ b\n", agent_notes="n", optimize_succeeded=True
    )
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    a = write_optimization_report(tmp_path / "a", **kwargs).read_text()
    b = write_optimization_report(tmp_path / "b", **kwargs, caveats=[]).read_text()
    assert a == b
    assert "## Caveats" not in a


# --- per-phase budgets ----------------------------------------------------
@pytest.mark.parametrize(
    ("phase", "expected"),
    [("translation", 111), ("code_optimization", 222)],
)
def test_phase_selects_its_own_budget(phase, expected):
    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

    config = CodegenConfig(agent_timeout_s=111, agent_optimize_timeout_s=222)
    assert ClaudeAgentDriver(config=config)._timeout_for_phase(phase) == expected


def test_budget_falls_back_to_the_constructor_default_without_config():
    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

    driver = ClaudeAgentDriver(cli_timeout=42)
    assert driver._timeout_for_phase("translation") == 42
    assert driver._timeout_for_phase("code_optimization") == 42


def test_a_driver_built_without_a_config_takes_the_runs_budgets(tmp_path, requires_compiler):
    from pathlib import Path

    from fomtools.codegen import StubAgentDriver, generate_code
    from tests.codegen.conftest import build_ista

    class Configless(StubAgentDriver):
        config = None

    case = build_ista()
    golden = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"
    driver = Configless(golden)
    config = case["config"].model_copy(update={"agent_optimize_timeout_s": 123})
    generate_code(case["verifier"], case["next_state"], tmp_path, config=config, driver=driver)
    assert driver.config is config


def test_optimize_budget_defaults_larger_than_translate():
    """The code optimization phase is open-ended, so it needs the longer leash."""
    config = CodegenConfig()
    assert config.agent_optimize_timeout_s > config.agent_timeout_s
