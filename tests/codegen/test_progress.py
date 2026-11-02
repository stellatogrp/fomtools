"""Progress output of `generate_code(verbose=True)` and its helpers."""

import io
import sys
import time

import pytest

from fomtools.codegen import generate_code
from fomtools.codegen.agent.driver import AgentResult
from fomtools.codegen.agent.prompts import (
    APPLIED_HEADING,
    BLOCKED_HEADING,
    SUMMARY_BEGIN,
    SUMMARY_END,
)
from fomtools.codegen.generate import OptimizationRound
from fomtools.codegen.harness import TimingReport
from fomtools.codegen.postprocess import extract_applied_optimizations
from fomtools.codegen.progress import Progress, bar, verify_error
from tests.codegen.conftest import build_ista
from tests.codegen.test_optimization_rounds import ScriptedDriver, _fake_timing


def test_helpers():
    assert bar(0.4, 10) == ("████", "░░░░░░")
    assert bar(0.0, 10) == ("█", "░" * 9)
    out = "OPSCHECK ...\nworst max-abs error over 8 samples: 1.554e-15 (tol 1.000e-06)\nVERIFY OK"
    assert verify_error(out) == (8, 1.554e-15)
    assert verify_error("VERIFY OK") is None
    notes = f"blah\n{APPLIED_HEADING}\n- fused two matvecs | D^T D is fixed\n- unrolled n=4\n\nmore"
    assert extract_applied_optimizations(notes) == ["fused two matvecs", "unrolled n=4"]
    assert extract_applied_optimizations(f"{APPLIED_HEADING}\n- none") == []


def test_round_lines_and_summary():
    stream = io.StringIO()
    p = Progress(True, stream)
    p.start("translate")
    p.translated(True, "worst max-abs error over 8 samples: 2e-16 (tol 1e-6)", 100.0)
    for r in (
        OptimizationRound(1, True, True, True, us_before=100.0, us_after=40.0),
        OptimizationRound(2, True, False, False),
        OptimizationRound(3, True, True, False, us_before=40.0, us_after=45.0),
    ):
        p.start(f"round {r.index}")
        p.round(r)
    p.summary("lasso", TimingReport(1000.0, 100.0, 40.0), None, ["fused matvecs"])
    text = stream.getvalue()
    assert "\033" not in text  # no colors or cursor codes off a terminal
    assert "VERIFY OK  max err 2e-16  100 µs" in text
    assert "100 µs → 40 µs  █████░░░░░░░  2.50x" in text
    assert "failed verification, reverted" in text
    assert "45 µs vs 40 µs best, not faster, reverted" in text
    assert "2.50x faster" in text and "• fused matvecs" in text


def test_disabled_prints_nothing(capsys):
    p = Progress(False)
    p.start("x")
    p.done()
    p.round(OptimizationRound(1, False, True, False))
    assert capsys.readouterr().out == ""


@pytest.mark.requires_compiler
def test_generate_code_verbose(tmp_path, monkeypatch, capsys, requires_compiler):
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 50.0, "r2": 60.0})
    case = build_ista(max_iters=50)
    generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"].model_copy(update={"agent_optimize_rounds": 2}),
        driver=ScriptedDriver(["r1", "r2"]),
        name="ista",
        code_optimization=True,
        profile=True,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "✓ extract + scaffold" in out
    assert "✓ translate → C (agent)" in out and "VERIFY OK" in out
    assert "optimize · round 1" in out and "2.00x" in out
    assert "optimize · round 2" in out and "not faster, reverted" in out
    assert "2.00x faster" in out


class SilentRoundDriver(ScriptedDriver):
    """Rounds end without a summary; the summary follow-up answers with one and
    also edits the solver, which the pipeline must undo."""

    def __init__(self, script):
        super().__init__(script)
        self.followups: list[str] = []

    def run(self, task):
        if not task.extra.get("summary_only"):
            return super().run(task)
        self.followups.append(task.instructions)
        (task.workdir / task.solver_rel_path).write_text("scribbled;\n")
        summary = (
            f"{SUMMARY_BEGIN}\n{APPLIED_HEADING}\n- fused the two matvecs | D is fixed\n\n"
            f"{BLOCKED_HEADING}\n- none\n{SUMMARY_END}\n"
        )
        return AgentResult(success=True, log="", agent_log=summary, turns_used=1)


@pytest.mark.requires_compiler
def test_missing_summary_is_requested(tmp_path, monkeypatch, capsys, requires_compiler):
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 50.0})
    case = build_ista(max_iters=50)
    driver = SilentRoundDriver(["r1"])
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"].model_copy(update={"agent_optimize_rounds": 1}),
        driver=driver,
        name="ista",
        code_optimization=True,
        profile=True,
        verbose=True,
    )
    assert len(driver.followups) == 1 and "/* r1 */" in driver.followups[0]
    assert "/* r1 */" in (tmp_path / "src" / "fom_solver.c").read_text()  # edit undone
    assert result.optimization_kept
    assert "• fused the two matvecs" in capsys.readouterr().out


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_tty_spinner_stops_and_close_marks_open_phase():
    stream = Tty()
    p = Progress(True, stream)
    p.start("translate")
    time.sleep(0.25)
    p.close()  # e.g. the run raised mid-phase
    assert p._thread is None and "⠋\033[0m translate" in stream.getvalue()
    assert stream.getvalue().rstrip().split("\r\033[K")[-1].startswith("\033[31m✗\033[0m translate")
    size = len(stream.getvalue())
    time.sleep(0.25)
    assert len(stream.getvalue()) == size  # nothing writes after close
    p.close()  # idempotent


def test_untimed_revert_and_unencodable_stream():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    p = Progress(True, stream)
    p.start("optimize · round 1")
    p.round(OptimizationRound(1, True, True, False))
    stream.seek(0)
    assert "? optimize ? round 1" in stream.read()  # the ✗ and · are replaced
    out = io.StringIO()
    p = Progress(True, out)
    p.start("r")
    p.round(OptimizationRound(1, True, True, False))
    assert "✗ r" in out.getvalue() and "not timed, not faster, reverted" in out.getvalue()


def test_summary_list_stops_at_bold_heading_and_reads_numbers():
    notes = f"**{APPLIED_HEADING}**\n- fused | D\n**{BLOCKED_HEADING}**\n- inverse | rule | 2x\n"
    assert extract_applied_optimizations(notes) == ["fused"]
    assert extract_applied_optimizations(f"{APPLIED_HEADING}\n1. a | b\n2) c\n") == ["a", "c"]


class FailingSummaryDriver(SilentRoundDriver):
    def run(self, task):
        if task.extra.get("summary_only"):
            raise RuntimeError("backend down")
        return super().run(task)


def _verbose_run(tmp_path, driver, rounds):
    return generate_code(
        *(lambda c: (c["verifier"], c["next_state"]))(case := build_ista(max_iters=50)),
        tmp_path,
        config=case["config"].model_copy(update={"agent_optimize_rounds": rounds}),
        driver=driver,
        name="ista",
        code_optimization=True,
        profile=True,
        verbose=True,
    )


@pytest.mark.requires_compiler
def test_followup_only_for_kept_rounds(tmp_path, monkeypatch, capsys, requires_compiler):
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 50.0, "r2": 60.0})
    driver = SilentRoundDriver(["r1", "r2"])
    _verbose_run(tmp_path, driver, 2)
    assert len(driver.followups) == 1 and "/* r1 */" in driver.followups[0]


@pytest.mark.requires_compiler
def test_failed_followup_warns_and_nothing_kept_prints_no_summary(
    tmp_path, monkeypatch, capsys, requires_compiler
):
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 50.0})
    with pytest.warns(UserWarning, match="round summary failed"):
        result = _verbose_run(tmp_path / "a", FailingSummaryDriver(["r1"]), 1)
    assert result.optimization_kept and "what the agent found" not in capsys.readouterr().out
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 150.0})
    result = _verbose_run(tmp_path / "b", SilentRoundDriver(["r1"]), 1)
    out = capsys.readouterr().out
    assert not result.optimization_kept and "translated C" not in out and "reverted" in out


def test_generate_code_closes_progress_on_error(tmp_path, monkeypatch):
    stream = Tty()
    monkeypatch.setattr(sys, "stdout", stream)

    class Boom(ScriptedDriver):
        def run(self, task):
            raise KeyboardInterrupt

    case = build_ista(max_iters=50)
    with pytest.raises(KeyboardInterrupt):
        generate_code(
            case["verifier"],
            case["next_state"],
            tmp_path,
            config=case["config"],
            driver=Boom([]),
            name="ista",
            verbose=True,
        )
    assert "translate → C (agent)" in stream.getvalue().rstrip().split("\r\033[K")[-1]
    assert "stopped" in stream.getvalue()
