"""The code optimization phase in rounds (`run_code_optimization`).

A scripted driver writes a different solver each round (the golden ISTA loop
with a marker comment, or a broken one), and the benchmark is faked from the
marker, so the round logic is tested without timing noise: keep a faster round,
revert a failing, slower or too-little-gain round and tell the next round why,
stop after agent_optimize_patience such rounds in a row or at the last one.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen import generate as gen
from fomtools.codegen.agent.driver import AgentDriver
from fomtools.codegen.agent.prompts import (
    APPLIED_HEADING,
    BLOCKED_HEADING,
    SUMMARY_BEGIN,
    SUMMARY_END,
    code_optimization_continuation,
)
from fomtools.codegen.harness import VerifyResult
from fomtools.codegen.postprocess import extract_applied_optimizations, extract_rule_friction
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"
BROKEN = "this is not C;\n"


class ScriptedDriver(AgentDriver):
    """Translation: the golden solver. Round k of optimization: ``script[k-1]``,
    the golden solver tagged with that marker (or ``BROKEN``)."""

    def __init__(self, script: list[str]):
        self.script = script
        self.prompts: list[str] = []

    def run(self, task):
        if task.phase == "translation":
            return StubAgentDriver(GOLDEN).run(task)
        self.prompts.append(task.instructions)
        marker = self.script[len(self.prompts) - 1]
        source = BROKEN if marker == "broken" else GOLDEN.read_text() + f"\n/* {marker} */\n"
        return StubAgentDriver(source).run(task)


def _fake_timing(monkeypatch, times):
    """``times`` maps marker -> fake bench us (the unmarked translated solver is
    ``times["translated"]``); profiling is skipped."""

    def fake_bench(out_dir):
        text = (Path(out_dir) / "src" / "fom_solver.c").read_text()
        for marker, us in times.items():
            if f"/* {marker} */" in text:
                return us
        return times["translated"]

    monkeypatch.setattr(gen, "_bench_mean_us", fake_bench)
    monkeypatch.setattr(gen, "run_profile", lambda out_dir: [])


def _run(tmp_path, monkeypatch, script, times, **config):
    """Generate with ``script`` per round and the fake ``times``."""
    _fake_timing(monkeypatch, times)
    case = build_ista(max_iters=50)
    driver = ScriptedDriver(script)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"].model_copy(update=config),
        driver=driver,
        name="ista",
        code_optimization=True,
        profile=True,
    )
    return result, driver, (tmp_path / "src" / "fom_solver.c").read_text()


@pytest.mark.requires_compiler
def test_rounds_continue_until_the_last(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0, "r2": 25.0, "r3": 12.0}
    result, driver, solver = _run(
        tmp_path, monkeypatch, ["r1", "r2", "r3"], times, agent_optimize_rounds=3
    )
    assert result.optimization_kept
    assert len(driver.prompts) == 3
    assert "/* r3 */" in solver
    assert result.timing.c_initial_us == 100.0 and result.timing.c_final_us == 12.0
    # Later rounds are told where they stand.
    assert "round 2 of at most 3" in driver.prompts[1]
    assert "50.000 us per solve" in driver.prompts[1]
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "## Rounds" in report and gen.STOP_MAX_ROUNDS in report


@pytest.mark.requires_compiler
def test_a_small_gain_is_reverted_and_uses_patience(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0, "r2": 49.5, "r3": 10.0}
    result, driver, solver = _run(
        tmp_path,
        monkeypatch,
        ["r1", "r2", "r3"],
        times,
        agent_optimize_min_gain=0.02,
        agent_optimize_rounds=3,
    )
    assert len(driver.prompts) == 3
    assert "/* r3 */" in solver and "/* r2 */" not in solver
    assert result.timing.c_final_us == 10.0
    assert (
        "Round 2: reverted (not enough gain: 49.500 us vs 50.000 us best (+1.0%))"
        in driver.prompts[2]
    )


@pytest.mark.requires_compiler
def test_a_failing_round_is_reverted(tmp_path, monkeypatch, requires_compiler):
    # r3 (not in times) runs at the translated 100 us: a second miss in a row.
    times = {"translated": 100.0, "r1": 50.0}
    result, driver, solver = _run(tmp_path, monkeypatch, ["r1", "broken", "r3"], times)
    assert len(driver.prompts) == 3
    assert "/* r1 */" in solver and BROKEN not in solver
    assert result.optimization_kept
    assert result.timing.c_final_us == 50.0
    # The next round is told the failed one was reverted, and why.
    assert "Round 2: reverted (failed verification: " in driver.prompts[2]
    assert "The previous round made it" not in driver.prompts[2]
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "reverted: failed verification: " in report and gen.STOP_PATIENCE in report


class StaleFailureDriver(ScriptedDriver):
    """A scripted driver whose logs open with a failure the agent fixed in an
    earlier attempt, as a real agent's do; only its last verify is current."""

    def run(self, task):
        result = super().run(task)
        if task.phase == "translation":
            return result
        stale = "VERIFY FAILED: stale failure from an earlier attempt\n"
        return replace(result, log=stale + result.log, agent_log=stale + result.agent_log)


@pytest.mark.requires_compiler
def test_the_reason_is_the_last_verify_not_a_stale_one(tmp_path, monkeypatch, requires_compiler):
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 50.0, "r3": 25.0})
    case = build_ista(max_iters=50)
    driver = StaleFailureDriver(["r1", "broken", "r3"])
    generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"].model_copy(update=dict(agent_optimize_rounds=3)),
        driver=driver,
        name="ista",
        code_optimization=True,
        profile=True,
    )
    line = next(ln for ln in driver.prompts[2].splitlines() if ln.startswith("Round 2:"))
    assert line.startswith("Round 2: reverted (failed verification: ")
    assert "stale" not in line


@pytest.mark.requires_compiler
def test_a_slower_round_is_reverted(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0, "r2": 50.5}
    result, driver, solver = _run(tmp_path, monkeypatch, ["r1", "r2", "r3"], times)
    assert len(driver.prompts) == 3
    assert "/* r1 */" in solver and "/* r2 */" not in solver
    assert (
        "Round 2: reverted (not enough gain: 50.500 us vs 50.000 us best (-1.0%))"
        in driver.prompts[2]
    )
    assert gen.STOP_PATIENCE in (tmp_path / "OPTIMIZATIONS.md").read_text()


@pytest.mark.requires_compiler
def test_patience_one_stops_at_the_first_miss(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0}
    result, driver, solver = _run(
        tmp_path, monkeypatch, ["r1", "broken", "r3"], times, agent_optimize_patience=1
    )
    assert len(driver.prompts) == 2
    assert "/* r1 */" in solver
    assert gen.STOP_PATIENCE in (tmp_path / "OPTIMIZATIONS.md").read_text()


@pytest.mark.requires_compiler
def test_a_gain_resets_patience(tmp_path, monkeypatch, requires_compiler):
    # miss (r2), gain (r3), miss (r4), miss (r5): without the reset r4 would stop.
    times = {"translated": 100.0, "r1": 50.0, "r2": 50.5, "r3": 25.0}
    result, driver, solver = _run(
        tmp_path, monkeypatch, ["r1", "r2", "r3", "r4", "r5", "r6"], times, agent_optimize_rounds=6
    )
    assert len(driver.prompts) == 5
    assert "/* r3 */" in solver
    assert result.timing.c_final_us == 25.0
    assert gen.STOP_PATIENCE in (tmp_path / "OPTIMIZATIONS.md").read_text()


@pytest.mark.requires_compiler
def test_a_first_round_kept_within_noise_uses_patience(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 101.0, "r2": 101.0}
    result, driver, solver = _run(tmp_path, monkeypatch, ["r1", "r2", "r3"], times)
    assert len(driver.prompts) == 2
    assert "/* r1 */" in solver  # kept: the first round's 2% tolerance
    assert (
        "Round 1: kept, -1.0% faster (not enough gain: 101.000 us vs 100.000 us best (-1.0%))"
        in driver.prompts[1]
    )


@pytest.mark.requires_compiler
def test_a_long_verify_log_gives_one_short_line(tmp_path, monkeypatch, requires_compiler):
    real = gen.build_and_verify

    def verify(out_dir, target):
        if "/* r2 */" in (Path(out_dir) / "src" / "fom_solver.c").read_text():
            out = "noise\n" * 10000 + "VERIFY FAILED: " + "x" * 5000 + "\n" + "tail\n" * 1000
            return VerifyResult(success=False, returncode=1, stdout=out, stderr="")
        return real(out_dir, target)

    monkeypatch.setattr(gen, "build_and_verify", verify)
    times = {"translated": 100.0, "r1": 50.0, "r3": 25.0}
    result, driver, solver = _run(
        tmp_path, monkeypatch, ["r1", "r2", "r3"], times, agent_optimize_rounds=3
    )
    assert "/* r3 */" in solver
    line = next(ln for ln in driver.prompts[2].splitlines() if ln.startswith("Round 2:"))
    assert line.startswith("Round 2: reverted (failed verification: VERIFY FAILED: xxx")
    assert len(line) <= 360 and "noise" not in line and "tail" not in line


def test_failure_line():
    assert gen._failure_line("cc -O2 a.c\nVERIFY FAILED: 3/8 samples\nmake: Error 1") == (
        "VERIFY FAILED: 3/8 samples"
    )
    assert gen._failure_line("cc a.c\na.c:1:1: error: bad\n") == "a.c:1:1: error: bad"
    assert gen._failure_line("one\n  last   line \n\n") == "last line"
    assert gen._failure_line("") == ""
    assert len(gen._failure_line("VERIFY FAILED: " + "x" * 1000, limit=50)) == 50


@pytest.mark.requires_compiler
def test_one_round_is_the_single_pass_phase(tmp_path, monkeypatch, requires_compiler):
    # The first round keeps a rewrite within the 2% noise tolerance, as before.
    times = {"translated": 100.0, "r1": 101.0}
    result, driver, solver = _run(tmp_path, monkeypatch, ["r1"], times, agent_optimize_rounds=1)
    assert len(driver.prompts) == 1
    assert "/* r1 */" in solver
    assert result.optimization_kept
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "## Rounds" not in report  # one round: no table


@pytest.mark.requires_compiler
def test_run_code_optimization_on_a_copy(tmp_path, monkeypatch, requires_compiler):
    """The entry point callers use to keep the translated solver: optimize a copy."""
    import shutil

    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 40.0})
    case = build_ista(max_iters=50)
    emitted = tmp_path / "translated_only"
    first = generate_code(
        case["verifier"],
        case["next_state"],
        emitted,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
    )
    copy = tmp_path / "copy"
    shutil.copytree(emitted, copy)
    driver = ScriptedDriver(["r1", "r1"])  # round 2 changes nothing
    opt = gen.run_code_optimization(
        first.ir,
        first.target,
        copy,
        driver,
        case["config"].model_copy(update={"agent_optimize_rounds": 2}),
    )
    assert opt.kept and opt.translated_us == 100.0 and opt.final_us == 40.0
    assert [r.kept for r in opt.rounds] == [True, False]  # round 2: unchanged sources
    assert opt.rounds[-1].stop_reason == gen.STOP_NO_CHANGE
    assert "/* r1 */" not in (emitted / "src" / "fom_solver.c").read_text()
    assert opt.optimizations_md and opt.optimizations_md.exists()


class ReportingDriver(ScriptedDriver):
    """A scripted driver whose rounds each end with a summary naming what it
    applied and what a rule blocked."""

    def run(self, task):
        result = super().run(task)
        if task.phase == "translation":
            return result
        k = len(self.prompts)
        summary = (
            f"{SUMMARY_BEGIN}\nRound {k} notes.\n\n{APPLIED_HEADING}\n"
            f"- hoist the step {k} | constant step size\n\n{BLOCKED_HEADING}\n"
            f"- inline the identity copy {k} | selection rule | ~5%\n"
            f"- merge stages\n\nDone.\n{SUMMARY_END}\n"
        )
        return replace(result, agent_log=(result.agent_log or "") + summary)


def test_extract_applied_optimizations():
    notes = (
        "Speedup 2x.\n\n**Optimizations applied:**\n"
        "- hoist A^T b | constant parameter\n"
        "* unroll the soft-threshold\n\nBlocked by rules:\n- merge | fused"
    )
    assert extract_applied_optimizations(notes) == [
        "hoist A^T b | constant parameter",
        "unroll the soft-threshold",
    ]
    assert extract_applied_optimizations(f"## {APPLIED_HEADING}\n- none\n") == []
    assert extract_applied_optimizations("no section at all") == []


def test_history_names_each_earlier_round():
    history = [
        dict(index=1, kept=True, gain=0.5, applied=["hoist x | constant"], blocked=[]),
        dict(
            index=2,
            kept=True,
            gain=0.25,
            applied=[],
            blocked=[dict(idea="merge stages", rule="fused rule", expected_gain="5%")],
        ),
        dict(index=3, kept=True, gain=0.1, applied=[], blocked=[]),
    ]
    text = code_optimization_continuation(4, 5, 10.0, 40.0, 0.1, [], history=history)
    assert "Round 1: kept, 50.0% faster\n  applied: hoist x | constant" in text
    assert "Round 2: kept, 25.0% faster\n  blocked: merge stages | fused rule" in text
    assert "Round 3: kept, 10.0% faster\n  (no list of changes recovered" in text
    # Without a history the block is absent, as before.
    assert "Earlier rounds" not in code_optimization_continuation(2, 5, 10.0, 40.0, 0.1)


def test_history_is_capped_and_keeps_the_newest_round():
    item = "x" * 1000
    history = [
        dict(index=i, kept=True, gain=0.1, applied=[item] * 5, blocked=[]) for i in range(1, 5)
    ]
    text = code_optimization_continuation(5, 5, 10.0, 40.0, 0.1, [], history=history)
    block = text[text.index("Earlier rounds") :]
    assert len(block) < 3500
    assert "x" * 1000 not in block  # each item is cut
    assert "Round 1: kept, 10.0% faster\nRound 2" in block  # older rounds: header only
    assert block.count("applied:") >= 1 and "Round 4: kept, 10.0% faster\n  applied:" in block


def test_extract_rule_friction():
    notes = (
        "Speedup 2x.\n\n**Blocked by rules:**\n"
        "- unroll with literals | no new floating-point constants | 2x\n"
        "- skip a | b\n\nTried and dropped: nothing."
    )
    assert extract_rule_friction(notes) == [
        dict(
            idea="unroll with literals", rule="no new floating-point constants", expected_gain="2x"
        ),
        dict(idea="skip a", rule="b", expected_gain=""),
    ]
    assert extract_rule_friction(f"x\n{BLOCKED_HEADING}\n- none\n") == []
    assert extract_rule_friction("no section at all") == []


@pytest.mark.requires_compiler
def test_rule_friction_is_collected_per_round_and_reported(
    tmp_path, monkeypatch, requires_compiler
):
    _fake_timing(monkeypatch, {"translated": 100.0, "r1": 50.0, "r2": 25.0})
    case = build_ista(max_iters=50)
    driver = ReportingDriver(["r1", "r2"])
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"].model_copy(update=dict(agent_optimize_rounds=2)),
        driver=driver,
        name="ista",
        code_optimization=True,
        profile=True,
    )
    assert BLOCKED_HEADING in driver.prompts[0] and APPLIED_HEADING in driver.prompts[0]
    # Round 2 is told what round 1 applied and what a rule blocked.
    assert "Round 1: kept, 50.0% faster" in driver.prompts[1]
    assert "applied: hoist the step 1 | constant step size" in driver.prompts[1]
    assert "blocked: inline the identity copy 1 | selection rule; merge stages" in driver.prompts[1]
    assert [(f["round"], f["idea"]) for f in result.rule_friction] == [
        (1, "inline the identity copy 1"),
        (1, "merge stages"),
        (2, "inline the identity copy 2"),
        (2, "merge stages"),
    ]
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "## Blocked by rules" in report
    assert "| 2 | inline the identity copy 2 | selection rule | ~5% |" in report
