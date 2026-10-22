"""The code optimization phase in rounds (`run_code_optimization`).

A scripted driver writes a different solver each round (the golden ISTA loop
with a marker comment, or a broken one), and the benchmark is faked from the
marker, so the round logic is tested without timing noise: keep a faster round,
stop on too small a gain, revert a failing or slower round, stop at the last one.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen import generate as gen
from fomtools.codegen.agent.driver import AgentDriver
from fomtools.codegen.agent.prompts import BLOCKED_HEADING, SUMMARY_BEGIN, SUMMARY_END
from fomtools.codegen.postprocess import extract_rule_friction
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
def test_a_small_gain_ends_the_phase(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0, "r2": 49.5, "r3": 10.0}
    result, driver, solver = _run(
        tmp_path, monkeypatch, ["r1", "r2", "r3"], times, agent_optimize_min_gain=0.02
    )
    assert len(driver.prompts) == 2
    assert "/* r2 */" in solver  # kept: it is faster, just not by enough to go on
    assert result.timing.c_final_us == 49.5


@pytest.mark.requires_compiler
def test_a_failing_round_is_reverted(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0}
    result, driver, solver = _run(tmp_path, monkeypatch, ["r1", "broken", "r3"], times)
    assert len(driver.prompts) == 2
    assert "/* r1 */" in solver and BROKEN not in solver
    assert result.optimization_kept
    assert result.timing.c_final_us == 50.0


@pytest.mark.requires_compiler
def test_a_slower_round_is_reverted(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0, "r2": 50.5}  # within 2%, but later rounds
    result, driver, solver = _run(tmp_path, monkeypatch, ["r1", "r2", "r3"], times)
    assert len(driver.prompts) == 2
    assert "/* r1 */" in solver and "/* r2 */" not in solver


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
    """A scripted driver whose rounds each end with a summary naming what a rule
    blocked."""

    def run(self, task):
        result = super().run(task)
        if task.phase == "translation":
            return result
        k = len(self.prompts)
        summary = (
            f"{SUMMARY_BEGIN}\nRound {k} notes.\n\n{BLOCKED_HEADING}\n"
            f"- inline the identity copy {k} | selection rule | ~5%\n"
            f"- merge stages\n\nDone.\n{SUMMARY_END}\n"
        )
        return replace(result, agent_log=(result.agent_log or "") + summary)


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
    assert BLOCKED_HEADING in driver.prompts[0]
    assert [(f["round"], f["idea"]) for f in result.rule_friction] == [
        (1, "inline the identity copy 1"),
        (1, "merge stages"),
        (2, "inline the identity copy 2"),
        (2, "merge stages"),
    ]
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert "## Blocked by rules" in report
    assert "| 2 | inline the identity copy 2 | selection rule | ~5% |" in report
