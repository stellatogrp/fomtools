"""The wall-clock budget of an agent phase: the retry loop, the CLI driver, and the
deadline hook in the bundled ``claude`` spec."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fomtools.codegen.agent import CliAgentDriver
from fomtools.codegen.agent.budget import GRACE_S, TimeBudget
from fomtools.codegen.agent.cli_specs import CliAgentSpec, get_spec
from fomtools.codegen.agent.driver import AgentTask, _decode_stream, run_verify_loop


def _failing_task(tmp_path):
    return AgentTask(
        workdir=Path(tmp_path),
        solver_rel_path="src/fom_solver.c",
        verify_cmd=[sys.executable, "-c", "raise SystemExit(1)"],  # always fails
    )


def test_no_new_attempt_starts_once_the_phase_budget_is_spent(tmp_path):
    task = _failing_task(tmp_path)
    calls = []

    def slow_agent(prompt):
        calls.append(time.monotonic())
        time.sleep(0.3)
        return "attempted"

    result = run_verify_loop(task, slow_agent, max_attempts=5, budget_s=0.2)
    assert not result.success
    assert len(calls) == 1
    assert result.turns_used == 1
    assert "budget for this phase is spent" in result.log
    assert isinstance(task.extra["budget"], TimeBudget)


def test_without_a_budget_every_attempt_runs(tmp_path):
    task = _failing_task(tmp_path)
    result = run_verify_loop(task, lambda prompt: "attempted", max_attempts=3)
    assert result.turns_used == 3
    assert "budget" not in task.extra
    assert _decode_stream(None) == ""


def test_agent_deadline_leaves_the_grace_period():
    budget = TimeBudget.starting_now(600.0)
    assert abs(budget.agent_deadline() - (time.time() + 600 - GRACE_S)) <= 1
    spent = TimeBudget.starting_now(10.0)  # less than the grace period left
    assert abs(spent.agent_deadline() - time.time()) <= 1


def _recording_driver(tmp_path, monkeypatch):
    """A CLI driver whose CLI is a script that dumps its env and exits."""
    spec = CliAgentSpec(name="probe", bin=sys.executable, model_flag=None)
    driver = CliAgentDriver(spec, max_attempts=1)
    seen = {}

    def fake_invoke(argv, workdir, timeout=None, deadline=None):
        seen.update(timeout=timeout, env=driver._child_env(deadline))
        return "ran"

    monkeypatch.setattr(driver, "_invoke_agent", fake_invoke)
    return driver, seen


def test_cli_driver_passes_the_deadline_and_the_timeout(tmp_path, monkeypatch):
    driver, seen = _recording_driver(tmp_path, monkeypatch)
    driver.config = type("C", (), {"agent_timeout_s": 600})()
    driver.run(_failing_task(tmp_path))
    assert 590 <= seen["timeout"] <= 600
    deadline = int(seen["env"]["FOMTOOLS_DEADLINE"])
    assert abs(deadline - (time.time() + 600 - GRACE_S)) <= 5


def test_a_task_timeout_replaces_the_phase_budget(tmp_path, monkeypatch):
    driver, seen = _recording_driver(tmp_path, monkeypatch)
    driver.config = type("C", (), {"agent_optimize_timeout_s": 7200})()
    task = _failing_task(tmp_path)
    task.phase = "code_optimization"
    task.extra["timeout_s"] = 30
    driver.run(task)
    assert seen["timeout"] <= 30


def test_without_a_budget_no_deadline_is_set(tmp_path, monkeypatch):
    driver, seen = _recording_driver(tmp_path, monkeypatch)
    driver.run(_failing_task(tmp_path))
    assert seen["timeout"] is None
    assert seen["env"] is None  # no key, no deadline: the CLI inherits the environment


def _claude_hook_command() -> str:
    flags = get_spec("claude").flags
    settings = json.loads(flags[flags.index("--settings") + 1])
    return settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


@pytest.mark.parametrize(
    ("offset", "code"),
    [(-10, 2), (3600, 0), (None, 0)],
    ids=["deadline-passed", "deadline-ahead", "no-deadline"],
)
def test_claude_hook_refuses_tools_after_the_deadline(offset, code):
    env = {"PATH": "/usr/bin:/bin"}
    if offset is not None:
        env["FOMTOOLS_DEADLINE"] = str(int(time.time()) + offset)
    proc = subprocess.run(
        ["sh", "-c", _claude_hook_command()], env=env, capture_output=True, text=True
    )
    assert proc.returncode == code
    assert ("print your summary" in proc.stderr) == (code == 2)
