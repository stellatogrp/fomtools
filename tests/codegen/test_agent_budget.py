"""The wall-clock budget of an agent phase (``fomtools.codegen.agent.budget``).

The hooks are plain async callables over a clock, so they are exercised directly
with a fake clock; nothing in the agent SDK is replaced.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from fomtools.codegen.agent.budget import GRACE_S, TimeBudget
from fomtools.codegen.agent.driver import AgentTask, run_verify_loop


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _budget(total_s=600.0):
    clock = FakeClock()
    return TimeBudget.starting_now(total_s, clock=clock), clock


def _call(hook, tool_name="Read", tool_input=None):
    data = {"tool_name": tool_name, "tool_input": tool_input or {}}
    return asyncio.run(hook(data, "tool-1", None))


def test_the_agent_is_told_the_time_left_after_each_tool():
    budget, clock = _budget(600)
    clock.now = 140
    out = _call(budget.after_tool)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    assert "7m 40s of 10m 00s remain" in out["additionalContext"]


def test_a_bash_timeout_is_capped_at_the_time_left():
    budget, clock = _budget(600)
    clock.now = 570  # 30 s left
    out = _call(budget.before_tool, "Bash", {"command": "make bench", "timeout": 600_000})
    spec = out["hookSpecificOutput"]
    assert spec["updatedInput"] == {"command": "make bench", "timeout": 30_000}
    # A timeout rewrite must not approve the command past the permission check.
    assert "permissionDecision" not in spec


@pytest.mark.parametrize("timeout", ["soon", None, [1]])
def test_a_non_numeric_bash_timeout_is_replaced(timeout):
    budget, clock = _budget(600)
    clock.now = 570
    spec = _call(budget.before_tool, "Bash", {"command": "make", "timeout": timeout})[
        "hookSpecificOutput"
    ]
    assert spec["updatedInput"]["timeout"] == 30_000
    assert "permissionDecision" not in spec


def test_a_bash_command_with_no_timeout_gets_one():
    budget, clock = _budget(600)
    clock.now = 300
    spec = _call(budget.before_tool, "Bash", {"command": "make bench"})["hookSpecificOutput"]
    assert spec["updatedInput"]["timeout"] == 300_000


def test_a_short_bash_command_and_other_tools_pass_unchanged():
    budget, clock = _budget(600)
    assert _call(budget.before_tool, "Bash", {"command": "ls", "timeout": 5_000}) == {}
    assert _call(budget.before_tool, "Edit", {"file_path": "src/fom_solver.c"}) == {}
    assert not budget.exhausted


def test_tools_are_denied_once_the_budget_is_spent():
    budget, clock = _budget(600)
    clock.now = 601
    spec = _call(budget.before_tool, "Bash", {"command": "make bench"})["hookSpecificOutput"]
    assert spec["permissionDecision"] == "deny"
    assert "summary" in spec["permissionDecisionReason"]
    assert budget.exhausted
    assert "exhausted" in _call(budget.after_tool)["hookSpecificOutput"]["additionalContext"]


def test_the_session_ends_after_the_grace_period():
    budget, clock = _budget(600)
    clock.now = 600 + GRACE_S + 1
    out = _call(budget.before_tool, "Edit")
    assert out["continue_"] is False
    assert "budget" in out["stopReason"]


def test_sdk_options_carry_the_budget_hooks(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions

    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

    task = AgentTask(workdir=tmp_path, solver_rel_path="src/fom_solver.c", verify_cmd=["true"])
    driver = ClaudeAgentDriver()
    assert "hooks" not in driver._sdk_options(task, max_turns=5)

    budget, _ = _budget(600)
    task.extra["budget"] = budget
    options = driver._sdk_options(task, max_turns=5)
    ClaudeAgentOptions(**options)  # accepted by the SDK as is
    pre = options["hooks"]["PreToolUse"][0].hooks
    post = options["hooks"]["PostToolUse"][0].hooks
    assert pre == [budget.before_tool] and post == [budget.after_tool]


def test_no_new_attempt_starts_once_the_phase_budget_is_spent(tmp_path):
    task = AgentTask(
        workdir=tmp_path,
        solver_rel_path="src/fom_solver.c",
        verify_cmd=[sys.executable, "-c", "raise SystemExit(1)"],  # always fails
    )
    calls = []

    def slow_agent(prompt, max_turns):
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
    task = AgentTask(
        workdir=Path(tmp_path),
        solver_rel_path="src/fom_solver.c",
        verify_cmd=[sys.executable, "-c", "raise SystemExit(1)"],
    )
    result = run_verify_loop(task, lambda prompt, turns: "attempted", max_attempts=3)
    assert result.turns_used == 3
    assert "budget" not in task.extra
