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
    short = TimeBudget.starting_now(30.0)  # a fifth of a short budget, not GRACE_S
    assert short.grace_s == 6.0
    assert abs(short.agent_deadline() - (time.time() + 24)) <= 1


def test_no_retry_starts_inside_the_grace_period(tmp_path):
    calls = []

    def agent(prompt):
        calls.append(1)
        time.sleep(0.85)  # leaves less than the 0.2 s grace of a 1 s budget
        return "attempted"

    result = run_verify_loop(_failing_task(tmp_path), agent, max_attempts=5, budget_s=1.0)
    assert len(calls) == 1 and "budget for this phase is spent" in result.log


def test_a_driver_without_config_gets_the_default_budgets():
    from fomtools.codegen.config import CodegenConfig

    driver = CliAgentDriver(get_spec("claude"))
    assert driver._timeout_for("translation") == CodegenConfig().agent_timeout_s
    assert driver._timeout_for("code_optimization") == CodegenConfig().agent_optimize_timeout_s


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
    driver.config = type("C", (), {"agent_timeout_s": None})()
    driver.run(_failing_task(tmp_path))
    assert seen["timeout"] is None
    assert seen["env"] is None  # no deadline: the CLI inherits the environment


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


def _script_driver(tmp_path, code: str) -> CliAgentDriver:
    """A driver whose "CLI" is ``python -c code``."""
    spec = CliAgentSpec(name="fake", bin=sys.executable, base_args=("-c", code), model_flag=None)
    return CliAgentDriver(spec)


def test_numbers_in_a_failing_transcript_are_not_an_auth_error(tmp_path):
    driver = _script_driver(
        tmp_path, "import sys; print('solve time 1.4031 ms; 401 iterations'); sys.exit(1)"
    )
    out = driver._invoke_agent(driver._build_argv(str(tmp_path), "p"), tmp_path)
    assert "401 iterations" in out


def test_an_auth_failure_at_the_end_fails_fast(tmp_path):
    from fomtools.codegen.agent.driver import AgentBackendError

    driver = _script_driver(tmp_path, "import sys; print('Error: Invalid API key'); sys.exit(1)")
    with pytest.raises(AgentBackendError, match="could not authenticate"):
        driver._invoke_agent(driver._build_argv(str(tmp_path), "p"), tmp_path)


def test_a_timeout_also_ends_what_the_agent_started(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time;"
        f"c = subprocess.Popen(['sleep', '60']); open({str(pid_file)!r}, 'w').write(str(c.pid));"
        "time.sleep(60)"
    )
    driver = _script_driver(tmp_path, code)
    out = driver._invoke_agent(driver._build_argv(str(tmp_path), "p"), tmp_path, timeout=2)
    assert "exceeded its 2s budget" in out
    child = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(child, 0)


def test_a_model_for_a_spec_without_a_model_flag_warns(tmp_path):
    driver = _script_driver(tmp_path, "pass")
    driver.model = "some-model"
    with pytest.warns(UserWarning, match="no model_flag"):
        driver._build_argv(str(tmp_path), "p")


def test_a_timeout_sends_sigterm_first(tmp_path):
    # The CLI traps SIGTERM and writes a marker: it got the chance to clean up.
    marker = tmp_path / "termed"
    code = (
        "import signal, sys, time;"
        f"signal.signal(signal.SIGTERM, lambda *a: (open({str(marker)!r}, 'w').close(), sys.exit(0)));"
        "time.sleep(60)"
    )
    driver = _script_driver(tmp_path, code)
    # 5 s so the handler is installed before the timeout, even on a loaded machine.
    driver._invoke_agent(driver._build_argv(str(tmp_path), "p"), tmp_path, timeout=5)
    assert marker.exists()


def test_the_agent_starts_with_no_blocked_signals(tmp_path):
    # As after `import cvxpygen`, which blocks SIGINT and SIGTERM in this thread.
    import signal

    old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        test_a_timeout_sends_sigterm_first(tmp_path)
        assert signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])  # restored
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def test_claude_runs_isolated_from_user_settings():
    flags = get_spec("claude").flags
    assert flags[flags.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in flags


def test_specs_no_longer_take_api_keys():
    from fomtools.codegen.agent.cli_specs import _spec_from_table

    with pytest.raises(ValueError, match="unknown key 'api_key_env'"):
        _spec_from_table("x", {"bin": "x", "api_key_env": "OPENAI_API_KEY"})


def test_removed_max_turns_option_warns():
    from fomtools.codegen.config import CodegenConfig

    with pytest.warns(FutureWarning, match="agent_optimize_max_turns was removed"):
        config = CodegenConfig(agent_optimize_max_turns=400)
    assert not hasattr(config, "agent_optimize_max_turns")


def _event(kind, **fields):
    return json.dumps({"type": kind, **fields})


def _say(text):
    return _event("assistant", message={"content": [{"type": "text", "text": text}]})


def test_stream_json_keeps_the_agent_text():
    from fomtools.codegen.agent.cli_driver import stream_json_text

    tool = _event("assistant", message={"content": [{"type": "tool_use", "name": "Bash"}]})
    stream = "\n".join(
        [
            _event("system", subtype="init"),
            _say("PLAN: fuse the matvecs."),
            tool,
            _event("user", message={"content": [{"type": "tool_result", "content": "x"}]}),
            _say("DONE"),
            _event("result", subtype="success", result="DONE", is_error=False),
        ]
    )
    assert stream_json_text(stream) == "PLAN: fuse the matvecs.\nDONE\n"
    failed = _event("result", subtype="success", result="Invalid API key", is_error=True)
    assert stream_json_text("Error: no prompt\n" + failed) == "Error: no prompt\nInvalid API key\n"
    assert stream_json_text(_say("half") + '\n{"type": "assist') == 'half\n{"type": "assist\n'


def test_a_stream_json_agent_ended_at_its_timeout_keeps_its_notes(tmp_path):
    code = f"import time; print({_say('round 1: hoisted the constant term')!r}, flush=True); time.sleep(60)"
    spec = CliAgentSpec(
        name="fake",
        bin=sys.executable,
        base_args=("-c", code),
        model_flag=None,
        output_format="stream-json",
    )
    driver = CliAgentDriver(spec)
    out = driver._invoke_agent(driver._build_argv(str(tmp_path), "p"), tmp_path, timeout=3)
    assert out.startswith("round 1: hoisted the constant term\n")
    assert "exceeded its 3s budget" in out and '"type"' not in out


def test_claude_streams_json():
    spec = get_spec("claude")
    assert spec.output_format == "stream-json"
    assert "stream-json" in spec.flags and "--verbose" in spec.flags


def test_claude_keeps_commands_in_the_foreground():
    # A background task keeps `claude -p` waiting after its reply (#166).
    flags = get_spec("claude").flags
    settings = json.loads(flags[flags.index("--settings") + 1])
    assert settings["env"] == {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1"}
    from fomtools.codegen.agent.cli_specs import SYSTEM_PROMPT

    assert "foreground" in SYSTEM_PROMPT and "nohup" in SYSTEM_PROMPT


def test_what_the_agent_left_running_is_ended_when_it_exits(tmp_path):
    # As `nohup make bench &`: a child that outlives the CLI, not holding its pipes.
    import os

    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess;"
        "c = subprocess.Popen(['sleep', '60'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);"
        f"open({str(pid_file)!r}, 'w').write(str(c.pid)); print('done')"
    )
    driver = _script_driver(tmp_path, code)
    start = time.monotonic()
    out = driver._invoke_agent(driver._build_argv(str(tmp_path), "p"), tmp_path, timeout=30)
    assert out.strip() == "done" and time.monotonic() - start < 10
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_specs_load_once_across_threads(monkeypatch):
    # A second thread used to find the "loaded" flag set and the registry empty.
    import threading

    from fomtools.codegen.agent import cli_specs

    monkeypatch.setattr(cli_specs, "_REGISTRY", {})
    monkeypatch.setattr(cli_specs, "_user_specs_loaded", False)
    paths = cli_specs._config_paths
    monkeypatch.setattr(cli_specs, "_config_paths", lambda: (time.sleep(0.2), paths())[1])
    errors = []

    def lookup():
        try:
            get_spec("claude")
        except KeyError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=lookup) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_codex_runs_without_the_user_config():
    flags = get_spec("codex").flags
    assert {"--ignore-user-config", "--ignore-rules", "--ephemeral"} <= set(flags)
