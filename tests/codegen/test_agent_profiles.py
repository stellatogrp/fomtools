"""The bundled agent profiles: the defaults, pinned, and the restricted variants."""

import json
from dataclasses import replace

import pytest

from fomtools.codegen import cli
from fomtools.codegen.agent import available_backends, get_default_driver, make_agent_driver
from fomtools.codegen.agent.cli_specs import SYSTEM_PROMPT, get_spec

DEADLINE_HOOK = (
    '[ -z "$FOMTOOLS_DEADLINE" ] || [ "$(date +%s)" -lt "$FOMTOOLS_DEADLINE" ] || '
    '{ echo "fomtools: the time budget is spent. Stop editing and print your summary now." '
    ">&2; exit 2; }"
)
HOOKS = {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": DEADLINE_HOOK}]}]}
FOREGROUND = {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1"}
CLAUDE_COMMON = ["-p", "--setting-sources", "", "--strict-mcp-config"]
CLAUDE_COMMON += ["--output-format", "stream-json", "--verbose"]
SANDBOX = {
    "enabled": True,
    "failIfUnavailable": True,
    "allowUnsandboxedCommands": False,
    "autoAllowBashIfSandboxed": False,
}
BASH_RULES = "Bash(make *),Bash(./verify_bin *),Bash(./bench_bin *),Bash(./profile_bin *)"


def _argv(name: str) -> list[str]:
    return make_agent_driver(name)._build_argv("/w", "PROMPT")


def _split_settings(argv: list[str]) -> tuple[list[str], dict]:
    """The argv with the ``--settings`` JSON taken out, and that JSON parsed."""
    at = argv.index("--settings")
    return argv[:at] + argv[at + 2 :], json.loads(argv[at + 1])


@pytest.fixture(autouse=True)
def _no_model_from_env(monkeypatch):
    monkeypatch.delenv("FOMTOOLS_CODEGEN_MODEL", raising=False)


# --- the defaults, pinned: an edit must not loosen or tighten them unnoticed ---


def test_default_claude_argv_is_pinned():
    argv, settings = _split_settings(_argv("claude"))
    assert argv == [
        "claude",
        *CLAUDE_COMMON,
        *["--permission-mode", "bypassPermissions"],
        *["--allowedTools", "Read,Write,Edit,Bash,Glob,Grep"],
        *["--add-dir", "/w", "--append-system-prompt", SYSTEM_PROMPT, "PROMPT"],
    ]
    assert settings == {"env": FOREGROUND, "hooks": HOOKS}


def test_default_codex_pi_and_cursor_argv_are_pinned():
    prompt = SYSTEM_PROMPT + "\n\nPROMPT"
    assert _argv("codex") == [
        *["codex", "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"],
        *["--ignore-user-config", "--ignore-rules", "--ephemeral", "--cd", "/w", prompt],
    ]
    assert _argv("pi") == ["pi", "--print", "--approve", prompt]
    assert _argv("cursor") == ["cursor-agent", "--print", "--force", prompt]


# --- the restricted variants --------------------------------------------------


def test_restricted_profiles_are_selectable_backends():
    assert {"claude-sandboxed", "codex-sandboxed"} <= set(available_backends())
    # Only the CLIs that can confine the agent have one.
    assert sorted(n for n in available_backends() if n.endswith("-sandboxed")) == [
        "claude-sandboxed",
        "codex-sandboxed",
    ]


def test_claude_sandboxed_argv():
    argv, settings = _split_settings(_argv("claude-sandboxed"))
    assert argv == [
        "claude",
        *CLAUDE_COMMON,
        *["--permission-mode", "acceptEdits", "--permission-prompts", "none"],
        *["--tools", "Read,Write,Edit,Bash,Glob,Grep"],
        *["--allowedTools", BASH_RULES],
        *["--add-dir", "/w", "--append-system-prompt", SYSTEM_PROMPT, "PROMPT"],
    ]
    assert "bypassPermissions" not in argv and "--dangerously-skip-permissions" not in argv
    # The deadline hook and the foreground setting are the default entry's.
    assert settings == {"env": FOREGROUND, "sandbox": SANDBOX, "hooks": HOOKS}


def test_claude_sandboxed_allows_only_what_the_targets_run():
    flags = get_spec("claude-sandboxed").flags
    rules = flags[flags.index("--allowedTools") + 1].split(",")
    # No rule for a whole tool: a bare Bash, Write or Edit would cover any
    # command or path. Edits come from acceptEdits, inside the output dir only.
    assert all(r.startswith("Bash(") and r.endswith(" *)") for r in rules)
    allowed = {r[len("Bash(") : -len(" *)")] for r in rules}
    assert allowed == {"make", "./verify_bin", "./bench_bin", "./profile_bin"}
    # Every command the targets tell the agent to run is a `make` target.
    from fomtools.codegen.config import Backend, CodegenConfig, Version
    from fomtools.codegen.targets import get_target

    for backend in Backend:
        version = Version.EMBEDDED if backend == Backend.C else Version.LARGE_SCALE
        target = get_target(CodegenConfig(version=version, backend=backend))
        assert target.verify_command()[0] == "make"


def test_claude_sandboxed_differs_from_claude_only_in_permissions():
    default, restricted = get_spec("claude"), get_spec("claude-sandboxed")
    same = replace(restricted, name="claude", flags=default.flags, display_name="Claude Code")
    assert same == default


def test_codex_sandboxed_argv():
    argv = _argv("codex-sandboxed")
    assert argv == [
        *["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write"],
        *["--ignore-user-config", "--ignore-rules", "--ephemeral"],
        *["--cd", "/w", SYSTEM_PROMPT + "\n\nPROMPT"],
    ]
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    default, restricted = get_spec("codex"), get_spec("codex-sandboxed")
    same = replace(restricted, name="codex", flags=default.flags, display_name="Codex")
    assert same == default


# --- selecting a profile by name ---------------------------------------------


def test_profile_selected_by_env_var(monkeypatch):
    monkeypatch.delenv("FOMTOOLS_CODEGEN_AGENT", raising=False)
    monkeypatch.setenv("FOMTOOLS_CODEGEN_BACKEND", "claude-sandboxed")
    monkeypatch.setattr("fomtools.codegen.agent.shutil.which", lambda name: f"/usr/bin/{name}")
    assert get_default_driver().spec.name == "claude-sandboxed"
    assert make_agent_driver().spec.name == "claude-sandboxed"
    # An explicit name wins over the environment.
    assert make_agent_driver("codex-sandboxed").spec.name == "codex-sandboxed"


def test_profile_selected_on_the_command_line(monkeypatch):
    monkeypatch.delenv("FOMTOOLS_CODEGEN_BACKEND", raising=False)
    parser = cli.build_parser()
    args = parser.parse_args(["--example", "ista", "--out", "o", "--agent", "codex-sandboxed"])
    driver = cli._make_driver(args, cli._build_config(args))
    assert driver.spec.name == "codex-sandboxed" and "--sandbox" in driver.spec.flags
    assert "claude-sandboxed" in parser.format_help()
