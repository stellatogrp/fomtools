"""Declarative specs for headless coding-agent CLIs + a runtime registry.

Most real backends are the same shape: a non-interactive CLI that takes a prompt,
edits files inside a working directory, and loops internally until done. They
differ only in *data* — the executable, a couple of flags, how the prompt and
working directory are passed, which env var holds the API key, and the strings
that mark an auth failure. :class:`CliAgentSpec` captures exactly those
differences so a single generic driver
(:class:`~fomtools.codegen.agent.cli_driver.CliAgentDriver`) can drive any of
them.

This makes code generation agent-agnostic: built-in specs cover Codex and
Gemini, and users add *their own* agents with no code changes — either at runtime
via :func:`register_agent`, or declaratively via a YAML config file (see
:func:`load_user_specs`). Anything that is not a prompt-in/files-out CLI (an
in-process SDK, an HTTP API) is still served by implementing the
:class:`~fomtools.codegen.agent.driver.AgentDriver` ABC directly — e.g. the
Claude SDK driver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

# The translation framing every backend shares. Backends with a system-prompt
# flag pass it through that flag; the rest get it prepended to the task prompt.
SYSTEM_PROMPT = (
    "You are an expert in high-performance numerical C. You translate a "
    "specified first-order optimization method into a correct, allocation-free, "
    "division-free embedded C solver and iterate until it passes verification."
)

# Substrings (matched case-insensitively against the CLI's own error output) that
# mark a non-recoverable authentication failure rather than a fixable task error.
# Specs may extend this with backend-specific markers.
DEFAULT_AUTH_MARKERS: tuple[str, ...] = (
    "unauthorized",
    "401",
    "403",
    "invalid api key",
    "incorrect api key",
    "missing api key",
    "no api key",
    "api key not valid",
    "api_key_invalid",
    "not logged in",
    "please log in",
    "authentication",
)
_REGISTRY = {}


@dataclass(frozen=True)
class CliAgentSpec:
    """How to drive one headless coding-agent CLI.

    Everything that varies between Codex, Gemini, and any user-supplied agent is
    a field here; the invocation logic lives once in
    :class:`~fomtools.codegen.agent.cli_driver.CliAgentDriver`.

    Attributes
    ----------
    name:
        Backend id selected via ``--agent <name>`` / ``FOMTOOLS_CODEGEN_BACKEND``.
    bin:
        Default executable (overridable per driver instance).
    base_args:
        Fixed leading subcommand args, e.g. ``("exec",)`` for ``codex exec``.
    flags:
        Fixed flags (sandbox/approval bypass, etc.), e.g. ``("--yolo",)``.
    model_flag:
        Flag that selects a model (``"--model"``); ``None`` if unsupported.
    prompt_arg:
        Flag the prompt is passed with (``"--prompt"``); ``None`` = positional.
    cwd_flag:
        Flag that tells the CLI which dir to operate in (``"--cd"``); ``None`` if
        it just uses the process working directory.
    sysprompt_flag:
        Flag for an extra system prompt (Claude's ``--append-system-prompt``);
        ``None`` means :data:`SYSTEM_PROMPT` is prepended to the task prompt.
    api_key_env:
        Primary env var holding the API key (``"OPENAI_API_KEY"``); a resolved
        key is forwarded here to force API-key auth over any on-disk login.
    api_key_aliases:
        Additional env vars to read a key from (resolution order after
        ``api_key_env``), e.g. ``("GOOGLE_API_KEY",)`` for Gemini.
    login_check_args:
        Subcommand args that report whether an on-disk login already exists,
        exiting 0 when authenticated (Codex's ``("login", "status")``); empty if
        the CLI has no such check.
    login_args:
        Subcommand args that persist an API-key login, reading the key from
        stdin (Codex's ``("login", "--with-api-key")``). Empty means the CLI
        needs no login step — the forwarded ``api_key_env`` is enough.
    auth_markers:
        Output substrings flagging an auth failure (merged with the defaults).
    display_name, auth_hint, install_hint:
        Human-facing strings used to build clear ``AgentBackendError`` messages.
    """

    name: str
    bin: str
    base_args: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    model_flag: str | None = "--model"
    prompt_arg: str | None = None
    cwd_flag: str | None = None
    sysprompt_flag: str | None = None
    api_key_env: str | None = None
    api_key_aliases: tuple[str, ...] = ()
    login_check_args: tuple[str, ...] = ()
    login_args: tuple[str, ...] = ()
    auth_markers: tuple[str, ...] = field(default_factory=tuple)
    display_name: str = ""
    auth_hint: str = ""
    install_hint: str = ""

    @property
    def all_auth_markers(self) -> tuple[str, ...]:
        """Backend-specific markers plus the shared defaults (de-duplicated)."""
        seen: dict[str, None] = {}
        for m in (*self.auth_markers, *DEFAULT_AUTH_MARKERS):
            seen.setdefault(m.lower(), None)
        return tuple(seen)

    @property
    def key_env_vars(self) -> tuple[str, ...]:
        """Env vars to read an API key from, in resolution order."""
        return tuple(v for v in (self.api_key_env, *self.api_key_aliases) if v)

_user_specs_loaded = False

# Field name -> coercion for the YAML loader (everything else is rejected so a
# typo surfaces loudly instead of being silently ignored).
_TUPLE_FIELDS = {
    "base_args",
    "flags",
    "api_key_aliases",
    "login_check_args",
    "login_args",
    "auth_markers",
}
_STR_FIELDS = {
    "bin",
    "model_flag",
    "prompt_arg",
    "cwd_flag",
    "sysprompt_flag",
    "api_key_env",
    "display_name",
    "auth_hint",
    "install_hint",
}


def register_agent(spec: CliAgentSpec) -> None:
    """Register (or override) a CLI agent spec under ``spec.name`` at runtime."""
    _REGISTRY[spec.name] = spec


def _config_paths() -> list[Path]:
    """Config files to load specs from (later wins on name collisions).

    Loaded in order, each overriding the previous on a name clash:

    1. the bundled ``agents.yaml`` shipped next to this module — editing it in
       place makes an agent available with no further setup;
    2. the per-user ``~/.config/fomtools/agents.yaml``;
    3. ``$FOMTOOLS_CODEGEN_AGENTS`` (an explicit path), which wins over all.
    """
    paths: list[Path] = [
        Path(__file__).resolve().parent / "agents.yaml",
        Path.home() / ".config" / "fomtools" / "agents.yaml",
    ]
    env = os.environ.get("FOMTOOLS_CODEGEN_AGENTS")
    if env:
        paths.append(Path(env).expanduser())
    return paths


def _spec_from_table(name: str, table: dict) -> CliAgentSpec:
    if "bin" not in table:
        raise ValueError(f"agent {name!r}: missing required key 'bin'")
    kwargs: dict = {"name": name}
    for key, value in table.items():
        if key in _TUPLE_FIELDS:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"agent {name!r}: {key!r} must be a list of strings")
            kwargs[key] = tuple(value)
        elif key in _STR_FIELDS:
            kwargs[key] = value
        else:
            raise ValueError(
                f"agent {name!r}: unknown key {key!r}; valid keys are "
                f"{sorted(_TUPLE_FIELDS | _STR_FIELDS)}"
            )
    return CliAgentSpec(**kwargs)


def load_user_specs(*, force: bool = False) -> None:
    """Load user-defined specs from the config files into the registry (once).

    Config format (YAML), one mapping per agent under a top-level ``agents:``::

        agents:
          aider:
            bin: aider
            flags: ["--yes", "--no-auto-commit"]
            prompt_arg: --message
            api_key_env: OPENAI_API_KEY
            display_name: Aider
            auth_hint: Set OPENAI_API_KEY.

    Built-in specs (``codex``, ``gemini``) can be overridden by name. Missing
    files are skipped; a malformed file raises so the user sees the problem.
    """
    global _user_specs_loaded
    if _user_specs_loaded and not force:
        return
    _user_specs_loaded = True
    for path in _config_paths():
        if not path.is_file():
            continue
        with path.open("rb") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level mapping with an 'agents:' key expected")
        agents = data.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError(f"{path}: 'agents' must be a mapping of name -> spec")
        for name, table in agents.items():
            if not isinstance(table, dict):
                raise ValueError(f"{path}: agent {name!r} must be a mapping")
            register_agent(_spec_from_table(name, table))


def get_spec(name: str) -> CliAgentSpec:
    """Return the spec registered under ``name`` (loading user config first)."""
    load_user_specs()
    if name not in _REGISTRY:
        raise KeyError(name)
    return _REGISTRY[name]


def available_specs() -> list[str]:
    """Names of all registered CLI agent specs (built-in + user), sorted."""
    load_user_specs()
    return sorted(_REGISTRY)


__all__ = [
    "SYSTEM_PROMPT",
    "DEFAULT_AUTH_MARKERS",
    "CliAgentSpec",
    "register_agent",
    "load_user_specs",
    "get_spec",
    "available_specs",
    "replace",
]
