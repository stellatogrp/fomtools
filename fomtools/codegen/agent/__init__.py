"""Coding-agent drivers for code generation.

Code generation is **agent-agnostic**: any prompt-in/files-out CLI agent is
supported via a declarative :class:`~fomtools.codegen.agent.specs.CliAgentSpec`
driven by the generic :class:`~fomtools.codegen.agent.cli_driver.CliAgentDriver`.
Built-in specs cover ``codex`` and ``gemini``; users add their own agents with no
code changes — at runtime via :func:`~fomtools.codegen.agent.specs.register_agent`
or declaratively via a YAML config (see
:func:`~fomtools.codegen.agent.specs.load_user_specs`). ``claude`` is served by
the dedicated :class:`~fomtools.codegen.agent.claude_sdk.ClaudeAgentDriver`
(in-process SDK with a CLI fallback). Anything that is not a CLI can implement the
:class:`~fomtools.codegen.agent.driver.AgentDriver` ABC directly.
"""

from __future__ import annotations

import os

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
)
from fomtools.codegen.agent.specs import (
    CliAgentSpec,
    available_specs,
    register_agent,
)
from fomtools.codegen.agent.stub import StubAgentDriver
from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver
from fomtools.codegen.agent.cli_driver import CliAgentDriver
from fomtools.codegen.agent.specs import get_spec
from fomtools.codegen.agent.codex_cli import CodexAgentDriver
from fomtools.codegen.agent.gemini_cli import GeminiAgentDriver

__all__ = [
    "AgentBackendError",
    "AgentDriver",
    "AgentResult",
    "AgentTask",
    "CliAgentSpec",
    "StubAgentDriver",
    "available_backends",
    "get_default_driver",
    "make_agent_driver",
    "register_agent",
]

# Claude has its own (non-CLI) driver; everything else is a CLI spec in the
# registry. Default backend is "claude".
_DEFAULT_BACKEND = "claude"


def available_backends() -> list[str]:
    """All selectable backends: ``claude`` plus every registered CLI agent."""
    return [_DEFAULT_BACKEND, *available_specs()]


def make_agent_driver(backend: str | None = None, config=None) -> AgentDriver:
    """Construct a real coding-agent driver for the selected backend.

    ``backend`` is any name from :func:`available_backends` (built-in
    ``claude`` / ``codex`` / ``gemini`` or a user-registered CLI agent). When
    ``None`` it is read from ``FOMTOOLS_CODEGEN_BACKEND`` and otherwise defaults
    to ``"claude"``. These backends are nondeterministic and need auth, so they
    never run in CI.
    """
    backend = backend or os.environ.get("FOMTOOLS_CODEGEN_BACKEND") or _DEFAULT_BACKEND
    match backend:
        case "claude":
            return ClaudeAgentDriver(config=config)
        case "codex":
            return CodexAgentDriver(config=config)
        case "gemini":
            return GeminiAgentDriver(config=config)

    try:
        spec = get_spec(backend)
    except KeyError:
        raise ValueError(
            f"unknown agent backend {backend!r}; choose from {available_backends()} "
            "or define one in ~/.config/fomtools/agents.yaml (or $FOMTOOLS_CODEGEN_AGENTS)"
        ) from None
    return CliAgentDriver(spec, config=config)


def get_default_driver(config=None, backend: str | None = None) -> AgentDriver:
    """Return the default coding agent.

    A real backend is used only when explicitly enabled via
    ``FOMTOOLS_CODEGEN_AGENT=1`` (they are nondeterministic and need auth).
    Without that, callers must pass an explicit driver (e.g. a
    :class:`StubAgentDriver`). The backend (``claude`` by default, or any
    registered CLI agent) is selected via ``backend`` /
    ``FOMTOOLS_CODEGEN_BACKEND``.
    """
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1":
        return make_agent_driver(backend=backend, config=config)
    raise RuntimeError(
        "No coding-agent driver provided. Either pass driver=... (e.g. "
        "StubAgentDriver) or set FOMTOOLS_CODEGEN_AGENT=1 to use a real "
        "coding-agent backend (claude, codex, gemini, or a custom CLI agent)."
    )
