"""Coding-agent drivers for code generation."""

from __future__ import annotations

import os

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
)
from fomtools.codegen.agent.stub import StubAgentDriver

__all__ = [
    "AGENT_BACKENDS",
    "AgentBackendError",
    "AgentDriver",
    "AgentResult",
    "AgentTask",
    "StubAgentDriver",
    "get_default_driver",
    "make_agent_driver",
]

# Selectable real coding-agent backends. "claude" is the default.
AGENT_BACKENDS = ("claude", "codex", "gemini")


def _resolve_backend(backend: str | None) -> str:
    """Resolve the agent backend from arg, then env, then the default."""
    backend = backend or os.environ.get("FOMTOOLS_CODEGEN_BACKEND") or "claude"
    if backend not in AGENT_BACKENDS:
        raise ValueError(
            f"unknown agent backend {backend!r}; choose from {list(AGENT_BACKENDS)}"
        )
    return backend


def make_agent_driver(backend: str | None = None, config=None) -> AgentDriver:
    """Construct a real coding-agent driver for the selected backend.

    ``backend`` is one of :data:`AGENT_BACKENDS`. When ``None``, it is read from
    ``FOMTOOLS_CODEGEN_BACKEND`` and otherwise defaults to ``"claude"`` (the
    original behavior). These backends are nondeterministic and need auth, so
    they never run in CI.
    """
    backend = _resolve_backend(backend)
    if backend == "codex":
        from fomtools.codegen.agent.codex_cli import CodexAgentDriver

        return CodexAgentDriver(config=config)
    if backend == "gemini":
        from fomtools.codegen.agent.gemini_cli import GeminiAgentDriver

        return GeminiAgentDriver(config=config)
    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

    return ClaudeAgentDriver(config=config)


def get_default_driver(config=None, backend: str | None = None) -> AgentDriver:
    """Return the default coding agent.

    A real backend is used only when explicitly enabled via
    ``FOMTOOLS_CODEGEN_AGENT=1`` (they are nondeterministic and need auth).
    Without that, callers must pass an explicit driver (e.g. a
    :class:`StubAgentDriver`). The backend (``claude`` by default, or ``codex`` /
    ``gemini``) is selected via ``backend`` / ``FOMTOOLS_CODEGEN_BACKEND``.
    """
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1":
        return make_agent_driver(backend=backend, config=config)
    raise RuntimeError(
        "No coding-agent driver provided. Either pass driver=... (e.g. "
        "StubAgentDriver) or set FOMTOOLS_CODEGEN_AGENT=1 to use a real "
        "coding-agent backend (claude, codex, or gemini)."
    )
