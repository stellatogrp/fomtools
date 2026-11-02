"""Coding-agent drivers for code generation.

Code generation is **agent-agnostic**: every agent is a headless CLI described by
a declarative `CliAgentSpec` and driven by the generic `CliAgentDriver`. Specs
for ``claude`` (the default), ``codex``, ``pi`` (https://pi.dev), and ``cursor``
ship in ``agents.yaml``; users add their own agents with no code changes, at
runtime via `register_agent` or in a YAML config (see `load_user_specs`). Each
agent runs through its own installed CLI, so it uses the user's login, config,
and default model; set ``CodegenConfig.agent_model`` or ``FOMTOOLS_CODEGEN_MODEL``
to pick another model. Anything that is not a CLI can implement the
`AgentDriver` ABC directly.
"""

from __future__ import annotations

import os

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
)
from fomtools.codegen.agent.cli_specs import (
    CliAgentSpec,
    available_specs,
    register_agent,
)
from fomtools.codegen.agent.stub import StubAgentDriver
from fomtools.codegen.agent.cli_driver import CliAgentDriver
from fomtools.codegen.agent.cli_specs import get_spec

__all__ = [
    "AgentBackendError",
    "AgentDriver",
    "AgentResult",
    "AgentTask",
    "CliAgentDriver",
    "CliAgentSpec",
    "StubAgentDriver",
    "available_backends",
    "get_default_driver",
    "make_agent_driver",
    "register_agent",
]

_DEFAULT_BACKEND = "claude"


def available_backends() -> list[str]:
    """All selectable backends: every registered CLI spec (bundled and user)."""
    return available_specs()


def make_agent_driver(backend: str | None = None, config=None) -> AgentDriver:
    """Construct a real coding-agent driver for the selected backend.

    ``backend`` is any name from `available_backends` (bundled ``claude`` /
    ``codex`` / ``pi`` / ``cursor``, or a user-registered CLI agent). When
    ``None`` it is read from ``FOMTOOLS_CODEGEN_BACKEND`` and otherwise defaults
    to ``"claude"``. The model is ``config.agent_model``, else
    ``FOMTOOLS_CODEGEN_MODEL``, else the CLI's own default. These backends are
    nondeterministic and need auth, so they never run in CI.
    """
    backend = backend or os.environ.get("FOMTOOLS_CODEGEN_BACKEND") or _DEFAULT_BACKEND
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
    `StubAgentDriver`). The backend (``claude`` by default, or any
    registered CLI agent) is selected via ``backend`` /
    ``FOMTOOLS_CODEGEN_BACKEND``.
    """
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1":
        return make_agent_driver(backend=backend, config=config)
    raise RuntimeError(
        "No coding-agent driver provided. Either pass driver=... (e.g. "
        "StubAgentDriver) or set FOMTOOLS_CODEGEN_AGENT=1 to use a real "
        "coding-agent backend (claude, codex, pi, or a custom CLI agent)."
    )
