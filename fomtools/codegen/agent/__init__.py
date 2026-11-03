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
import shutil

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
    """Return the coding agent used when no driver is passed.

    This is the real CLI agent from `make_agent_driver`: ``claude`` by default,
    or the one selected via ``backend`` / ``FOMTOOLS_CODEGEN_BACKEND``. Its CLI
    must be installed and logged in. Set ``FOMTOOLS_CODEGEN_AGENT=0`` to disable
    real agents (the test suite does), so that a call without a driver fails
    instead of running a nondeterministic, paid agent.

    Raises
    ------
    RuntimeError
        If real agents are disabled, or the backend's CLI is not on PATH.
    """
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "0":
        raise RuntimeError(
            "No coding-agent driver provided and real agents are disabled "
            "(FOMTOOLS_CODEGEN_AGENT=0). Pass driver=... (e.g. StubAgentDriver) "
            "or unset FOMTOOLS_CODEGEN_AGENT."
        )
    driver = make_agent_driver(backend=backend, config=config)
    if shutil.which(driver.bin) is None:
        spec = driver.spec
        raise RuntimeError(
            f"No coding-agent driver provided and the {spec.display_name or spec.name} "
            f"CLI {driver.bin!r} is not on PATH. {spec.install_hint} Or choose another "
            "agent with FOMTOOLS_CODEGEN_BACKEND (claude, codex, pi, cursor), or pass "
            "driver=..."
        )
    return driver
