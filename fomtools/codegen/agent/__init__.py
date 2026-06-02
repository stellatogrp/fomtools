"""Coding-agent drivers for code generation."""

from __future__ import annotations

import os

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask
from fomtools.codegen.agent.stub import StubAgentDriver

__all__ = [
    "AgentDriver",
    "AgentResult",
    "AgentTask",
    "StubAgentDriver",
    "get_default_driver",
]


def get_default_driver(config=None) -> AgentDriver:
    """Return the default coding agent.

    The real Claude Agent SDK driver is used only when explicitly enabled via
    ``FOMTOOLS_CODEGEN_AGENT=1`` (it is nondeterministic and needs auth). Without
    that, callers must pass an explicit driver (e.g. a
    :class:`StubAgentDriver`).
    """
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") == "1":
        from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

        return ClaudeAgentDriver(config=config)
    raise RuntimeError(
        "No coding-agent driver provided. Either pass driver=... (e.g. "
        "StubAgentDriver) or set FOMTOOLS_CODEGEN_AGENT=1 to use the real "
        "Claude Agent SDK driver."
    )
