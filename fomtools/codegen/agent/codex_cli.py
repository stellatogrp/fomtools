"""OpenAI Codex CLI backend.

A thin convenience wrapper over the generic
:class:`~fomtools.codegen.agent.cli_driver.CliAgentDriver`, bound to
:data:`~fomtools.codegen.agent.specs.CODEX_SPEC`. The invocation logic, the
shared verify/retry loop, and API-key handling all live in the generic driver;
this class only exposes Codex-specific constructor knobs (``codex_bin``,
``bypass_sandbox``) and resolves ``OPENAI_API_KEY``.

Like every real backend it is nondeterministic and needs auth (``OPENAI_API_KEY``
or a logged-in ``codex`` CLI), so it never runs in CI; the deterministic pipeline
is tested via :class:`~fomtools.codegen.agent.stub.StubAgentDriver`.
"""

from __future__ import annotations

from fomtools.codegen.agent.cli_driver import CliAgentDriver
from fomtools.codegen.agent.specs import CODEX_SPEC, replace


class CodexAgentDriver(CliAgentDriver):
    """Drive code generation with the OpenAI ``codex`` CLI (``codex exec``)."""

    def __init__(
        self,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        max_turns: int = 16,
        codex_bin: str = "codex",
        bypass_sandbox: bool = True,
        api_key: str | None = None,
    ):
        spec = CODEX_SPEC
        # bypass_sandbox auto-approves writes/`make` (safe in the isolated output
        # dir); set False to keep Codex's workspace-write sandbox instead.
        if not bypass_sandbox:
            spec = replace(spec, flags=("--skip-git-repo-check", "--sandbox", "workspace-write"))
        super().__init__(
            spec,
            config=config,
            model=model,
            max_attempts=max_attempts,
            max_turns=max_turns,
            bin=codex_bin,
            api_key=api_key,
        )
