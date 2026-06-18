"""Google Gemini CLI backend.

A thin convenience wrapper over the generic
:class:`~fomtools.codegen.agent.cli_driver.CliAgentDriver`, bound to
:data:`~fomtools.codegen.agent.specs.GEMINI_SPEC`. The invocation logic, the
shared verify/retry loop, and API-key handling all live in the generic driver;
this class only exposes Gemini-specific constructor knobs (``gemini_bin``,
``yolo``) and resolves ``GEMINI_API_KEY`` (then ``GOOGLE_API_KEY``).

Like every real backend it is nondeterministic and needs auth (``GEMINI_API_KEY``
or a logged-in ``gemini`` CLI), so it never runs in CI; the deterministic
pipeline is tested via :class:`~fomtools.codegen.agent.stub.StubAgentDriver`.
"""

from __future__ import annotations

from fomtools.codegen.agent.cli_driver import CliAgentDriver
from fomtools.codegen.agent.specs import GEMINI_SPEC, replace


class GeminiAgentDriver(CliAgentDriver):
    """Drive code generation with the Google ``gemini`` CLI (``gemini -p``)."""

    def __init__(
        self,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        max_turns: int = 16,
        gemini_bin: str = "gemini",
        yolo: bool = True,
        api_key: str | None = None,
    ):
        spec = GEMINI_SPEC
        # --yolo auto-approves writes/`make` (safe in the isolated output dir);
        # set False to keep the default approval mode.
        if not yolo:
            spec = replace(spec, flags=())
        super().__init__(
            spec,
            config=config,
            model=model,
            max_attempts=max_attempts,
            max_turns=max_turns,
            bin=gemini_bin,
            api_key=api_key,
        )
