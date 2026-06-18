"""Real coding-agent driver backed by the Claude Agent SDK.

This driver is intentionally only used when explicitly enabled
(``FOMTOOLS_CODEGEN_AGENT=1``): it is nondeterministic, needs authentication
(``ANTHROPIC_API_KEY`` or a logged-in ``claude`` CLI), and therefore never runs
in CI. The deterministic pipeline is tested via
:class:`~fomtools.codegen.agent.stub.StubAgentDriver`.

It prefers the in-process Python SDK (``claude_agent_sdk``) and falls back to
shelling out to the ``claude`` CLI in headless mode. After each agent attempt it
authoritatively runs the verify command itself to decide success, and re-prompts
with the failure output until the command passes or the attempt budget is spent.

Authentication works in two modes:

* **Subscription / OAuth ("auth mode")** — a logged-in ``claude`` CLI. Nothing
  extra is needed; the SDK and CLI pick up the on-disk credential.
* **API key** — pass ``api_key=...`` or set ``ANTHROPIC_API_KEY``. Both the
  Claude Agent SDK and the ``claude`` CLI resolve an explicit
  ``ANTHROPIC_API_KEY`` *before* the on-disk subscription login, so we forward
  the key into the spawned process environment (``ClaudeAgentOptions.env`` for
  the in-process SDK, ``subprocess`` ``env`` for the CLI fallback) to force
  API-key auth even when a subscription login is also present.
"""

from __future__ import annotations

import os
import subprocess

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
    run_verify_loop,
)

_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# Substrings (matched case-insensitively in the CLI's own error output) that mark
# a non-recoverable authentication failure rather than a fixable task error.
_AUTH_FAILURE_MARKERS = (
    "invalid api key",
    "incorrect api key",
    "authentication_error",
    "authentication error",
    "invalid x-api-key",
    "please run /login",
    "please log in",
    "not logged in",
    "claude login",
    "unauthorized",
    "401",
)

_SYSTEM_PROMPT = (
    "You are an expert in high-performance numerical C. You translate a "
    "specified first-order optimization method into a correct, allocation-free, "
    "division-free embedded C solver and iterate until it passes verification."
)


class ClaudeAgentDriver(AgentDriver):
    """Drive code generation with the Claude Agent SDK (or the ``claude`` CLI)."""

    def __init__(
        self,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        max_turns: int = 16,
        prefer_cli: bool = False,
        cli_permission_mode: str = "bypassPermissions",
        api_key: str | None = None,
    ):
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        self.max_turns = max_turns
        self.prefer_cli = prefer_cli
        # Permission mode for the CLI fallback. bypassPermissions lets the agent
        # write files and run `make` non-interactively inside the isolated output
        # directory.
        self.cli_permission_mode = cli_permission_mode
        # Explicit Anthropic API key. When None we fall back to ANTHROPIC_API_KEY
        # from the environment; when neither is set the backend uses the on-disk
        # subscription/OAuth login instead ("auth mode"). A resolved key is
        # forwarded into the spawned process so it takes precedence over any
        # subscription login (matching Claude's own credential resolution order).
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def _child_env(self) -> dict[str, str] | None:
        """Process environment for the SDK/CLI, forcing API-key auth if we have one.

        Returns ``None`` when no key is configured so the backend keeps its
        default behavior (subscription/OAuth auth). Otherwise returns the full
        parent environment with ``ANTHROPIC_API_KEY`` injected, so PATH and the
        rest of the environment are preserved while the key forces API-key auth.
        """
        if not self.api_key:
            return None
        return {**os.environ, "ANTHROPIC_API_KEY": self.api_key}

    # -- public API --------------------------------------------------------
    def run(self, task: AgentTask) -> AgentResult:
        return run_verify_loop(
            task,
            lambda prompt, max_turns: self._invoke_agent(task, prompt, max_turns),
            max_attempts=self.max_attempts,
            max_turns=self.max_turns,
        )

    # -- agent invocation --------------------------------------------------
    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        if not self.prefer_cli:
            try:
                return self._invoke_sdk(task, prompt, max_turns)
            except ImportError:
                pass  # fall back to CLI
        return self._invoke_cli(task, prompt)

    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        import asyncio

        from claude_agent_sdk import ClaudeAgentOptions, query  # may raise ImportError

        async def _run() -> str:
            chunks: list[str] = []
            env = self._child_env()
            options = ClaudeAgentOptions(
                cwd=str(task.workdir),
                add_dirs=[str(task.workdir)],
                allowed_tools=_ALLOWED_TOOLS,
                permission_mode=self.cli_permission_mode,
                max_turns=max_turns,
                # Append to (rather than replace) the Claude Code system prompt so
                # the agent keeps its tool-use behavior.
                system_prompt={"type": "preset", "preset": "claude_code", "append": _SYSTEM_PROMPT},
                # Forward ANTHROPIC_API_KEY into the spawned CLI so an explicit key
                # is used; omitted entirely in auth mode (subscription/OAuth login).
                **({"env": env} if env else {}),
                **({"model": self.model} if self.model else {}),
            )
            async for message in query(prompt=prompt, options=options):
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(text)
                result = getattr(message, "result", None)
                if result:
                    chunks.append(str(result))
            return "\n".join(chunks)

        return asyncio.run(_run())

    def _invoke_cli(self, task: AgentTask, prompt: str) -> str:
        # Note: the `claude` CLI (>=2.x) has no --max-turns flag; in --print mode
        # it loops internally until the task is done. We grant the file/build
        # tools and append (not replace) the default Claude Code system prompt so
        # it retains its tool-use behavior. The agent works inside an isolated
        # output directory, so bypassing permission prompts is safe and required
        # for non-interactive runs.
        cmd = [
            "claude",
            "-p",
            prompt,
            "--add-dir",
            str(task.workdir),
            "--allowedTools",
            *_ALLOWED_TOOLS,
            "--permission-mode",
            self.cli_permission_mode,
            "--append-system-prompt",
            _SYSTEM_PROMPT,
            "--output-format",
            "text",
        ]
        if self.model:
            cmd += ["--model", self.model]
        # Forward ANTHROPIC_API_KEY into the CLI's environment when configured so
        # an explicit key forces API-key auth; None keeps the inherited
        # environment (subscription/OAuth auth).
        proc = subprocess.run(
            cmd,
            cwd=str(task.workdir),
            capture_output=True,
            text=True,
            env=self._child_env(),
        )
        output = proc.stdout + proc.stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately so run_verify_loop fails fast
        # instead of masking it as a failed attempt.
        if proc.returncode != 0:
            lowered = output.lower()
            if any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
                raise AgentBackendError(
                    "Claude CLI could not authenticate. Set ANTHROPIC_API_KEY (or "
                    "pass api_key=...), or run `claude login`. Claude output:\n"
                    + output[-1000:]
                )
        return output
