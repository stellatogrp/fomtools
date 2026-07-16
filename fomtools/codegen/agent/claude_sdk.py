"""Real coding-agent driver backed by the Claude Agent SDK.

This driver is intentionally only used when explicitly enabled
(``FOMTOOLS_CODEGEN_AGENT=1``): it is nondeterministic, needs authentication
(``ANTHROPIC_API_KEY`` or a logged-in ``claude`` CLI).

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

The common SDK-backend machinery (retry loop wiring, child environment, CLI
auth-error handling, SDK→CLI fallback) lives in
:class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver`; this module supplies
only the Claude-specific argv and in-process SDK invocation.
"""

from __future__ import annotations

from fomtools.codegen.agent.driver import AgentTask
from fomtools.codegen.agent.sdk_base import (
    _ALLOWED_TOOLS,
    _SYSTEM_PROMPT,
    SdkAgentDriver,
)


class ClaudeAgentDriver(SdkAgentDriver):
    """Drive code generation with the Claude Agent SDK (or the ``claude`` CLI).

    A concrete :class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver` that runs
    a real Claude coding agent through the authoritative retry loop. It prefers
    the in-process Python SDK and falls back to the ``claude`` CLI in headless
    mode. Claude has no filesystem sandbox of its own, so ``bypass_sandbox`` is
    accepted for interface parity but unused.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes.
    """

    api_key_env = "ANTHROPIC_API_KEY"
    cli_name = "claude"
    label = "Claude"

    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
        """Assemble the ``claude`` CLI argument vector for one headless attempt.

        Builds a ``claude -p`` invocation that grants the file/build tools in
        ``_ALLOWED_TOOLS``, scopes the agent to ``workdir`` via ``--add-dir``,
        applies the configured permission mode, and appends ``_SYSTEM_PROMPT``
        to the default Claude Code system prompt. The ``--model`` flag is added
        only when a model is configured.

        Parameters
        ----------
        workdir : str
            Absolute working directory exposed to the CLI via ``--add-dir``.
        prompt : str
            The task prompt for this attempt, passed to ``claude -p``.

        Returns
        -------
        list of str
            The argument vector to pass to :func:`subprocess.run`.

        Note
        -----
        The ``claude`` CLI (>=2.x) has no ``--max-turns`` flag; in ``--print``
        mode it loops internally until the task is done. The argument vector
        grants the file/build tools and appends (not replaces) the default
        Claude Code system prompt so the agent retains its tool-use behavior.
        The agent works inside an isolated output directory, so bypassing
        permission prompts is safe and required for non-interactive runs.
        """
        argv = [
            "claude",
            "-p",
            prompt,
            "--add-dir",
            workdir,
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
            argv += ["--model", self.model]
        return argv

    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one attempt in-process via the ``claude_agent_sdk`` Python SDK.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to
            write, the restore hook, and the verify command to run.
        prompt : str
            The prompt for this attempt.
        max_turns : int
            Turn budget passed to ``ClaudeAgentOptions``.

        Returns
        -------
        str
            The concatenated text and result blocks from the agent.

        Raises
        ------
        ImportError
            If ``claude_agent_sdk`` is not installed, so the caller can fall
            back to the CLI.
        NotImplementedError
            If called from a thread with a running event loop, where the
            in-process path cannot work, so the caller can fall back to the CLI.
        """
        import asyncio

        from claude_agent_sdk import ClaudeAgentOptions, query  # may raise ImportError

        # ``asyncio.run`` refuses to nest inside an existing loop, which is the
        # normal state for callers that already drive one (marimo, Jupyter). Treat
        # that as "SDK unavailable here" so the CLI fallback runs, rather than
        # letting the RuntimeError be masked as a failed agent attempt.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise NotImplementedError(
                "claude_agent_sdk cannot run in-process inside a running event loop"
            )

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
