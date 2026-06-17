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
"""

from __future__ import annotations

import subprocess

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask, run_verify_loop

_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

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
            options = ClaudeAgentOptions(
                cwd=str(task.workdir),
                add_dirs=[str(task.workdir)],
                allowed_tools=_ALLOWED_TOOLS,
                permission_mode=self.cli_permission_mode,
                max_turns=max_turns,
                # Append to (rather than replace) the Claude Code system prompt so
                # the agent keeps its tool-use behavior.
                system_prompt={"type": "preset", "preset": "claude_code", "append": _SYSTEM_PROMPT},
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
        proc = subprocess.run(cmd, cwd=str(task.workdir), capture_output=True, text=True)
        return proc.stdout + proc.stderr
