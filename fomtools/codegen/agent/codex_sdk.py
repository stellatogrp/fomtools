"""OpenAI Codex coding-agent backend.

A real, nondeterministic :class:`~fomtools.codegen.agent.driver.AgentDriver`
that drives the OpenAI Codex agent through the authoritative verify/retry loop.
It prefers the in-process ``openai_codex`` Python SDK and falls back to the
``codex exec`` CLI in headless mode; after each attempt the shared loop runs the
verify command itself to decide success, never the agent's self-report.

Like every real backend it needs authentication (``OPENAI_API_KEY`` or a
logged-in ``codex`` CLI) and so never runs in CI; the deterministic pipeline is
exercised via :class:`~fomtools.codegen.agent.stub.StubAgentDriver`. When an API
key is configured it is forwarded into the spawned process environment so it
takes precedence over any on-disk ``codex`` login.

The common SDK-backend machinery (retry loop wiring, child environment, CLI
auth-error handling, SDK→CLI fallback) lives in
:class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver`; this module supplies
only the Codex-specific argv and in-process SDK invocation.
"""

from __future__ import annotations

from pathlib import Path

from fomtools.codegen.agent.driver import AgentTask
from fomtools.codegen.agent.sdk_base import _SYSTEM_PROMPT, SdkAgentDriver


class CodexAgentDriver(SdkAgentDriver):
    """Drive code generation with the OpenAI Codex agent.

    A concrete :class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver` that runs
    a real Codex coding agent through the authoritative retry loop. It prefers
    the in-process ``openai_codex`` Python SDK and falls back to the ``codex
    exec`` CLI in headless mode. The Codex SDK/CLI loop internally, so the
    ``max_turns`` budget is accepted for interface parity but unused.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes.
    """

    api_key_env = "OPENAI_API_KEY"
    cli_name = "codex"
    label = "Codex"

    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
        """Assemble the ``codex exec`` argument vector for one headless attempt.

        Builds a ``codex exec`` invocation that runs non-interactively and loops
        internally until done. ``--skip-git-repo-check`` lets it run in the
        non-git output dir; the sandbox flags mirror ``bypass_sandbox``
        (full bypass that auto-approves writes/``make``, or Codex's
        ``workspace-write`` sandbox). ``--cd`` scopes it to the absolute
        ``workdir``, and ``--model`` is added only when a model is configured.
        Codex has no system-prompt flag, so the shared ``_SYSTEM_PROMPT`` framing
        is prepended to the prompt, which is passed positionally (last).

        Parameters
        ----------
        workdir : str
            Absolute working directory passed via ``--cd``. Codex resolves
            ``--cd`` relative to its own cwd, so an absolute path is required to
            avoid double-resolution.
        prompt : str
            The task prompt for this attempt.

        Returns
        -------
        list of str
            The argument vector to pass to :func:`subprocess.run`.
        """
        sandbox_flags = (
            ["--dangerously-bypass-approvals-and-sandbox"]
            if self.bypass_sandbox
            else ["--sandbox", "workspace-write"]
        )
        argv = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            *sandbox_flags,
            "--cd",
            workdir,
        ]
        if self.model:
            argv += ["--model", self.model]
        argv.append(_SYSTEM_PROMPT + "\n\n" + prompt)
        return argv

    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one attempt in-process via the ``openai_codex`` Python SDK.

        Starts a fresh Codex thread scoped to the task's resolved absolute
        workdir and runs the prompt to completion, returning the agent's final
        response. The ``max_turns`` budget is not honored here: like ``codex
        exec``, the SDK loops internally until the turn finishes. Success is
        decided later by the caller's verify command, not here.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to
            write, the restore hook, and the verify command to run.
        prompt : str
            The prompt for this attempt.
        max_turns : int
            Per-attempt turn budget. Accepted for interface parity but unused:
            the Codex turn loops internally until it is done.

        Returns
        -------
        str
            The agent's final response text (empty string if it produced none).
            Does not decide success.

        Raises
        ------
        ImportError
            If ``openai_codex`` is not installed, so the caller can fall back
            to the CLI.
        """
        import asyncio

        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

        workdir = str(Path(task.workdir).resolve())
        # bypass_sandbox mirrors the CLI's --dangerously-bypass-approvals-and-sandbox:
        # full filesystem access so the agent can write the solver and run `make`
        # in its isolated dir. Otherwise keep Codex's workspace-write sandbox.
        sandbox = Sandbox.full_access if self.bypass_sandbox else Sandbox.workspace_write

        async def _run() -> str:
            config = CodexConfig(
                codex_bin=getattr(self, "codex_bin", None),
                cwd=workdir,
                env=self._child_env(),
            )
            async with AsyncCodex(config) as codex:
                thread = await codex.thread_start(
                    cwd=workdir,
                    approval_mode=ApprovalMode.auto_review,
                    sandbox=sandbox,
                    developer_instructions=_SYSTEM_PROMPT,
                    **({"model": self.model} if self.model else {}),
                )
                result = await thread.run(prompt)
            return result.final_response or ""

        return asyncio.run(_run())
