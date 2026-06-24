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
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
    run_verify_loop,
)

_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

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


class CodexAgentDriver(AgentDriver):
    """Drive code generation with the OpenAI Codex agent.

    A concrete :class:`~fomtools.codegen.agent.driver.AgentDriver` that runs a
    real Codex coding agent through the authoritative retry loop. It prefers the
    in-process ``openai_codex`` Python SDK and falls back to the ``codex exec``
    CLI in headless mode. After each attempt the shared loop runs the verify
    command itself to decide success, never the agent's self-report.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes.

    Attributes
    ----------
    config
        Codegen configuration object, retained for backend use.
    model : str or None
        Explicit model identifier passed to the SDK/CLI; ``None`` uses Codex's
        default model.
    max_attempts : int
        Maximum number of prompt/verify rounds.
    max_turns : int
        Per-attempt turn budget (honored only by backends that support it; the
        Codex SDK/CLI loop internally, so it is accepted but unused).
    prefer_cli : bool
        When ``True``, skip the in-process SDK and shell out to ``codex`` directly.
    cli_permission_mode : str
        Permission mode retained for interface parity with the Claude backend.
    bypass_sandbox : bool
        When ``True``, run Codex with full filesystem access and auto-approval;
        otherwise keep its ``workspace-write`` sandbox.
    api_key : str or None
        Resolved OpenAI API key, or ``None`` to use an on-disk ``codex`` login.
    """

    def __init__(
        self,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        max_turns: int = 16,
        prefer_cli: bool = False,
        cli_permission_mode: str = "bypassPermissions",
        bypass_sandbox: bool = True,
        api_key: str | None = None,
    ):
        """Configure the backend, model, retry budget, sandbox, and authentication.

        Parameters
        ----------
        config : optional
            Codegen configuration object, retained for backend use. Defaults to
            ``None``.
        model : str, optional
            Explicit model identifier to pass to the SDK/CLI. When ``None`` the
            backend's default model is used.
        max_attempts : int, optional
            Maximum number of prompt/verify rounds. Defaults to 5.
        max_turns : int, optional
            Per-attempt turn budget. Defaults to 16. Accepted for interface
            parity; the Codex SDK/CLI loop internally and do not honor it.
        prefer_cli : bool, optional
            When ``True``, skip the in-process SDK and use the ``codex`` CLI
            directly. Defaults to ``False``.
        cli_permission_mode : str, optional
            Permission mode retained for parity with the Claude backend.
            Defaults to ``"bypassPermissions"``.
        bypass_sandbox : bool, optional
            When ``True`` (default), run Codex with full filesystem access and
            auto-approval so it can write the solver and run ``make`` non-
            interactively inside the isolated output directory. Set ``False`` to
            keep Codex's ``workspace-write`` sandbox instead.
        api_key : str, optional
            Explicit OpenAI API key. When ``None`` it falls back to the
            ``OPENAI_API_KEY`` environment variable; when neither is set the
            backend uses the on-disk ``codex`` login.
        """
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        self.max_turns = max_turns
        self.prefer_cli = prefer_cli
        self.cli_permission_mode = cli_permission_mode
        self.bypass_sandbox = bypass_sandbox
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _child_env(self) -> dict[str, str] | None:
        """Process environment for the SDK/CLI, forcing API-key auth if we have one.

        Returns ``None`` when no key is configured so the backend keeps its
        default behavior (an on-disk ``codex`` login). Otherwise returns the full
        parent environment with ``OPENAI_API_KEY`` injected, so PATH and the rest
        of the environment are preserved while the key forces API-key auth.

        Returns
        -------
        dict of str to str, or None
            The child environment mapping, or ``None`` to inherit the parent
            environment unchanged.
        """
        if not self.api_key:
            return None
        return {**os.environ, "OPENAI_API_KEY": self.api_key}

    def run(self, task: AgentTask) -> AgentResult:
        """Generate/refine the solver until the verify command passes.

        Delegates to :func:`~fomtools.codegen.agent.driver.run_verify_loop`,
        supplying a closure that invokes the Codex agent for one attempt. The
        loop runs the verify command itself to decide success and re-prompts with
        the failure output until it passes or the attempt budget is spent.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to write,
            the restore hook, and the verify command to run.

        Returns
        -------
        AgentResult
            The outcome, whose ``success`` reflects whether the verify command
            passed.

        Raises
        ------
        AgentBackendError
            If the backend cannot authenticate or its CLI is unavailable;
            propagated from the retry loop rather than retried.
        """
        return run_verify_loop(
            task,
            lambda prompt, max_turns: self._invoke_agent(task, prompt, max_turns),
            max_attempts=self.max_attempts,
            max_turns=self.max_turns,
        )

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

    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one agent attempt, preferring the SDK and falling back to the CLI.

        Tries the in-process ``openai_codex`` SDK first (unless ``prefer_cli`` is
        set); on :class:`ImportError` it shells out to the ``codex`` CLI instead,
        scoping it to the task's resolved absolute workdir. Success is decided
        later by the caller's verify command, not here.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to write,
            the restore hook, and the verify command to run.
        prompt : str
            The prompt for this attempt.
        max_turns : int
            Per-attempt turn budget. Forwarded to the SDK path for interface
            parity; both Codex backends loop internally and do not honor it.

        Returns
        -------
        str
            The captured agent log text. Does not decide success.
        """
        if not self.prefer_cli:
            try:
                return self._invoke_sdk(task, prompt, max_turns)
            except ImportError:
                pass  # fall back to CLI
        workdir = str(Path(task.workdir).resolve())
        return self._invoke_cli(self._build_argv(workdir, prompt), workdir)

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

    def _invoke_cli(self, argv: list[str], workdir: str) -> str:
        """Run one attempt by shelling out to the ``codex`` CLI in headless mode.

        Executes ``argv`` in ``workdir`` with the child environment from
        :meth:`_child_env` (``OPENAI_API_KEY`` injected when configured,
        otherwise the inherited login environment), capturing output. The CLI
        loops internally in ``codex exec`` mode until the task is done. Success
        is left to the caller's verify command; this method only returns the
        captured text or fails fast on an unrecoverable auth error.

        Parameters
        ----------
        argv : list of str
            The full ``codex`` CLI argument vector, as built by
            :meth:`_build_argv`.
        workdir : str
            Absolute working directory to run the CLI in.

        Returns
        -------
        str
            The combined stdout/stderr from the CLI. Does not decide success.

        Raises
        ------
        AgentBackendError
            If the CLI exits non-zero with output matching a known
            authentication-failure marker, since retrying cannot fix it.
        """
        proc = subprocess.run(
            argv,
            cwd=workdir,
            capture_output=True,
            text=True,
            env=self._child_env(),
        )
        output = proc.stdout + proc.stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately so run_verify_loop fails fast
        # instead of masking it as a failed attempt.
        if proc.returncode != 0:
            if any(marker in output.lower() for marker in _AUTH_FAILURE_MARKERS):
                raise AgentBackendError(
                    "Codex CLI could not authenticate. Set OPENAI_API_KEY (or "
                    "pass api_key=...), or run `codex login`. codex  output:\n" + output[-1000:]
                )
        return output
