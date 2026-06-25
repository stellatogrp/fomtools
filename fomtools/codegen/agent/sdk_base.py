from __future__ import annotations

import os
import subprocess
from abc import abstractmethod
from pathlib import Path

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
    run_verify_loop,
)

_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# Substrings that mark non-recoverable authentication failure rather than a fixable task error
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


class SdkAgentDriver(AgentDriver):
    """Base for the real, SDK-backed coding-agent drivers.

    A concrete :class:`~fomtools.codegen.agent.driver.AgentDriver` that runs a
    real coding agent through the authoritative retry loop. It prefers an
    in-process Python SDK and falls back to the vendor's CLI in headless mode.
    After each attempt the shared loop runs the verify command itself to decide
    success, never the agent's self-report.

    Warnings
    --------
    The defaults (``cli_permission_mode="bypassPermissions"``,
    ``bypass_sandbox=True``) are intended for a fully automated, isolated
    code-generation run: the agent must write files and run ``make``
    non-interactively. They apply to **both** the in-process SDK path and the
    CLI fallback, so ``prefer_cli=False`` is not a safer mode. Because the agent
    runs with permission prompts bypassed and the vendor sandbox disabled,
    ``AgentTask.workdir`` must be a dedicated output directory rather than a
    subdirectory of a project repository; otherwise a misconfigured
    ``--add-dir`` could expose the surrounding repository to the agent.

    Subclasses set the class attributes :attr:`api_key_env`, :attr:`cli_name`,
    and :attr:`label`, and implement :meth:`_build_argv` and :meth:`_invoke_sdk`.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes.

    Attributes
    ----------
    config
        Codegen configuration object, retained for backend use.
    model : str or None
        Explicit model identifier passed to the SDK/CLI; ``None`` uses the
        backend's default model.
    max_attempts : int
        Maximum number of prompt/verify rounds.
    max_turns : int
        Per-attempt turn budget (honored only by backends that support it).
    prefer_cli : bool
        When ``True``, skip the in-process SDK and shell out to the CLI directly.
    cli_permission_mode : str
        Permission mode for the CLI fallback (used by backends that honor it;
        retained for interface parity otherwise).
    bypass_sandbox : bool
        When ``True``, run the agent with full filesystem access and
        auto-approval; otherwise keep the vendor's sandbox. Ignored by backends
        without a sandbox concept (e.g. Claude).
    api_key : str or None
        Resolved API key, or ``None`` to use the on-disk vendor login.
    cli_timeout: int
        Maximum number of seconds to wait before invoking the fallback mechanism that starts
        the agent using the CLI.
    """

    #: Name of the environment variable holding this backend's API key.
    api_key_env: str
    #: CLI command name, used in the auth-failure hint (e.g. ``"claude login"``).
    cli_name: str
    #: Human-readable backend name, used in auth-failure messages.
    label: str

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
        cli_timeout: int = 60,

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
            Per-attempt turn budget. Defaults to 16. Honored only by backends
            whose SDK exposes a turn cap; others accept it for interface parity.
        prefer_cli : bool, optional
            When ``True``, skip the in-process SDK and use the CLI directly.
            Defaults to ``False``.
        cli_permission_mode : str, optional
            Permission mode for the CLI fallback. Defaults to
            ``"bypassPermissions"`` so the agent can write files and run ``make``
            non-interactively inside the isolated output directory.
        bypass_sandbox : bool, optional
            When ``True`` (default), run the agent with full filesystem access
            and auto-approval so it can write the solver and run ``make``
            non-interactively. Set ``False`` to keep the vendor's sandbox.
            Ignored by backends without a sandbox concept.
        api_key : str, optional
            Explicit API key. When ``None`` it falls back to the
            :attr:`api_key_env` environment variable; when neither is set the
            backend uses the on-disk vendor login.
        cli_timeout: int
            Maximum number of seconds to wait before invoking the fallback mechanism that starts
            the agent using the CLI.
        """
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        self.max_turns = max_turns
        self.prefer_cli = prefer_cli
        self.cli_permission_mode = cli_permission_mode
        self.bypass_sandbox = bypass_sandbox
        self.api_key = api_key or os.environ.get(self.api_key_env)
        self.cli_timeout = cli_timeout

    def _child_env(self) -> dict[str, str] | None:
        """Process environment for the SDK/CLI, forcing API-key auth if we have one.

        Returns ``None`` when no key is configured so the backend keeps its
        default behavior (the on-disk vendor login). Otherwise returns the full
        parent environment with :attr:`api_key_env` injected, so PATH and the
        rest of the environment are preserved while the key forces API-key auth.

        Returns
        -------
        dict of str to str, or None
            The child environment mapping, or ``None`` to inherit the parent
            environment unchanged.
        """
        if not self.api_key:
            return None
        return {**os.environ, self.api_key_env: self.api_key}

    def run(self, task: AgentTask) -> AgentResult:
        """Generate/refine the solver until the verify command passes.

        Delegates to :func:`~fomtools.codegen.agent.driver.run_verify_loop`,
        supplying a closure that invokes the agent for one attempt. The loop runs
        the verify command itself to decide success and re-prompts with the
        failure output until it passes or the attempt budget is spent.

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

    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one agent attempt, preferring the SDK and falling back to the CLI.

        Tries the in-process Python SDK first (unless ``prefer_cli`` is set); on
        :class:`ImportError` it shells out to the vendor CLI instead, scoping it
        to the task's resolved absolute workdir. Success is decided later by the
        caller's verify command, not here.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to write,
            the restore hook, and the verify command to run.
        prompt : str
            The prompt for this attempt.
        max_turns : int
            Turn budget for this attempt (honored by backends that support it).

        Returns
        -------
        str
            The captured agent log text. Does not decide success.
        """
        if not self.prefer_cli:
            try:
                return self._invoke_sdk(task, prompt, max_turns)
            except (ImportError, NotImplementedError):
                pass  # fall back to CLI
        workdir = str(Path(task.workdir).resolve())
        return self._invoke_cli(self._build_argv(workdir, prompt), workdir)

    def _invoke_cli(self, argv: list[str], workdir: str) -> str:
        """Run one attempt by shelling out to the vendor CLI in headless mode.

        Executes ``argv`` in ``workdir`` with the child environment from
        :meth:`_child_env` (the API key injected when configured, otherwise the
        inherited login environment), capturing output. The CLI loops internally
        in headless mode until the task is done. Success is left to the caller's
        verify command; this method only returns the captured text or fails fast
        on an unrecoverable auth error.

        Parameters
        ----------
        argv : list of str
            The full CLI argument vector, as built by :meth:`_build_argv`.
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
            timeout=self.cli_timeout
        )
        output = proc.stdout + proc.stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately so run_verify_loop fails fast
        # instead of masking it as a failed attempt.
        if proc.returncode != 0 and any(
            marker in output.lower() for marker in _AUTH_FAILURE_MARKERS
        ):
            raise AgentBackendError(
                f"{self.label} CLI could not authenticate. Set {self.api_key_env} "
                f"(or pass api_key=...), or run `{self.cli_name} login`. "
                f"{self.label} output:\n" + output[-1000:]
            )
        return output

    @abstractmethod
    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
        """Assemble the vendor CLI argument vector for one headless attempt.

        Parameters
        ----------
        workdir : str
            Absolute working directory exposed to the CLI.
        prompt : str
            The task prompt for this attempt.

        Returns
        -------
        list of str
            The argument vector to pass to :func:`subprocess.run`.
        """

    @abstractmethod
    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one attempt in-process via the vendor's Python SDK.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to write,
            the restore hook, and the verify command to run.
        prompt : str
            The prompt for this attempt.
        max_turns : int
            Per-attempt turn budget (honored by backends that support it).

        Returns
        -------
        str
            The captured agent log text. Does not decide success.

        Raises
        ------
        ImportError
            If the vendor SDK is not installed, so the caller can fall back to
            the CLI.
        """
