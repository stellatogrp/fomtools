"""Google Gemini coding-agent backend.

A real, nondeterministic :class:`~fomtools.codegen.agent.driver.AgentDriver` that
drives the Google Gemini agent through the authoritative verify/retry loop. It
prefers an in-process agentic loop built on the ``google-genai`` Python SDK and
falls back to the ``gemini`` CLI in headless mode; after each attempt the shared
loop runs the verify command itself to decide success, never the agent's
self-report.

Like every real backend it needs authentication (``GEMINI_API_KEY``/
``GOOGLE_API_KEY`` or a logged-in ``gemini`` CLI) and so never runs in CI; the
deterministic pipeline is exercised via
:class:`~fomtools.codegen.agent.stub.StubAgentDriver`. When an API key is
configured it is forwarded into the spawned process environment so it takes
precedence over any on-disk ``gemini`` login.
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


class GeminiAgentDriver(AgentDriver):
    """Drive code generation with the Google Gemini agent.

    A concrete :class:`~fomtools.codegen.agent.driver.AgentDriver` that runs a
    real Gemini coding agent through the authoritative retry loop. It prefers an
    in-process agentic loop built on the ``google-genai`` Python SDK (file/build
    tools exposed via automatic function calling) and falls back to the
    ``gemini`` CLI in headless mode. After each attempt the shared loop runs the
    verify command itself to decide success, never the agent's self-report.

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
        Per-attempt turn/tool-call budget (honored by the SDK path; the CLI
        loops internally in headless mode).
    prefer_cli : bool
        When ``True``, skip the in-process SDK and shell out to ``gemini``
        directly.
    cli_permission_mode : str
        Permission mode retained for interface parity with the Claude backend.
    bypass_sandbox : bool
        When ``True``, run Gemini with full filesystem access (no ``--sandbox``);
        otherwise confine it to Gemini's ``--sandbox``.
    api_key : str or None
        Resolved Gemini API key, or ``None`` to use an on-disk ``gemini`` login.
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
            Per-attempt turn budget. Defaults to 16. Honored by the SDK path as
            the maximum number of automatic tool calls; the ``gemini`` CLI loops
            internally in headless mode and does not honor it.
        prefer_cli : bool, optional
            When ``True``, skip the in-process SDK and use the ``gemini`` CLI
            directly. Defaults to ``False``.
        cli_permission_mode : str, optional
            Permission mode retained for parity with the Claude backend.
            Defaults to ``"bypassPermissions"``.
        bypass_sandbox : bool, optional
            When ``True`` (default), run Gemini with full filesystem access (no
            ``--sandbox``) so it can write the solver and run ``make`` non-
            interactively inside the isolated output directory. Set ``False`` to
            confine it to Gemini's ``--sandbox`` instead.
        api_key : str, optional
            Explicit Gemini API key. When ``None`` it falls back to the
            ``GEMINI_API_KEY`` environment variable; when neither is set the
            backend uses the on-disk ``gemini`` login.
        """
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        self.max_turns = max_turns
        self.prefer_cli = prefer_cli
        self.cli_permission_mode = cli_permission_mode
        self.bypass_sandbox = bypass_sandbox
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def _child_env(self) -> dict[str, str] | None:
        """Process environment for the SDK/CLI, forcing API-key auth if we have one.

        Returns ``None`` when no key is configured so the backend keeps its
        default behavior (an on-disk ``gemini`` login). Otherwise returns the full
        parent environment with ``GEMINI_API_KEY`` injected, so PATH and the rest
        of the environment are preserved while the key forces API-key auth.

        Returns
        -------
        dict of str to str, or None
            The child environment mapping, or ``None`` to inherit the parent
            environment unchanged.
        """
        if not self.api_key:
            return None
        return {**os.environ, "GEMINI_API_KEY": self.api_key}

    def run(self, task: AgentTask) -> AgentResult:
        """Generate/refine the solver until the verify command passes.

        Delegates to :func:`~fomtools.codegen.agent.driver.run_verify_loop`,
        supplying a closure that invokes the Gemini agent for one attempt. The
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

    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one agent attempt, preferring the SDK and falling back to the CLI.

        Tries the in-process ``google-genai`` SDK first (unless ``prefer_cli`` is
        set); on :class:`ImportError` it shells out to the ``gemini`` CLI instead,
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
            Per-attempt turn budget. Honored by the SDK path as the maximum
            number of automatic tool calls; the CLI loops internally and does not
            honor it.

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

    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
        """Assemble the ``gemini`` argument vector for one headless attempt.

        Builds a non-interactive ``gemini`` invocation. ``--prompt`` carries the
        task and triggers headless mode; ``--yolo`` auto-approves writes and
        ``make`` so the agent runs unattended in its isolated output dir. The
        sandbox flag mirrors ``bypass_sandbox`` (full filesystem access when
        bypassing, otherwise Gemini's ``--sandbox``), and ``--model`` is added
        only when a model is configured. Gemini has no working-directory flag, so
        ``workdir`` is scoped via the subprocess cwd rather than an argv flag, and
        no system-prompt flag, so the shared ``_SYSTEM_PROMPT`` framing is
        prepended to the prompt.

        Parameters
        ----------
        workdir : str
            Absolute working directory. Accepted for interface parity with the
            other backends; Gemini takes no cwd flag, so it is applied via the
            subprocess cwd in :meth:`_invoke_cli` instead of the argv.
        prompt : str
            The task prompt for this attempt.

        Returns
        -------
        list of str
            The argument vector to pass to :func:`subprocess.run`.
        """
        sandbox_flags = [] if self.bypass_sandbox else ["--sandbox"]
        argv = [
            "gemini",
            "--yolo",
            *sandbox_flags,
        ]
        if self.model:
            argv += ["--model", self.model]
        argv += ["--prompt", _SYSTEM_PROMPT + "\n\n" + prompt]
        return argv

    def _invoke_cli(self, argv: list[str], workdir: str) -> str:
        """Run one attempt by shelling out to the ``gemini`` CLI in headless mode.

        Executes ``argv`` in ``workdir`` with the child environment from
        :meth:`_child_env` (``GEMINI_API_KEY`` injected when configured,
        otherwise the inherited login environment), capturing output. The CLI
        loops internally in headless ``--prompt`` mode until the task is done.
        Success is left to the caller's verify command; this method only returns
        the captured text or fails fast on an unrecoverable auth error.

        Parameters
        ----------
        argv : list of str
            The full ``gemini`` CLI argument vector, as built by
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
                    "Gemini CLI could not authenticate. Set GEMINI_API_KEY (or "
                    "pass api_key=...), or run `gemini login`. gemini  output:\n" + output[-1000:]
                )
        return output

    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one attempt in-process via the ``google-genai`` Python SDK.

        Unlike the Codex/Claude SDKs, ``google-genai`` is a model client rather
        than a ready-made coding agent, so the agentic loop is built here from its
        *automatic function calling*: the file/build tools in ``_ALLOWED_TOOLS``
        are exposed as plain Python callables scoped to the task's resolved
        absolute workdir, and the SDK introspects, invokes, and loops them on the
        model's behalf until it stops requesting tools or the ``max_turns`` budget
        of tool calls is exhausted. ``_SYSTEM_PROMPT`` is supplied as the system
        instruction. Success is decided later by the caller's verify command, not
        here.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to write,
            the restore hook, and the verify command to run.
        prompt : str
            The prompt for this attempt.
        max_turns : int
            Per-attempt budget, applied as the maximum number of automatic tool
            calls the SDK will make before returning.

        Returns
        -------
        str
            The model's final response text (empty string if it produced none).
            Does not decide success.

        Raises
        ------
        ImportError
            If ``google-genai`` is not installed, so the caller can fall back to
            the CLI.
        """
        from google import genai  # may raise ImportError
        from google.genai import types

        workdir = Path(task.workdir).resolve()

        def _resolve(path: str) -> Path:
            return (workdir / path).resolve()

        def read_file(path: str) -> str:
            """Read and return the UTF-8 text of a file in the working directory.

            Parameters
            ----------
            path : str
                File path relative to the working directory.
            """
            return _resolve(path).read_text()

        def write_file(path: str, content: str) -> str:
            """Create or overwrite a file with the given UTF-8 text.

            Parameters
            ----------
            path : str
                File path relative to the working directory.
            content : str
                Full new contents of the file.
            """
            target = _resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"wrote {len(content)} bytes to {path}"

        def run_bash(command: str) -> str:
            """Run a shell command in the working directory; return its output.

            Parameters
            ----------
            command : str
                The shell command to execute (e.g. ``make verify``).
            """
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                env=self._child_env(),
            )
            return f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}"

        client = genai.Client(**({"api_key": self.api_key} if self.api_key else {}))
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            # tools are the Ccde that enables the system to interact with external systems to
            # perform an action outside of the knowledge and scope of the model
            tools=[read_file, write_file, run_bash],
            # Build the agentic loop from automatic function calling, capping the
            # tool-call budget at max_turns so a stuck attempt cannot loop forever.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=max_turns,
            ),
        )
        response = client.models.generate_content(
            model=self.model or "gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        return response.text or ""
