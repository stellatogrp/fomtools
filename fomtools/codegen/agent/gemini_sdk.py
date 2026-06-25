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

The common SDK-backend machinery (retry loop wiring, child environment, CLI
auth-error handling, SDK→CLI fallback) lives in
:class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver`; this module supplies
only the Gemini-specific argv and in-process agentic loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fomtools.codegen.agent.driver import AgentTask
from fomtools.codegen.agent.sdk_base import _SYSTEM_PROMPT, SdkAgentDriver


class GeminiAgentDriver(SdkAgentDriver):
    """Drive code generation with the Google Gemini agent.

    A concrete :class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver` that runs
    a real Gemini coding agent through the authoritative retry loop. It prefers
    an in-process agentic loop built on the ``google-genai`` Python SDK (file/
    build tools exposed via automatic function calling) and falls back to the
    ``gemini`` CLI in headless mode.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes.
    """

    api_key_env = "GEMINI_API_KEY"
    cli_name = "gemini"
    label = "Gemini"

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

    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Run one attempt in-process via the ``google-genai`` Python SDK.

        Unlike the Codex/Claude SDKs, ``google-genai`` is a model client rather
        than a ready-made coding agent, so the agentic loop is built here from its
        *automatic function calling*: the file/build tools are exposed as plain
        Python callables scoped to the task's resolved absolute workdir, and the
        SDK introspects, invokes, and loops them on the model's behalf until it
        stops requesting tools or the ``max_turns`` budget of tool calls is
        exhausted. ``_SYSTEM_PROMPT`` is supplied as the system instruction.
        Success is decided later by the caller's verify command, not here.

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
            # tools are the code that lets the model interact with external systems
            # to perform actions outside the knowledge and scope of the model.
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
