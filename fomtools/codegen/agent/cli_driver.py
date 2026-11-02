"""Generic driver for any headless coding-agent CLI, configured by a spec.

A single `CliAgentDriver` drives every coding-agent backend: the ``claude``,
``codex``, ``pi``, and ``cursor`` specs bundled in ``agents.yaml`` and any
user-registered agent. The per-tool differences live in `CliAgentSpec`. Each
agent runs through its own installed CLI, so it uses the user's login, config,
and default model unless ``model`` is set. fomtools never handles API keys or
logins: each CLI must already be logged in on its own (it also inherits the
environment, so a key the CLI itself reads still works). The authoritative
verify/retry loop is shared via `run_verify_loop`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import warnings
from collections.abc import Sequence
from pathlib import Path

from fomtools.codegen.agent.cli_specs import SYSTEM_PROMPT, CliAgentSpec
from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
    _decode_stream,
    run_verify_loop,
)


class CliAgentDriver(AgentDriver):
    """Drive code generation with a headless CLI described by a `CliAgentSpec`.

    A concrete `AgentDriver` for every
    prompt-in/files-out CLI backend (the bundled specs and user-registered
    agents).
    All per-tool differences are carried by the ``spec``; this class supplies the
    agent-agnostic core and shares the authoritative verify/retry loop via
    `run_verify_loop`.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes.
    """

    def __init__(
        self,
        spec: CliAgentSpec,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        bin: str | None = None,
        extra_args: Sequence[str] = (),
    ):
        """Configure the driver from a CLI spec, model, and retry budget.

        Parameters
        ----------
        spec : CliAgentSpec
            The per-tool specification describing the CLI's binary, flags,
            prompt/model/cwd flags, and auth markers.
        config : optional
            Codegen configuration object, retained for backend use. Defaults to
            ``None``.
        model : str, optional
            Model identifier, passed via the spec's ``model_flag``. When ``None``
            it is ``config.agent_model``, else ``FOMTOOLS_CODEGEN_MODEL``, else
            the CLI's default model, resolved at each invocation (so a config
            given to the driver later is honored).
        max_attempts : int, optional
            Maximum number of prompt/verify rounds. Defaults to 5.
        bin : str, optional
            Override for the CLI binary name. When ``None`` the spec's ``bin``
            is used.
        extra_args : Sequence of str, optional
            Additional arguments appended to the CLI invocation. Defaults to an
            empty tuple.
        """
        self.spec = spec
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        self.bin = bin or spec.bin
        self.extra_args = tuple(extra_args)

    def _child_env(self, deadline: int | None = None) -> dict[str, str] | None:
        """Process env for the CLI: the parent's plus ``FOMTOOLS_DEADLINE`` (Unix time).

        ``None`` without a deadline: the CLI inherits the environment as is.
        """
        if deadline is None:
            return None
        return {**os.environ, "FOMTOOLS_DEADLINE": str(deadline)}

    def run(self, task: AgentTask) -> AgentResult:
        """Drive the CLI until the task's verify command passes (or attempts run out).

        Preflights that the CLI binary is on PATH. It then resolves the
        task workdir to an absolute path and hands control to
        `run_verify_loop`, which repeatedly
        invokes the CLI and runs the verify command to decide success.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to write,
            the restore hook, and the verify command to run.

        Returns
        -------
        AgentResult
            The outcome of the verify/retry loop, whose ``success`` reflects
            whether the verify command passed.

        Raises
        ------
        AgentBackendError
            If the CLI binary is not on PATH; retrying cannot fix that.
        """
        if shutil.which(self.bin) is None:
            raise AgentBackendError(
                f"{self.spec.display_name or self.spec.name} CLI {self.bin!r} not found "
                f"on PATH. {self.spec.install_hint} Or select a different backend "
                "(`--agent claude`)."
            )

        # Resolve to an ABSOLUTE workdir: some CLIs resolve their dir flag
        # relative to their own cwd, so a relative path applied as both --cd and
        # the subprocess cwd would be doubled.
        workdir = str(Path(task.workdir).resolve())
        self._truncated = False

        def attempt(prompt: str) -> str:
            # Each attempt gets what is left of the phase budget as a hard timeout,
            # and the agent is told to stop GRACE_S before it (FOMTOOLS_DEADLINE).
            budget = task.extra.get("budget")
            timeout = max(1, int(budget.remaining())) if budget is not None else None
            deadline = budget.agent_deadline() if budget is not None else None
            return self._invoke_agent(
                self._build_argv(workdir, prompt), workdir, timeout=timeout, deadline=deadline
            )

        return run_verify_loop(
            task,
            attempt,
            max_attempts=self.max_attempts,
            truncated_probe=lambda: self._truncated,
            budget_s=task.extra.get("timeout_s") or self._timeout_for(task.phase),
        )

    def _timeout_for(self, phase: str) -> int | None:
        """Wall-clock budget for the phase, from the codegen config.

        Without a config the `CodegenConfig` defaults apply, so a driver run
        directly is never unbounded.
        """
        config = getattr(self, "config", None)
        if config is None:
            from fomtools.codegen.config import CodegenConfig

            config = CodegenConfig()
        key = "agent_optimize_timeout_s" if phase == "code_optimization" else "agent_timeout_s"
        value = getattr(config, key, None)
        return int(value) if value else None

    def _model(self) -> str | None:
        """The model to request: explicit, else the config's, else the env's, else none."""
        return (
            self.model
            or getattr(self.config, "agent_model", None)
            or os.environ.get("FOMTOOLS_CODEGEN_MODEL")
            or None
        )

    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
        """Assemble the full CLI argument vector for one invocation.

        Combines the binary, the spec's base args/flags, and ``extra_args``,
        then appends the cwd, model, and system-prompt flags when the spec
        defines them. If the spec has no system-prompt flag, the shared
        ``SYSTEM_PROMPT`` is prepended to ``prompt`` instead.

        Parameters
        ----------
        workdir : str
            Absolute working directory passed via the spec's ``cwd_flag``.
        prompt : str
            The task prompt for this attempt.

        Returns
        -------
        list of str
            The argument vector to pass to `subprocess.run`.
        """
        spec = self.spec
        argv = [self.bin, *spec.base_args, *spec.flags, *self.extra_args]
        if spec.cwd_flag:
            argv += [spec.cwd_flag, workdir]
        model = self._model()
        if model and spec.model_flag:
            argv += [spec.model_flag, model]
        elif model:
            warnings.warn(
                f"the {spec.name!r} agent spec has no model_flag, so model {model!r} is ignored",
                stacklevel=2,
            )
        full_prompt = prompt
        if spec.sysprompt_flag:
            argv += [spec.sysprompt_flag, SYSTEM_PROMPT]
        else:
            # No system-prompt flag: prepend the shared framing to the task prompt.
            full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
        argv += [spec.prompt_arg, full_prompt] if spec.prompt_arg else [full_prompt]
        return argv

    def _text(self, stdout: str) -> str:
        """The agent's text from its stdout, per the spec's ``output_format``."""
        if self.spec.output_format == "text":
            return stdout
        if self.spec.output_format == "stream-json":
            return stream_json_text(stdout)
        raise ValueError(
            f"agent {self.spec.name!r}: unknown output_format {self.spec.output_format!r}; "
            "use 'text' or 'stream-json'"
        )

    def _invoke_agent(
        self,
        argv: list[str],
        workdir: str | Path,
        timeout: int | None = None,
        deadline: int | None = None,
    ) -> str:
        """Run one agent attempt by shelling out to the CLI.

        Executes ``argv`` in ``workdir`` with the child environment from
        `_child_env` (the agent's deadline), capturing
        output. Success is left to the caller's verify command; this method only
        returns the captured text or fails fast on an unrecoverable auth error.

        Parameters
        ----------
        argv : list[str]
            The full CLI argument vector.
        workdir : str or Path
            Absolute working directory to run the CLI in.
        timeout : int, optional
            Seconds after which the process is ended and its partial output kept.
        deadline : int, optional
            Unix time passed to the agent as ``FOMTOOLS_DEADLINE``.

        Returns
        -------
        str
            The combined stdout/stderr from the CLI. Does not decide success.

        Raises
        ------
        AgentBackendError
            If the CLI binary cannot be found at invocation time, or it exits
            non-zero with output matching one of the spec's auth-failure markers.
        """

        # The child inherits this thread's signal mask, and some imports block
        # SIGINT and SIGTERM (`import cvxpygen` does). Spawn with an empty mask
        # so the agent can still be interrupted and asked to stop.
        mask = signal.pthread_sigmask(signal.SIG_SETMASK, ())
        try:
            # Its own session, so a timeout or an interrupt ends the agent together
            # with what it started (a `make bench`, the solver binary).
            proc = subprocess.Popen(
                argv,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._child_env(deadline),
                start_new_session=True,
            )
        except FileNotFoundError as exc:  # binary vanished after preflight
            raise AgentBackendError(
                f"{self.spec.display_name or self.spec.name} CLI {self.bin!r} not found: {exc}"
            ) from exc
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:  # a detached child still holds the pipes
                stdout, stderr = "", ""
            # Keep the partial output and say so. Propagating would discard it and
            # leak the argv (hence the prompt) into the log via str(exc).
            self._truncated = True
            name = self.spec.display_name or self.spec.name
            return (
                self._text(_decode_stream(stdout))
                + _decode_stream(stderr)
                + (
                    f"\n[fomtools: the {name} CLI exceeded its {timeout}s budget for this attempt "
                    "and was terminated; any output above is partial.]\n"
                )
            )
        except BaseException:  # e.g. KeyboardInterrupt: do not leave the agent running
            _kill_group(proc)
            raise
        output = self._text(stdout) + stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately so run_verify_loop fails fast
        # instead of masking it as a failed attempt.
        if proc.returncode != 0:
            # Only the end of the output: an auth failure is the CLI's last word,
            # while the transcript before it can mention anything.
            lowered = output[-2000:].lower()
            if any(marker in lowered for marker in self.spec.all_auth_markers):
                raise AgentBackendError(
                    f"{self.spec.display_name or self.spec.name} CLI could not "
                    f"authenticate. {self.spec.auth_hint}\n"
                    f"{self.spec.display_name or self.spec.name} output:\n" + output[-1000:]
                )
        return output


def stream_json_text(stdout: str) -> str:
    """The text blocks of Claude Code's ``stream-json`` events, one per line.

    Assistant text is kept in order. The final ``result`` is added only when it
    reports an error (it repeats the last text otherwise). Lines that are not
    JSON, such as a CLI error printed before the stream starts, are kept as is,
    and so is a last line cut off by a timeout.
    """
    chunks: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            if line.strip():
                chunks.append(line)
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if event.get("type") == "assistant" and isinstance(message, dict):
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    chunks.append(block["text"])
        elif event.get("type") == "result" and event.get("is_error"):
            chunks.append(str(event.get("result") or event.get("subtype") or "error"))
    return "\n".join(chunks) + ("\n" if chunks else "")


def _kill_group(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """End ``proc`` and every process in its session (it was started in its own).

    SIGTERM first so the CLI can stop what it started, then SIGKILL after
    ``grace_s`` for whatever is left in the group.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if sig == signal.SIGTERM:
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
