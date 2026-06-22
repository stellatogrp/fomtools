"""Generic driver for any headless coding-agent CLI, configured by a spec.

A single :class:`CliAgentDriver` drives every prompt-in/files-out CLI backend
(Codex, Gemini, and any user-registered agent); the per-tool differences live in
:class:`~fomtools.codegen.agent.specs.CliAgentSpec`. This is the agent-agnostic
core: the authoritative verify/retry loop is shared via
:func:`~fomtools.codegen.agent.driver.run_verify_loop`, and authentication is
forced to API-key mode when a key is available (the key is forwarded into the
spawned process environment so it takes precedence over any on-disk login).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
    run_verify_loop,
)
from fomtools.codegen.agent.specs import SYSTEM_PROMPT, CliAgentSpec


class CliAgentDriver(AgentDriver):
    """Drive code generation with a headless CLI described by a :class:`CliAgentSpec`.

    A concrete :class:`~fomtools.codegen.agent.driver.AgentDriver` for every
    prompt-in/files-out CLI backend (Codex, Gemini, and user-registered agents).
    All per-tool differences are carried by the ``spec``; this class supplies the
    agent-agnostic core and shares the authoritative verify/retry loop via
    :func:`~fomtools.codegen.agent.driver.run_verify_loop`.

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
        max_turns: int = 16,
        bin: str | None = None,
        api_key: str | None = None,
        extra_args: Sequence[str] = (),
    ):
        """Configure the driver from a CLI spec, model, retry budget, and auth.

        Parameters
        ----------
        spec : CliAgentSpec
            The per-tool specification describing the CLI's binary, flags,
            prompt/model/cwd flags, login flow, and auth markers.
        config : optional
            Codegen configuration object, retained for backend use. Defaults to
            ``None``.
        model : str, optional
            Explicit model identifier, passed via the spec's ``model_flag`` when
            both are set. When ``None`` the CLI's default model is used.
        max_attempts : int, optional
            Maximum number of prompt/verify rounds. Defaults to 5.
        max_turns : int, optional
            Per-attempt turn budget, kept for parity with the shared loop; these
            CLIs loop internally so it is not passed through. Defaults to 16.
        bin : str, optional
            Override for the CLI binary name. When ``None`` the spec's ``bin``
            is used.
        api_key : str, optional
            Explicit API key. When ``None`` it falls back to the first set
            variable among the spec's key env vars; when neither is found the
            backend keeps its default (login) auth.
        extra_args : Sequence of str, optional
            Additional arguments appended to the CLI invocation. Defaults to an
            empty tuple.
        """
        self.spec = spec
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        self.max_turns = max_turns
        self.bin = bin or spec.bin
        self.extra_args = tuple(extra_args)
        self.api_key = api_key or self._key_from_env()

    def _key_from_env(self) -> str | None:
        """Return the first API key found among the spec's key env vars.

        Returns
        -------
        str or None
            The value of the first set variable in ``spec.key_env_vars``, or
            ``None`` when none are set.
        """
        for var in self.spec.key_env_vars:
            value = os.environ.get(var)
            if value:
                return value
        return None

    def _child_env(self) -> dict[str, str] | None:
        """Process env for the CLI, forcing API-key auth if we have a key.

        ``None`` when no key is configured (keep default login auth); otherwise
        the full parent environment with the spec's primary key var injected, so
        PATH and the rest of the environment are preserved.
        """
        if not (self.api_key and self.spec.api_key_env):
            return None
        return {**os.environ, self.spec.api_key_env: self.api_key}

    def _ensure_login(self) -> None:
        """Persist an API-key login for CLIs that need one (spec-driven).

        Some CLIs (e.g. Codex) authenticate from an on-disk session written by
        their own ``login`` subcommand and do not reliably pick the key up from
        the environment for every transport. When the spec describes an API-key
        login flow and we have a key, log in once: skip if ``login_check_args``
        reports an existing session, otherwise pipe the key to ``login_args`` on
        stdin. A no-op for backends with no login flow or no resolved key.
        """
        if not (self.api_key and self.spec.login_args):
            return
        # login_check_args exits 0 when an already authenticated session exists.
        if self.spec.login_check_args:
            check = subprocess.run(
                [self.bin, *self.spec.login_check_args],
                capture_output=True,
                text=True,
            )
            if check.returncode == 0:
                return
        proc = subprocess.run(
            [self.bin, *self.spec.login_args],
            input=self.api_key,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            output = proc.stdout + proc.stderr
            raise AgentBackendError(
                f"{self.spec.display_name or self.spec.name} CLI login with the "
                f"resolved API key failed. {self.spec.auth_hint}\n"
                f"{self.spec.display_name or self.spec.name} output:\n" + output[-1000:]
            )

    def run(self, task: AgentTask) -> AgentResult:
        """Drive the CLI until the task's verify command passes (or attempts run out).

        Preflights that the CLI binary is on PATH and, when the spec requires
        it, persists an API-key login as a side effect. It then resolves the
        task workdir to an absolute path and hands control to
        :func:`~fomtools.codegen.agent.driver.run_verify_loop`, which repeatedly
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
            If the CLI binary is not on PATH, or an API-key login is required
            but fails. Neither condition is recoverable by retrying.
        """
        if shutil.which(self.bin) is None:
            raise AgentBackendError(
                f"{self.spec.display_name or self.spec.name} CLI {self.bin!r} not found "
                f"on PATH. {self.spec.install_hint} Or select a different backend "
                "(`--agent claude`)."
            )

        self._ensure_login()
        # Resolve to an ABSOLUTE workdir: some CLIs resolve their dir flag
        # relative to their own cwd, so a relative path applied as both --cd and
        # the subprocess cwd would be doubled.
        workdir = str(Path(task.workdir).resolve())
        return run_verify_loop(
            task,
            lambda prompt, max_turns: self._invoke_agent(
                self._build_argv(workdir, prompt), workdir
            ),
            max_attempts=self.max_attempts,
            max_turns=self.max_turns,
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
            The argument vector to pass to :func:`subprocess.run`.
        """
        spec = self.spec
        argv = [self.bin, *spec.base_args, *spec.flags, *self.extra_args]
        if spec.cwd_flag:
            argv += [spec.cwd_flag, workdir]
        if self.model and spec.model_flag:
            argv += [spec.model_flag, self.model]
        full_prompt = prompt
        if spec.sysprompt_flag:
            argv += [spec.sysprompt_flag, SYSTEM_PROMPT]
        else:
            # No system-prompt flag: prepend the shared framing to the task prompt.
            full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
        argv += [spec.prompt_arg, full_prompt] if spec.prompt_arg else [full_prompt]
        return argv

    def _invoke_agent(self, argv: list[str], workdir: str | Path) -> str:
        """Run one agent attempt by shelling out to the CLI.

        Executes ``argv`` in ``workdir`` with the child environment from
        :meth:`_child_env` (API-key auth injected when configured), capturing
        output. Success is left to the caller's verify command; this method only
        returns the captured text or fails fast on an unrecoverable auth error.

        Parameters
        ----------
        argv : list[str]
            The full CLI argument vector.
        workdir : str or Path
            Absolute working directory to run the CLI in.

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

        try:
            proc = subprocess.run(
                argv,
                cwd=workdir,
                capture_output=True,
                text=True,
                env=self._child_env(),
            )
        except FileNotFoundError as exc:  # binary vanished after preflight
            raise AgentBackendError(
                f"{self.spec.display_name or self.spec.name} CLI {self.bin!r} not found: {exc}"
            ) from exc
        output = proc.stdout + proc.stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately so run_verify_loop fails fast
        # instead of masking it as a failed attempt.
        if proc.returncode != 0:
            lowered = output.lower()
            if any(marker in lowered for marker in self.spec.all_auth_markers):
                raise AgentBackendError(
                    f"{self.spec.display_name or self.spec.name} CLI could not "
                    f"authenticate. {self.spec.auth_hint}\n"
                    f"{self.spec.display_name or self.spec.name} output:\n" + output[-1000:]
                )
        return output
