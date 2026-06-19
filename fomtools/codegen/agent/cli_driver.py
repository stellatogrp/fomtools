"""Generic driver for any headless coding-agent CLI, configured by a spec.

A single :class:`CliAgentDriver` drives every prompt-in/files-out CLI backend
(Codex, Gemini, and any user-registered agent); the per-tool differences live in
:class:`~fomtools.codegen.agent.specs.CliAgentSpec`. This is the agent-agnostic
core: the authoritative verify/retry loop is shared via
:func:`~fomtools.codegen.agent.driver.run_verify_loop`, and authentication is
forced to API-key mode when a key is available (the key is forwarded into the
spawned process environment so it takes precedence over any on-disk login).

These backends are nondeterministic and need auth, so they never run in CI; the
deterministic pipeline is tested via
:class:`~fomtools.codegen.agent.stub.StubAgentDriver`.
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
    """Drive code generation with a headless CLI described by a :class:`CliAgentSpec`."""

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
        self.spec = spec
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        # These CLIs loop internally until done (no per-call turn cap to pass);
        # kept for parity with the shared loop's signature.
        self.max_turns = max_turns
        self.bin = bin or spec.bin
        self.extra_args = tuple(extra_args)
        # Explicit API key wins; else the first of the spec's env vars that is
        # set; else None -> the backend keeps its default (login) auth. A
        # resolved key is forwarded into the spawned process so it takes
        # precedence over any on-disk login (matching each CLI's resolution
        # order).
        self.api_key = api_key or self._key_from_env()

    # -- auth helpers ------------------------------------------------------
    def _key_from_env(self) -> str | None:
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
        # Already authenticated? login_check_args exits 0 when a session exists.
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

    # -- public API --------------------------------------------------------
    def run(self, task: AgentTask) -> AgentResult:
        # Preflight before the retry loop: a missing CLI is fatal and must not be
        # absorbed as a failed attempt (which, against a stale output dir, could
        # otherwise verify and look successful).
        if shutil.which(self.bin) is None:
            raise AgentBackendError(
                f"{self.spec.display_name or self.spec.name} CLI {self.bin!r} not found "
                f"on PATH. {self.spec.install_hint} Or select a different backend "
                "(`--agent claude`)."
            )

        # Persist an API-key login for backends that authenticate from an
        # on-disk session rather than the environment (no-op otherwise).
        self._ensure_login()
        return run_verify_loop(
            task,
            lambda prompt, max_turns: self._invoke_agent(task, prompt, max_turns),
            max_attempts=self.max_attempts,
            max_turns=self.max_turns,
        )

    # -- agent invocation --------------------------------------------------
    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
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

    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        # Resolve to an ABSOLUTE workdir: some CLIs resolve their dir flag
        # relative to their own cwd, so a relative path applied as both --cd and
        # the subprocess cwd would be doubled.
        workdir = str(Path(task.workdir).resolve())
        argv = self._build_argv(workdir, prompt)
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
