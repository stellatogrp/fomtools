"""Real coding-agent driver backed by the Google Gemini CLI.

Like the Claude and Codex drivers, this backend is only used when explicitly
selected (``--agent gemini`` or ``FOMTOOLS_CODEGEN_BACKEND=gemini``): it is
nondeterministic, needs authentication (``GEMINI_API_KEY`` or a logged-in
``gemini`` CLI), and therefore never runs in CI. The deterministic pipeline is
tested via :class:`~fomtools.codegen.agent.stub.StubAgentDriver`.

It shells out to ``gemini -p`` — the non-interactive (headless) mode that runs
the agent to completion in one invocation — inside the isolated output
directory, with ``--yolo`` so file writes and ``make`` run without approval
prompts. The authoritative verify/retry loop is shared with every other real
backend via :func:`~fomtools.codegen.agent.driver.run_verify_loop`, which
restores the deterministic files (``restore_hook``) and runs the verify command
itself after each attempt, re-prompting with the failure output until it passes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from fomtools.codegen.agent.driver import (
    AgentBackendError,
    AgentDriver,
    AgentResult,
    AgentTask,
    run_verify_loop,
)

# Substrings (matched case-insensitively in gemini's own error output) that mark
# a non-recoverable authentication failure rather than a fixable task error.
_AUTH_FAILURE_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "missing api key",
    "gemini_api_key",
    "google_api_key",
    "please set an auth method",
    "unauthorized",
    "permission denied",
    "401",
    "403",
    "authentication",
)

# Gemini has no per-call "system prompt" flag (its persistent guidance comes from
# GEMINI.md context files). We prepend the same guidance the other backends use
# to the task prompt so every backend gets equivalent framing.
_SYSTEM_PROMPT = (
    "You are an expert in high-performance numerical C. You translate a "
    "specified first-order optimization method into a correct, allocation-free, "
    "division-free embedded C solver and iterate until it passes verification."
)


class GeminiAgentDriver(AgentDriver):
    """Drive code generation with the Google ``gemini`` CLI (``gemini -p``)."""

    def __init__(
        self,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        max_turns: int = 16,
        gemini_bin: str = "gemini",
        yolo: bool = True,
    ):
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        # `gemini -p` runs the agent to completion in one invocation (no turn cap
        # to pass); kept for parity with the shared loop's signature.
        self.max_turns = max_turns
        self.gemini_bin = gemini_bin
        # The agent works inside an isolated output directory, so auto-approving
        # actions is safe and required for non-interactive runs (it must write the
        # solver and run `make`). Set False to keep the default approval mode.
        self.yolo = yolo

    # -- public API --------------------------------------------------------
    def run(self, task: AgentTask) -> AgentResult:
        # Preflight before the retry loop: a missing CLI is fatal and must not be
        # absorbed as a failed attempt (which, against a stale output dir, could
        # otherwise verify and look successful).
        if shutil.which(self.gemini_bin) is None:
            raise AgentBackendError(
                f"Gemini CLI {self.gemini_bin!r} not found on PATH. Install it "
                "(e.g. `npm install -g @google/gemini-cli`), make sure it is on "
                "PATH, or select a different backend (`--agent claude`)."
            )
        return run_verify_loop(
            task,
            lambda prompt, max_turns: self._invoke_agent(task, prompt, max_turns),
            max_attempts=self.max_attempts,
            max_turns=self.max_turns,
        )

    # -- agent invocation --------------------------------------------------
    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        # The Gemini CLI operates on its current working directory, so we start
        # the process there. Resolve to an ABSOLUTE path: a relative workdir would
        # be wrong if the process's cwd ever differs, and absolute paths avoid the
        # path-resolution pitfalls seen with other CLIs.
        workdir = str(Path(task.workdir).resolve())
        full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
        cmd = [self.gemini_bin]
        if self.yolo:
            cmd.append("--yolo")
        if self.model:
            cmd += ["--model", self.model]
        # `-p/--prompt` triggers non-interactive (headless) mode.
        cmd += ["--prompt", full_prompt]
        try:
            proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        except FileNotFoundError as exc:  # binary vanished after preflight
            raise AgentBackendError(f"Gemini CLI {self.gemini_bin!r} not found: {exc}") from exc
        output = proc.stdout + proc.stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately.
        if proc.returncode != 0:
            lowered = output.lower()
            if any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
                raise AgentBackendError(
                    "Gemini CLI could not authenticate. Set GEMINI_API_KEY or run "
                    "`gemini` once to sign in. Gemini output:\n" + output[-1000:]
                )
        return output
