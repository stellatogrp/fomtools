"""Real coding-agent driver backed by the OpenAI Codex CLI.

Like the Claude driver, this backend is only used when explicitly selected
(``--agent codex`` or ``FOMTOOLS_CODEGEN_BACKEND=codex``): it is
nondeterministic, needs authentication (``OPENAI_API_KEY`` or a logged-in
``codex`` CLI), and therefore never runs in CI. The deterministic pipeline is
tested via :class:`~fomtools.codegen.agent.stub.StubAgentDriver`.

It shells out to ``codex exec`` — the non-interactive subcommand that loops
internally until the task is done — inside the isolated output directory. The
authoritative verify/retry loop is shared with every other real backend via
:func:`~fomtools.codegen.agent.driver.run_verify_loop`, which restores the
deterministic files (``restore_hook``) and runs the verify command itself after
each attempt, re-prompting with the failure output until it passes.
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

# Substrings (matched case-insensitively in codex's own error output) that mark
# a non-recoverable authentication failure rather than a fixable task error.
_AUTH_FAILURE_MARKERS = (
    "not logged in",
    "codex login",
    "please log in",
    "unauthorized",
    "401",
    "invalid api key",
    "incorrect api key",
    "missing api key",
    "no api key",
    "openai_api_key",
    "authentication",
)

# Codex has no per-call "system prompt" flag (Claude's --append-system-prompt);
# its persistent guidance comes from AGENTS.md/config. We prepend the same
# guidance to the task prompt so the two backends get equivalent framing.
_SYSTEM_PROMPT = (
    "You are an expert in high-performance numerical C. You translate a "
    "specified first-order optimization method into a correct, allocation-free, "
    "division-free embedded C solver and iterate until it passes verification."
)


class CodexAgentDriver(AgentDriver):
    """Drive code generation with the OpenAI ``codex`` CLI (``codex exec``)."""

    def __init__(
        self,
        config=None,
        model: str | None = None,
        max_attempts: int = 5,
        max_turns: int = 16,
        codex_bin: str = "codex",
        bypass_sandbox: bool = True,
    ):
        self.config = config
        self.model = model
        self.max_attempts = max_attempts
        # codex exec has no turn cap (it loops internally like the claude CLI);
        # kept for parity with the shared loop's signature.
        self.max_turns = max_turns
        self.codex_bin = codex_bin
        # The agent works inside an isolated output directory, so bypassing the
        # sandbox/approval prompts is safe and required for non-interactive runs
        # (it must write the solver and run `make`). Set False to keep Codex's
        # workspace-write sandbox instead.
        self.bypass_sandbox = bypass_sandbox

    # -- public API --------------------------------------------------------
    def run(self, task: AgentTask) -> AgentResult:
        # Preflight before the retry loop: a missing CLI is fatal and must not be
        # absorbed as a failed attempt (which, against a stale output dir, could
        # otherwise verify and look successful).
        if shutil.which(self.codex_bin) is None:
            raise AgentBackendError(
                f"Codex CLI {self.codex_bin!r} not found on PATH. Install it "
                "(e.g. `npm install -g @openai/codex`), make sure it is on PATH, "
                "or select a different backend (`--agent claude`)."
            )
        return run_verify_loop(
            task,
            lambda prompt, max_turns: self._invoke_agent(task, prompt, max_turns),
            max_attempts=self.max_attempts,
            max_turns=self.max_turns,
        )

    # -- agent invocation --------------------------------------------------
    def _invoke_agent(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        # `codex exec` runs non-interactively and loops until done (no
        # --max-turns flag, so max_turns is unused here, matching the claude
        # CLI). --skip-git-repo-check lets it run in the output dir even when
        # that is not a git repo.
        #
        # Resolve the workdir to an ABSOLUTE path: codex resolves --cd relative
        # to its own working directory, and we also start the process in the
        # workdir via subprocess cwd. A relative workdir (e.g. "out/ista") would
        # therefore be applied twice ("out/ista/out/ista"), which codex reports
        # as "No such file or directory (os error 2)".
        workdir = str(Path(task.workdir).resolve())
        full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
        cmd = [
            self.codex_bin,
            "exec",
            "--cd",
            workdir,
            "--skip-git-repo-check",
        ]
        if self.bypass_sandbox:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd += ["--sandbox", "workspace-write"]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(full_prompt)
        try:
            proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        except FileNotFoundError as exc:  # binary vanished after preflight
            raise AgentBackendError(f"Codex CLI {self.codex_bin!r} not found: {exc}") from exc
        output = proc.stdout + proc.stderr
        # An auth failure is not a fixable task error — retrying would just burn
        # the attempt budget. Surface it immediately.
        if proc.returncode != 0:
            lowered = output.lower()
            if any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
                raise AgentBackendError(
                    "Codex CLI could not authenticate. Run `codex login` or set "
                    "OPENAI_API_KEY. Codex output:\n" + output[-1000:]
                )
        return output
