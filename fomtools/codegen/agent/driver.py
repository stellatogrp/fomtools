"""The (mockable) coding-agent interface.

Code generation delegates the actual translation of the Intermediate Representation (IR) into
target source to an :class:`AgentDriver`. The real driver shells out to the Claude Agent SDK; the
:class:`~fomtools.codegen.agent.stub.StubAgentDriver` drops a known-good source so
the whole pipeline is testable without an LLM.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AgentTask:
    """A unit of work handed to a coding agent.

    Attributes
    ----------
    workdir:
        The emitted-solver directory the agent edits. This **must be an
        isolated output directory**, not a project subdirectory: the agent runs
        with bypassed permissions / a disabled sandbox and is scoped to
        ``workdir`` (as ``cwd`` and via ``--add-dir``). If ``workdir`` sits
        inside a larger repository, anything reachable from it — and any path a
        misconfigured ``--add-dir`` widens to — is exposed to the agent.
    solver_rel_path:
        Path (relative to ``workdir``) of the source file the agent must write
        (e.g. ``"src/fom_solver.c"``).
    verify_cmd:
        Command (run in ``workdir``) that builds and verifies; must exit 0 when
        the solver is correct.
    instructions:
        The natural-language task description / prompt for the agent.
    phase:
        ``"A_translate"`` for the initial faithful translation, ``"B_optimize"``
        for the performance pass.
    max_turns:
        Maximum agent turns before giving up.
    bench_cmd:
        Optional benchmark command for the optimization phase.
    restore_hook:
        Optional callable invoked before each verify to restore the
        deterministic files (Makefile, headers, numeric data, solve/ops). The
        agent may only change the solver body, so any edits it makes elsewhere
        are reverted.
    extra:
        Free-form dictionary for backend-specific extra data.
    """

    workdir: Path
    solver_rel_path: str
    verify_cmd: list[str]
    instructions: str = ""
    phase: Literal["A_translate", "B_optimize"] = "A_translate"
    max_turns: int = 40
    bench_cmd: list[str] | None = None
    restore_hook: Callable[[], None] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of an agent run.

    Attributes
    ----------
    success:
        Whether the verify command ultimately passed.
    log:
        Captured agent and verify output, concatenated across attempts.
    turns_used:
        Number of attempts (or turns) consumed producing this result.
    artifacts:
        Paths to the files the agent produced or refined.
    """

    success: bool
    log: str = ""
    turns_used: int = 0
    artifacts: list[Path] = field(default_factory=list)


class AgentBackendError(RuntimeError):
    """A fatal problem with the coding-agent backend itself.

    Raised when the backend cannot run at all — its CLI is missing, or it is not
    authenticated. Unlike an ordinary failed attempt (a turn-cap, a transient
    API error, or a verify that did not pass yet), this is not something
    re-prompting can fix.
    """


class AgentDriver(ABC):
    """Abstract coding-agent that generates a source file until verify passes.

    Subclasses implement :meth:`run` to drive a particular backend (the real
    Claude/Codex/Pi CLIs, or :class:`~fomtools.codegen.agent.stub.StubAgentDriver`
    for tests). The driver owns only ``task.solver_rel_path``; all other files
    are restored via ``task.restore_hook`` before each verify.

    Methods
    -------
    run
        Produce/refine the solver source until the verify command passes.
    """

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        """Produce/refine ``task.solver_rel_path`` until ``task.verify_cmd`` passes.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to
            write, the restore hook, and the verify command to run.

        Returns
        -------
        AgentResult
            The outcome, whose ``success`` reflects whether the verify command
            passed.
        """


def run_verify_loop(
    task: AgentTask,
    invoke_agent: Callable[[str, int], str],
    *,
    max_attempts: int = 5,
    max_turns: int = 16,
) -> AgentResult:
    """Drive a real coding backend against ``task.verify_cmd`` until it passes.

    This is the authoritative retry loop shared by every real backend (the
    Claude SDK/CLI driver, the Codex CLI driver, ...). On each attempt it calls
    runs one backend attempt, restores the deterministic files via ``task.restore_hook``
    (the agent may only change the solver body), then runs ``task.verify_cmd`` *itself*
    to decide success. On failure it re-prompts with the captured verify output until the command
    passes or the budget is spent.

    Parameters
    ----------
    task:
        The unit of work describing the workspace, the solver path to
        write, the restore hook, and the verify command to run.
    invoke_agent:
        Callable running one backend attempt given ``(prompt, max_turns)`` and
        returning its captured log text. It must NOT decide success.
    max_attempts:
        How many prompt/verify rounds to attempt.
    max_turns:
        Per-attempt turn budget for the translation phase (the optimization
        phase uses ``task.max_turns`` instead).

    Returns
    -------
    AgentResult
        The outcome, whose ``success`` reflects whether the verify command
        passed.

    Raises
    ------
    AgentBackendError
        If the backend cannot run at all (missing CLI or not authenticated);
        propagated rather than absorbed as a failed attempt.
    """
    log_parts: list[str] = []
    prompt = task.instructions

    # Larger turn budget for the open-ended optimization phase.
    turns = task.max_turns if task.phase == "B_optimize" else max(max_turns, 20)

    for attempt in range(max_attempts):
        log_parts.append(f"\n===== agent attempt {attempt + 1} =====\n")
        # The agent hitting its turn cap (or any transient error) must not crash
        # the run: capture it and let the authoritative verify below decide,
        # since the agent may already have produced passing code.
        try:
            agent_log = invoke_agent(prompt, turns)
        except AgentBackendError:
            # The backend cannot run at all (missing CLI / not authenticated).
            # Fail fast rather than retrying or masking it as a failed attempt —
            # otherwise a stale solver already in the output dir could verify and
            # make a non-functional backend look successful.
            raise
        except Exception as exc:  # noqa: BLE001 - surface as a failed attempt
            agent_log = f"[agent invocation raised: {exc}]"
        log_parts.append(agent_log)

        if task.restore_hook is not None:
            task.restore_hook()
        verify = subprocess.run(
            task.verify_cmd, cwd=str(task.workdir), capture_output=True, text=True
        )
        output = verify.stdout + verify.stderr
        log_parts.append(output)
        if verify.returncode == 0:
            return AgentResult(
                success=True,
                log="".join(log_parts),
                turns_used=attempt + 1,
                artifacts=[task.workdir / task.solver_rel_path],
            )

        # Re-prompt with the failure.
        prompt = (
            task.instructions
            + "\n\nThe verify command FAILED with this output:\n"
            + output[:4000]
            +"..."
            + output[-4000:]
            + "\nFix `"
            + task.solver_rel_path
            + "` and make it pass."
        )

    return AgentResult(success=False, log="".join(log_parts), turns_used=max_attempts)
