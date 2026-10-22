"""The (mockable) coding-agent interface.

Code generation delegates the actual translation of the Intermediate Representation (IR) into
target source to an `AgentDriver`. The real driver shells out to the Claude Agent SDK; the
`StubAgentDriver` drops a known-good source so
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
        ``"translation"`` for the translation phase (the initial faithful
        translation), ``"code_optimization"`` for the code optimization phase.
    max_turns:
        Maximum agent turns before giving up.
    bench_cmd:
        Optional benchmark command for the code optimization phase.
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
    phase: Literal["translation", "code_optimization"] = "translation"
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
    agent_log:
        Only the agent's own output, concatenated across attempts, with no
        verify/build output and no harness commentary mixed in. Use this when
        reading anything the agent *said* — notably
        `extract_agent_notes`, which would
        otherwise have to pick the agent's words out of compiler noise.
    turns_used:
        Number of attempts (or turns) consumed producing this result.
    artifacts:
        Paths to the files the agent produced or refined.
    truncated:
        Whether any attempt was cut short by its wall-clock budget. A truncated
        attempt can still leave correct code on disk, so this is independent of
        `success`; it means the agent had no chance to finish, and in
        particular may never have written its summary.
    verify_output:
        The output of the last verify the driver ran, on the sources as the run
        left them. Unlike `log`, which also holds earlier attempts and any
        verify the agent ran itself, this is what decided `success`, so it says
        why a failed run failed. ``""`` if no verify ran.
    """

    success: bool
    log: str = ""
    agent_log: str = ""
    turns_used: int = 0
    artifacts: list[Path] = field(default_factory=list)
    truncated: bool = False
    verify_output: str = ""


def _decode_stream(stream) -> str:
    """Decode a captured subprocess stream that may be bytes, str, or absent.

    ``subprocess.run`` populates ``TimeoutExpired.stdout`` with whatever it
    captured before killing the child, but skips the text decoding it would have
    applied on the normal path — so the timeout branch sees ``bytes`` even under
    ``text=True``.
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


def _redact_exception(exc: BaseException) -> str:
    """Describe an exception without echoing anything it may carry.

    ``subprocess`` exceptions keep the argument vector on ``.cmd``, and
    ``str(exc)`` interpolates it — which for a coding agent means the entire
    prompt, and the system prompt, land in the log. That is both a leak and a
    hazard: the prompt asks the agent to print a summary, so a prompt echo makes
    the log look as if the agent had answered. Report only the safe fields.
    """
    kind = type(exc).__name__
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"{kind}: the agent process exceeded its {exc.timeout}s budget"
    if isinstance(exc, subprocess.CalledProcessError):
        return f"{kind}: the agent process exited with status {exc.returncode}"
    if getattr(exc, "cmd", None) is not None:
        return f"{kind} (details withheld: the exception carries the command line)"
    return f"{kind}: {str(exc)[:500]}"


class AgentBackendError(RuntimeError):
    """A fatal problem with the coding-agent backend itself.

    Raised when the backend cannot run at all — its CLI is missing, or it is not
    authenticated. Unlike an ordinary failed attempt (a turn-cap, a transient
    API error, or a verify that did not pass yet), this is not something
    re-prompting can fix.
    """


class AgentDriver(ABC):
    """Abstract coding-agent that generates a source file until verify passes.

    Subclasses implement `run` to drive a particular backend (the real
    Claude/Codex/Pi CLIs, or `StubAgentDriver`
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
    truncated_probe: Callable[[], bool] | None = None,
    budget_s: float | None = None,
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
        Per-attempt turn budget for the translation phase (the code
        optimization phase uses ``task.max_turns`` instead).
    truncated_probe:
        Optional callable the backend supplies to report whether its last
        attempt was cut short by a wall-clock budget. Its result is recorded on
        `AgentResult.truncated`, so a caller can say so rather than
        presenting a half-finished run as a complete one.
    budget_s:
        Wall-clock budget, in seconds, for the whole phase: every attempt shares
        it, and no new attempt starts once it is spent. It is published to the
        backend as ``task.extra["budget"]`` (a
        `TimeBudget`) so an attempt can
        enforce the time left. ``None`` leaves the phase unbounded.

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
    # Only what the agent itself emitted. Kept apart from the build/verify output
    # so that reading the agent's own words never means sifting through compiler
    # noise — see AgentResult.agent_log.
    agent_parts: list[str] = []
    prompt = task.instructions

    # Larger turn budget for the open-ended code optimization phase.
    turns = task.max_turns if task.phase == "code_optimization" else max(max_turns, 20)

    budget = None
    if budget_s is not None:
        from fomtools.codegen.agent.budget import TimeBudget

        budget = TimeBudget.starting_now(float(budget_s))
        task.extra["budget"] = budget

    attempts_made = 0
    output = ""  # the last verify's output
    for attempt in range(max_attempts):
        if budget is not None and attempt > 0 and budget.remaining() <= 0:
            log_parts.append(
                f"\n[fomtools: the {budget_s:g}s budget for this phase is spent; "
                "no further attempts.]\n"
            )
            break
        attempts_made = attempt + 1
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
            agent_log = f"[agent invocation raised: {_redact_exception(exc)}]"
        log_parts.append(agent_log)
        agent_parts.append(agent_log)

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
                agent_log="".join(agent_parts),
                turns_used=attempt + 1,
                artifacts=[task.workdir / task.solver_rel_path],
                truncated=bool(truncated_probe and truncated_probe()),
                verify_output=output,
            )

        # Re-prompt with the failure.
        prompt = (
            task.instructions
            + "\n\nThe verify command FAILED with this output:\n"
            + output[:4000]
            + "..."
            + output[-4000:]
            + "\nFix "
            + ", ".join(
                f"`{rel}`" for rel in task.extra.get("editable_paths", (task.solver_rel_path,))
            )
            + " and make it pass."
        )

    return AgentResult(
        success=False,
        log="".join(log_parts),
        agent_log="".join(agent_parts),
        turns_used=attempts_made,
        truncated=bool(truncated_probe and truncated_probe()),
        verify_output=output,
    )
