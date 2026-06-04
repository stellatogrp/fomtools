"""The (mockable) coding-agent interface.

Code generation delegates the actual translation of the IR into target source to
an :class:`AgentDriver`. The real driver shells out to the Claude Agent SDK; the
:class:`~fomtools.codegen.agent.stub.StubAgentDriver` drops a known-good source so
the whole pipeline is testable without an LLM.
"""

from __future__ import annotations

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
        The emitted-solver directory the agent edits.
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
    """

    workdir: Path
    solver_rel_path: str
    verify_cmd: list[str]
    instructions: str = ""
    phase: Literal["A_translate", "B_optimize"] = "A_translate"
    max_turns: int = 40
    bench_cmd: list[str] | None = None
    # Called before each verify to restore the deterministic files (Makefile,
    # headers, numeric data, solve/ops): the agent may only change the solver
    # body, so any edits it makes elsewhere are reverted.
    restore_hook: Callable[[], None] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of an agent run."""

    success: bool
    log: str = ""
    turns_used: int = 0
    artifacts: list[Path] = field(default_factory=list)


class AgentDriver(ABC):
    """Drives generation of one source file until ``verify_cmd`` passes."""

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        """Produce/refine ``task.solver_rel_path`` until ``task.verify_cmd`` passes."""
