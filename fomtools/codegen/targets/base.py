"""Target descriptor base class.

A :class:`TargetDescriptor` knows how to lay out an emitted solver for one
target (embedded C, large-scale CUDA, …): which deterministic files to write
(ABI header, numeric data, verify harness, build files, Python wrapper) and how
to statically check the agent-written solver source for the target's invariants
(e.g. embedded C forbids dynamic allocation and division).

The single file a target does *not* emit is ``src/fom_solver.c`` — that is the
coding agent's (or a golden fixture's) responsibility.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.ir import FomIR
from fomtools.codegen.kernel_rules import strip_c_comments_and_strings


class TargetDescriptor(ABC):
    """Abstract emitter + checker for one code-generation target."""

    #: Short identifier, e.g. ``"embedded_c"``.
    name: str

    def __init__(self, config: CodegenConfig):
        self.config = config

    @abstractmethod
    def scaffold(self, ir: FomIR, out_dir: Path) -> None:
        """Write every deterministic file for the target into ``out_dir``.

        Must not write ``src/fom_solver.c`` (the agent/golden owns it).
        """

    @abstractmethod
    def verify_command(self) -> list[str]:
        """Shell command (run in ``out_dir``) that builds and verifies; exits
        non-zero on numerical mismatch."""

    @abstractmethod
    def solver_source_path(self) -> str:
        """Relative path (within ``out_dir``) of the agent-written solver."""

    def ops_source_path(self) -> str:
        """Relative path of the deterministic matrix-vector product source."""
        return "src/fom_ops.c"

    def linsolve_source_path(self) -> str:
        """Relative path of the generated linear-solve source."""
        return "src/fom_linsolve.c"

    def objective_source_path(self) -> str:
        """Relative path of the source holding the generated ``fom_objective``.

        Default: the matrix-vector product source. Never editable by the code
        optimization phase: the metric the solver stops on stays package-owned.
        """
        return self.ops_source_path()

    def optimization_editable_paths(self) -> tuple[str, ...]:
        """Sources the code optimization phase may rewrite.

        The solver, the linear solves and the matrix-vector products, which is
        where the time goes: all are re-verified end to end, the solves and the
        products also directly against the package's factorizations and
        matrices (see the generated ``test/verify_main.c``), and kept only if
        faster. Every other file, including the numeric data in ``fom_data.c``,
        is restored from the generator. The translation phase may write only
        the solver.
        """
        return (self.solver_source_path(), self.linsolve_source_path(), self.ops_source_path())

    def agent_notes(self) -> str:
        """Target-specific guidance appended to the coding-agent prompts.

        Default: none. Targets whose helpers deviate from the host-C semantics
        (e.g. CUDA, where ``fom_apply_*`` takes device pointers) override this.
        """
        return ""

    def kernel_checks(self, rel: str, source: str, ir, baseline: str | None = None) -> list[str]:
        """Violations of the rules that keep rewritten kernels checkable (empty = OK).

        Run on each source the code optimization phase changed; ``baseline`` is
        its translated version. Default: none. Targets whose matrix-vector
        products are editable and checked against the package's matrices
        override this (see :mod:`fomtools.codegen.kernel_rules`).
        """
        return []

    def ops_data_path(self) -> str:
        """Where the matrix data the matrix-vector products read lives."""
        return "src/fom_data.c"

    def translation_checks(self, source: str, ir) -> list[str]:
        """Violations of the kernel rules in the translated solver (empty = OK).

        Default: none (see :meth:`kernel_checks`).
        """
        return []

    def write_kernel_check_config(self, out_dir, ir, baselines: dict[str, str] | None) -> None:
        """Let ``make verify`` apply the kernel rules while the agent works.

        Called when the code optimization phase starts, with the translated
        sources as ``baselines`` (:meth:`kernel_checks`), and when translation
        starts, with ``None`` (:meth:`translation_checks`). Default: nothing to
        write.
        """

    def remove_kernel_check_config(self, out_dir) -> None:
        """Undo :meth:`write_kernel_check_config` once the phase is over."""

    def static_checks(self, source: str, ir: FomIR | None = None) -> list[str]:
        """Return a list of invariant violations in ``source`` (empty = OK).

        Default: the contract every target shares (see
        :func:`objective_violations`, which needs ``ir``). Embedded targets add
        their own invariants.
        """
        return objective_violations(source, ir)


def objective_violations(source: str, ir: FomIR | None) -> list[str]:
    """The objective-termination contract of a solver body, for any target.

    Under ``TerminationKind.OBJECTIVE`` the solver must stop through the
    generated ``fom_objective`` (never a metric of its own) and read the runtime
    ``check_termination``. ``make verify`` then checks that it stops where the
    oracle does; this catches the omission statically, with a clearer message.
    """
    if ir is None or not ir.termination.uses_objective:
        return []
    code = strip_c_comments_and_strings(source)
    violations = []
    if not re.search(r"\bfom_objective\s*\(", code):
        violations.append(
            "termination kind 'objective': the solver never calls fom_objective(), the "
            "generated verification objective it must stop on"
        )
    if not re.search(r"\bcheck_termination\b", code):
        violations.append(
            "termination kind 'objective': the solver never reads "
            "settings.check_termination, the runtime interval of the objective check"
        )
    return violations
