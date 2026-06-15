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
        """Relative path of the deterministic linear-solve source."""
        return "src/fom_linsolve.c"

    def agent_notes(self) -> str:
        """Target-specific guidance appended to the coding-agent prompts.

        Default: none. Targets whose helpers deviate from the host-C semantics
        (e.g. CUDA, where ``fom_apply_*`` takes device pointers) override this.
        """
        return ""

    def static_checks(self, source: str) -> list[str]:
        """Return a list of invariant violations in ``source`` (empty = OK).

        Default: no checks. Embedded targets override.
        """
        return []


_C_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_C_LINE_COMMENT = re.compile(r"//[^\n]*")
_C_STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def strip_c_comments_and_strings(source: str) -> str:
    """Remove C comments and string literals so token checks don't false-match.

    Comments/strings are replaced with spaces (preserving rough offsets).
    """
    source = _C_BLOCK_COMMENT.sub(" ", source)
    source = _C_LINE_COMMENT.sub(" ", source)
    source = _C_STRING.sub(" ", source)
    return source
