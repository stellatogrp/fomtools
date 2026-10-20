"""Scaffold an emitted-solver directory from a `FomIR`.

This writes every deterministic artifact (ABI header, numeric data, verify
harness, build files, Python wrapper) but *not* the solver body
``src/fom_solver.c`` — that is produced by the coding agent (or supplied by a
golden fixture) in a later stage.
"""

from __future__ import annotations

from pathlib import Path

from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets import TargetDescriptor, get_target
from fomtools.codegen.wrapper import emit_wrapper


def scaffold(
    ir: FomIR,
    out_dir: str | Path,
    config: CodegenConfig | None = None,
    target: TargetDescriptor | None = None,
) -> TargetDescriptor:
    """Write the deterministic skeleton for ``ir`` into ``out_dir``.

    Returns the `TargetDescriptor` used (so callers can find the verify
    command and the solver-source path).
    """
    config = config or CodegenConfig()
    target = target or get_target(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    target.scaffold(ir, out)
    emit_wrapper(ir, out, config)
    return target
