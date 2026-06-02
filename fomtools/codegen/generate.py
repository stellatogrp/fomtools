"""End-to-end code generation: extract IR, scaffold, drive the coding agent.

This is the high-level entry point used by ``Verifier.codegen`` and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets import TargetDescriptor


@dataclass
class CodegenResult:
    """Outcome of a code-generation run."""

    ir: FomIR
    out_dir: Path
    target: TargetDescriptor
    translate: AgentResult
    optimize: AgentResult | None = None

    @property
    def success(self) -> bool:
        return self.translate.success and (self.optimize is None or self.optimize.success)


def generate_code(
    verifier,
    next_state,
    out_dir: str | Path,
    config: CodegenConfig | None = None,
    driver: AgentDriver | None = None,
    state=None,
    params=None,
    name: str = "fom",
    run_phase_b: bool = False,
) -> CodegenResult:
    """Generate a low-level solver for the FOM recorded in ``verifier``.

    Parameters
    ----------
    verifier, next_state, state, params:
        Forwarded to :func:`fomtools.codegen.extract.extract_ir`.
    out_dir:
        Directory to emit the solver into.
    config:
        Code-generation configuration.
    driver:
        The coding-agent driver. If ``None``, uses
        :func:`fomtools.codegen.agent.get_default_driver`.
    name:
        FOM name used for file naming.
    run_phase_b:
        Whether to run the performance-optimization phase after a successful
        translation.
    """
    config = config or CodegenConfig()
    out_dir = Path(out_dir)

    ir = extract_ir(verifier, next_state, state=state, params=params, config=config, name=name)
    target = scaffold(ir, out_dir, config)

    if driver is None:
        from fomtools.codegen.agent import get_default_driver

        driver = get_default_driver(config)

    from fomtools.codegen.agent.prompts import phase_a_instructions, phase_b_instructions

    task_a = AgentTask(
        workdir=out_dir,
        solver_rel_path=target.solver_source_path(),
        verify_cmd=target.verify_command(),
        instructions=phase_a_instructions(ir, target),
        phase="A_translate",
    )
    translate = driver.run(task_a)

    optimize = None
    if run_phase_b and translate.success:
        task_b = AgentTask(
            workdir=out_dir,
            solver_rel_path=target.solver_source_path(),
            verify_cmd=target.verify_command(),
            instructions=phase_b_instructions(ir, target),
            phase="B_optimize",
            bench_cmd=["make", "bench"],
        )
        optimize = driver.run(task_b)

    return CodegenResult(
        ir=ir, out_dir=out_dir, target=target, translate=translate, optimize=optimize
    )
