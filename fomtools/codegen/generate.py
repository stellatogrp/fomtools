"""End-to-end code generation: extract IR, scaffold, drive the coding agent.

This is the high-level entry point used by ``Verifier.codegen`` and the CLI.

Flow:

1. Extract the IR and scaffold the solver directory.
2. Phase A — the agent translates the spec into the target language and iterates
   until ``make verify`` passes (numerical equivalence to the reference oracle).
3. If ``run_phase_b`` is set, measure the baseline (Python reference time + the
   initial generated-C time + a profiling breakdown), then enter the
   performance-optimization phase: the agent iteratively profiles, optimizes, and
   re-verifies. We re-verify afterward as a hard correctness gate and re-measure.
4. Assemble a :class:`~fomtools.codegen.harness.TimingReport` (speedups +
   profiling) and attach it to the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.harness import (
    TimingReport,
    build_and_verify,
    find_compiler,
    representative_params,
    run_bench,
    run_profile,
    time_python_oracle,
)
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
    timing: TimingReport | None = None

    @property
    def success(self) -> bool:
        return self.translate.success and (self.optimize is None or self.optimize.success)


def _bench_mean_us(out_dir: Path) -> float | None:
    try:
        return run_bench(out_dir).mean_us
    except Exception:
        return None


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
    profile: bool = False,
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
        Run the iterative performance-optimization phase after a successful,
        verified translation.
    profile:
        Collect before/after timing + a profiling breakdown (needs a C
        compiler). Always skipped silently when no compiler is available.
    """
    config = config or CodegenConfig()
    out_dir = Path(out_dir)

    ir = extract_ir(verifier, next_state, state=state, params=params, config=config, name=name)
    target = scaffold(ir, out_dir, config)

    if driver is None:
        from fomtools.codegen.agent import get_default_driver

        driver = get_default_driver(config)

    from fomtools.codegen.agent.prompts import phase_a_instructions, phase_b_instructions

    # --- Phase A: faithful translation ---------------------------------------
    task_a = AgentTask(
        workdir=out_dir,
        solver_rel_path=target.solver_source_path(),
        verify_cmd=target.verify_command(),
        instructions=phase_a_instructions(ir, target),
        phase="A_translate",
    )
    translate = driver.run(task_a)

    timing = TimingReport() if profile else None
    have_cc = find_compiler() is not None

    # Baseline timing once a correct solver exists.
    if translate.success and profile and have_cc:
        sample = representative_params(ir)
        try:
            timing.python_us = time_python_oracle(ir, sample)
        except Exception:
            timing.python_us = None
        timing.c_initial_us = _bench_mean_us(out_dir)
        timing.c_final_us = timing.c_initial_us  # updated after phase B
        try:
            timing.profile = run_profile(out_dir)
        except Exception:
            timing.profile = []

    # --- Phase B: iterative performance optimization -------------------------
    optimize = None
    if run_phase_b and translate.success:
        solver_path = out_dir / target.solver_source_path()
        phase_a_source = solver_path.read_text()  # known-good fallback

        task_b = AgentTask(
            workdir=out_dir,
            solver_rel_path=target.solver_source_path(),
            verify_cmd=target.verify_command(),
            instructions=phase_b_instructions(ir, target),
            phase="B_optimize",
            max_turns=60,
            bench_cmd=["make", "bench"],
        )
        optimize = driver.run(task_b)

        # Hard correctness gate: the optimized solver must still verify. If the
        # optimization left the code broken (or unverified), restore the
        # known-good Phase A solver so we never ship a regressed solver.
        recheck = build_and_verify(out_dir, target) if optimize.success else None
        if not optimize.success or (recheck and not recheck.success):
            solver_path.write_text(phase_a_source)
            note = "[optimization not kept; restored verified phase-A solver]"
            extra = recheck.output if recheck else ""
            optimize = AgentResult(
                success=False,
                log=optimize.log + "\n" + note + "\n" + extra,
                turns_used=optimize.turns_used,
            )

        if profile and have_cc and timing is not None:
            timing.c_final_us = _bench_mean_us(out_dir)
            try:
                timing.profile = run_profile(out_dir)
            except Exception:
                pass

    return CodegenResult(
        ir=ir,
        out_dir=out_dir,
        target=target,
        translate=translate,
        optimize=optimize,
        timing=timing,
    )
