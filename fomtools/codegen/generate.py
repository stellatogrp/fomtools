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
from fomtools.codegen.postprocess import (
    extract_agent_notes,
    unified_solver_diff,
    write_optimization_report,
    write_production_variant,
)
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
    optimization_diff: str | None = None
    optimizations_md: Path | None = None
    production_path: Path | None = None

    @property
    def success(self) -> bool:
        """Whether a correct, verified solver was produced.

        The optimization phase never fails the run: if it does not yield a
        faster, still-correct solver, the verified Phase A solver is shipped
        instead. Use :attr:`optimization_kept` to see whether an optimization
        was actually adopted.
        """
        return self.translate.success

    @property
    def optimization_kept(self) -> bool:
        """True iff Phase B produced a faster, still-correct solver that was kept."""
        return self.optimize is not None and self.optimize.success


def _bench_mean_us(out_dir: Path) -> float | None:
    try:
        return run_bench(out_dir).mean_us
    except Exception:
        return None


def optimization_is_faster(
    c_phase_b: float | None, c_phase_a: float | None, tol: float = 1.02
) -> bool:
    """Decide whether to keep a Phase-B solver based on head-to-head timing.

    Keeps Phase B only if it is at least as fast as Phase A within a small
    tolerance (``tol``) that absorbs benchmark noise. Returns ``False`` if either
    measurement is missing.
    """
    if c_phase_b is None or c_phase_a is None:
        return False
    return c_phase_b <= c_phase_a * tol


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

    # The agent may edit only the solver body; restore all deterministic files
    # (Makefile, headers, numeric data, factorized solve, sparse matvecs) before
    # each verify so it cannot break the build infrastructure or the data.
    def restore_deterministic_files():
        target.scaffold(ir, out_dir)

    # --- Phase A: faithful translation ---------------------------------------
    task_a = AgentTask(
        workdir=out_dir,
        solver_rel_path=target.solver_source_path(),
        verify_cmd=target.verify_command(),
        instructions=phase_a_instructions(ir, target),
        phase="A_translate",
        restore_hook=restore_deterministic_files,
    )
    translate = driver.run(task_a)

    timing = TimingReport() if profile else None
    have_cc = find_compiler() is not None

    # Baseline timing once a correct solver exists.
    if translate.success and profile and have_cc:
        restore_deterministic_files()  # ensure a clean build before measuring
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
    optimization_diff = None
    optimizations_md = None
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
            restore_hook=restore_deterministic_files,
        )
        optimize = driver.run(task_b)
        restore_deterministic_files()  # discard any non-solver edits
        phase_b_source = solver_path.read_text()
        changed = phase_b_source != phase_a_source
        notes: list[str] = []

        # Gate 1 (correctness): the optimized solver must still verify. If it
        # broke (or never verified), fall back to the known-good Phase A solver.
        recheck = build_and_verify(out_dir, target) if (optimize.success and changed) else None
        correct = bool(optimize.success and (recheck is None or recheck.success))
        if not correct and changed:
            solver_path.write_text(phase_a_source)
            phase_b_source = phase_a_source
            changed = False
            notes.append("[optimization not kept: phase-B solver failed verification]")
            if recheck and not recheck.success:
                notes.append(recheck.output)

        # Gate 2 (performance): keep a *changed* solver only if it is at least as
        # fast as Phase A, measured head-to-head now (timings are noisy, so a
        # small tolerance avoids discarding genuine wins). Otherwise ship the
        # faster Phase A solver — the optimization phase never regresses speed.
        kept = changed and correct
        if profile and have_cc and timing is not None and changed and correct:
            c_phase_b = _bench_mean_us(out_dir)
            solver_path.write_text(phase_a_source)
            c_phase_a = _bench_mean_us(out_dir)
            if optimization_is_faster(c_phase_b, c_phase_a):
                solver_path.write_text(phase_b_source)
                timing.c_initial_us, timing.c_final_us = c_phase_a, c_phase_b
            else:
                kept = False
                timing.c_initial_us = timing.c_final_us = c_phase_a
                notes.append(
                    f"[optimization not kept: phase-B build ({c_phase_b} us) was not "
                    f"faster than phase A ({c_phase_a} us)]"
                )

        if profile and have_cc and timing is not None:
            try:
                timing.profile = run_profile(out_dir)
            except Exception:
                pass

        if not changed and not notes:
            notes.append("[no source change from the optimization phase]")

        optimize = AgentResult(
            success=kept,
            log=optimize.log + ("\n" + "\n".join(notes) if notes else ""),
            turns_used=optimize.turns_used,
            artifacts=[solver_path],
        )

        # Record exactly what changed (diff phase A -> final) + the agent's notes.
        final_source = solver_path.read_text()
        optimization_diff = unified_solver_diff(
            phase_a_source, final_source, target.solver_source_path()
        )
        optimizations_md = write_optimization_report(
            out_dir,
            timing_summary=timing.summary() if timing is not None else "",
            diff=optimization_diff,
            agent_notes=extract_agent_notes(optimize.log),
            optimize_succeeded=optimize.success,
        )

    # Always emit a clean, marker-free production copy of the final solver.
    production_path = None
    if translate.success:
        production_path = write_production_variant(out_dir, target.solver_source_path())

    return CodegenResult(
        ir=ir,
        out_dir=out_dir,
        target=target,
        translate=translate,
        optimize=optimize,
        timing=timing,
        optimization_diff=optimization_diff,
        optimizations_md=optimizations_md,
        production_path=production_path,
    )
