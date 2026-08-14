"""Emit two C solvers for one recorded iteration: as translated, and optimized.

:func:`fomtools.codegen.generate.generate_code` runs both agent phases in one
directory, and the optimization phase overwrites the translation it started from.
That is right for shipping one solver and wrong here, where the two are the
experiment: the emitted solver and the optimized solver are two implementations
of the same algorithm and both have to fly.

So the translation phase runs once, its verified directory is copied, and the
optimization phase runs on the copy.  Both solvers then descend from the same
translation, which is what makes the comparison between them a statement about
the optimization phase rather than about two independent agent runs.

The optimization phase is driven here rather than through ``generate_code``, and
mirrors the gating in :mod:`fomtools.codegen.generate`: restore the deterministic
files so only the solver body can have changed, require the result to still
verify, require it to be no slower than what it started from, and fall back to
the translated source otherwise.

Agent runs are expensive and nondeterministic, so both phases are skipped when
the directory already holds a solver that verifies.  Pass ``force=True`` to
regenerate.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from fomtools.codegen import CodegenConfig, TerminationSpec
from fomtools.codegen.agent.driver import AgentTask
from fomtools.codegen.agent.prompts import phase_b_instructions
from fomtools.codegen.emit import scaffold
from fomtools.codegen.generate import extract_fom_ir, generate_code, optimization_is_faster
from fomtools.codegen.harness import build_and_verify, run_bench
from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets import get_target

#: Iteration count the solvers are generated at.  ``settings.max_iter`` is read
#: at run time by the emitted loop, so this only sets the default and the size of
#: the verify and benchmark runs; the flights choose their own.  Kept small
#: because the emitted benchmark runs five thousand solves and the optimization
#: phase runs it repeatedly: twenty iterations is long enough that the loop, not
#: the per-solve reset, is what the agent sees in its profile.
DEFAULT_MAX_ITERS = 20


def config_for(max_iters: int = DEFAULT_MAX_ITERS) -> CodegenConfig:
    """The code-generation configuration both phases run under."""
    return CodegenConfig(
        termination=TerminationSpec(max_iters=max_iters),
        agent_optimize_max_turns=120,
    )


def default_driver(config: CodegenConfig):
    """The coding agent both phases use."""
    from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

    return ClaudeAgentDriver(config=config)


def extract(recording, boxes, config: CodegenConfig | None = None, name: str = "fom") -> FomIR:
    """The IR alone, without running an agent.

    Used by anything that needs the emitted dimensions or the reference oracle
    without paying for a generation run.  Passing the recorded iterates turns on
    the stationarity check, which is the only thing that would catch a method
    whose coefficients vary with the iteration index.
    """
    return extract_fom_ir(
        recording.verifier,
        recording.next_state,
        config=config or config_for(),
        state=recording.state,
        params=recording.params,
        name=name,
        iterates=recording.iterates,
        param_values=recording.param_values(boxes),
    )


def translate(
    recording,
    boxes,
    out_dir: Path,
    name: str,
    config: CodegenConfig | None = None,
    driver=None,
    force: bool = False,
) -> tuple[FomIR, Path]:
    """Run the translation phase into ``out_dir/emitted``. Returns the IR and path.

    ``recording`` is a :class:`common.recording.Recording` and ``boxes`` its
    :class:`common.recording.ParameterBoxes`, so either formulation works
    unchanged.
    """
    config = config or config_for()
    emitted = Path(out_dir) / "emitted"

    if not force and _verifies(emitted, config):
        print(f"reusing the verified solver already in {emitted}")
        return extract(recording, boxes, config=config, name=name), emitted

    result = generate_code(
        recording.verifier,
        recording.next_state,
        emitted,
        config=config,
        driver=driver or default_driver(config),
        state=recording.state,
        params=recording.params,
        name=name,
        run_phase_b=False,
        iterates=recording.iterates,
        param_values=recording.param_values(boxes),
    )
    if not result.translate.success:
        raise RuntimeError(
            f"the translation phase did not produce a verified solver in {emitted}.\n"
            f"{result.translate.log[-4000:]}"
        )
    return result.ir, emitted


@dataclass
class OptimizeResult:
    """Outcome of running the optimization phase on a copy of the emitted solver."""

    path: Path
    kept: bool
    notes: str = ""
    emitted_us: float | None = None  # per solve, at the generated iteration count
    optimized_us: float | None = None


def optimize(
    ir: FomIR,
    emitted: Path,
    out_dir: Path,
    config: CodegenConfig | None = None,
    driver=None,
    force: bool = False,
) -> OptimizeResult:
    """Copy the emitted solver and run the optimization phase on the copy."""
    config = config or config_for()
    optimized = Path(out_dir) / "optimized"

    if not force and _verifies(optimized, config):
        print(f"reusing the optimized solver already in {optimized}")
        return OptimizeResult(path=optimized, kept=True, notes="reused from a previous run")

    if optimized.exists():
        shutil.rmtree(optimized)
    shutil.copytree(emitted, optimized)

    target = scaffold(ir, optimized, config)
    solver_path = optimized / target.solver_source_path()
    before = solver_path.read_text()

    task = AgentTask(
        workdir=optimized,
        solver_rel_path=target.solver_source_path(),
        verify_cmd=target.verify_command(),
        instructions=phase_b_instructions(ir, target),
        phase="B_optimize",
        max_turns=config.agent_optimize_max_turns,
        bench_cmd=["make", "bench"],
        restore_hook=lambda: target.scaffold(ir, optimized),
    )
    outcome = (driver or default_driver(config)).run(task)
    target.scaffold(ir, optimized)  # discard any edit outside the solver body
    (optimized / "agent_phase_b_optimize.log").write_text(outcome.agent_log or outcome.log)

    after = solver_path.read_text()
    if after == before:
        return OptimizeResult(
            path=optimized, kept=False, notes="the optimization phase left the source unchanged"
        )

    recheck = build_and_verify(optimized, target)
    if not (outcome.success and recheck.success):
        solver_path.write_text(before)
        return OptimizeResult(
            path=optimized,
            kept=False,
            notes=f"the optimized solver did not verify:\n{recheck.output[-2000:]}",
        )

    fast = _bench(optimized)
    solver_path.write_text(before)
    slow = _bench(optimized)
    if not optimization_is_faster(fast, slow):
        return OptimizeResult(
            path=optimized,
            kept=False,
            notes=f"the optimized solver ({fast} us) was not faster than the emitted one ({slow} us)",
            emitted_us=slow,
            optimized_us=fast,
        )
    solver_path.write_text(after)
    return OptimizeResult(path=optimized, kept=True, emitted_us=slow, optimized_us=fast)


def _verifies(directory: Path, config: CodegenConfig) -> bool:
    """Whether ``directory`` already holds a solver that passes ``make verify``."""
    if not (directory / "src" / "fom_solver.c").exists():
        return False
    return build_and_verify(directory, get_target(config)).success


def _bench(workdir: Path) -> float | None:
    try:
        return run_bench(workdir).mean_us
    except Exception:
        return None
