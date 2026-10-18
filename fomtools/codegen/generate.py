"""End-to-end code generation: extract IR, scaffold, drive the coding agent.

This is the high-level entry point used by ``Verifier.codegen`` and the CLI.

Flow:

1. Extract the IR and scaffold the solver directory.
2. Translation phase — the agent translates the spec into the target language and
   iterates until ``make verify`` passes (numerical equivalence to the reference oracle).
3. If ``code_optimization`` is set, measure the baseline (Python reference time + the
   translated generated-C time + a profiling breakdown), then enter the code
   optimization phase: the agent iteratively profiles, optimizes, and
   re-verifies. We re-verify afterward as a hard correctness gate and re-measure.
4. Assemble a :class:`~fomtools.codegen.harness.TimingReport` (speedups +
   profiling) and attach it to the result.
"""

from __future__ import annotations

import warnings
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
from fomtools.codegen.kernel_rules import function_body, strip_c_comments_and_strings
from fomtools.codegen.oracle import check_stationary
from fomtools.codegen.postprocess import (
    extract_agent_notes,
    parse_ops_checks,
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
    #: How verify checked each shipped ``fom_apply_<name>`` against its matrix (one
    #: dict per matrix: ``name, shape, nnz, rewritten, mode, probes, max_err``), or
    #: ``None`` when the products were not editable or not optimized.
    kernel_checks: list[dict] | None = None

    @property
    def success(self) -> bool:
        """Whether a correct, verified solver was produced.

        The code optimization phase never fails the run: if it does not yield a
        faster, still-correct solver, the verified translated solver is shipped
        instead. Use :attr:`optimization_kept` to see whether an optimization
        was actually adopted.
        """
        return self.translate.success

    @property
    def optimization_kept(self) -> bool:
        """True iff the code optimization phase produced a faster, still-correct solver that was kept."""
        return self.optimize is not None and self.optimize.success


def _static_issues(target, ir, optimized: dict[str, str], translated: dict[str, str]) -> list[str]:
    """Invariant violations in the sources the code optimization phase rewrote.

    Target invariants (e.g. division, malloc) on rewritten non-solver sources:
    the solver is held to these by its prompt and verify, as before, while a
    rewritten linear solve or matrix-vector product is new ground for the agent.
    And, on every rewritten source, the rules that keep its kernels tied to the
    checked data (:meth:`~fomtools.codegen.targets.base.TargetDescriptor.kernel_checks`).
    """
    issues = []
    for rel, text in optimized.items():
        if text == translated.get(rel):
            continue
        found = target.kernel_checks(rel, text, ir, translated.get(rel))
        if rel != target.solver_source_path():
            found = target.static_checks(text) + found
        issues += [f"[optimization not kept: {rel}: {v}]" for v in found]
    return issues


def _routine_rewritten(name: str, translated: str, final: str) -> bool:
    """Whether ``fom_apply_<name>`` differs between two ``fom_ops.c`` sources
    (ignoring comments and whitespace). A routine that cannot be found in either
    is not the generated one, so it counts as rewritten."""

    def body(text: str) -> str | None:
        found = function_body(strip_c_comments_and_strings(text), f"fom_apply_{name}")
        return None if found is None else " ".join(found.split())

    before, after = body(translated), body(final)
    return before is None or after is None or before != after


def _kernel_checks(target, ir, out_dir, final, translated, verify_output) -> list[dict] | None:
    """Rows of the report's kernel-check table, from a verify of the shipped
    ``fom_ops.c`` (the last ``OPSCHECK`` line per matrix in ``verify_output``;
    a fresh ``make verify`` only if it has none)."""
    ops = target.ops_source_path()
    if ops not in final:
        return None
    checks = {c["name"]: c for c in parse_ops_checks(verify_output or "")}
    if not checks:
        checks = {c["name"]: c for c in parse_ops_checks(build_and_verify(out_dir, target).output)}
    rows = []
    for check in checks.values():
        md = ir.matrices[check["name"]]
        rewritten = _routine_rewritten(check["name"], translated[ops], final[ops])
        rows.append({**check, "shape": tuple(md.shape), "nnz": md.nnz(), "rewritten": rewritten})
    return rows


def _bench_mean_us(out_dir: Path) -> float | None:
    try:
        return run_bench(out_dir).mean_us
    except Exception:
        return None


def optimization_is_faster(
    c_optimized: float | None, c_translated: float | None, tol: float = 1.02
) -> bool:
    """Decide whether to keep the optimized solver based on head-to-head timing.

    Keeps the optimized solver only if it is at least as fast as the translated
    solver within a small
    tolerance (``tol``) that absorbs benchmark noise. Returns ``False`` if either
    measurement is missing.
    """
    if c_optimized is None or c_translated is None:
        return False
    return c_optimized <= c_translated * tol


def _write_agent_transcripts(out_dir: Path, translate, optimize) -> None:
    """Persist each phase's raw agent output next to the generated solver.

    The optimization report keeps only a bounded excerpt, so without this the
    agent's full reasoning exists nowhere after the process exits. Best-effort:
    failing to write a transcript must never fail a successful generation.
    """
    for name, result in (("translation", translate), ("code_optimization", optimize)):
        if result is None or not getattr(result, "agent_log", ""):
            continue
        try:
            (out_dir / f"agent_{name}.log").write_text(result.agent_log)
        except OSError:
            pass


def _gate_stationarity(ir, verifier, iterates, param_values, state, params) -> None:
    """Check the rolled IR against the recorded tape, or warn that nothing did.

    Extraction reads a single iteration and repeats it, which silently produces
    the wrong algorithm when the recording is not stationary. This is the only
    check that can catch that, and it needs the recorded iterates to compare
    against, so it stays opt-in — but its absence is worth saying out loud.
    """
    if iterates is None:
        warnings.warn(
            # Not "generate_code() did not ...": the CLI reaches this gate too,
            # including from --scaffold-only, where nothing is generated at all.
            "this extraction did not check that the recorded method is stationary. "
            "Extraction reads one iteration and repeats it, so a coefficient that "
            "changes with the iteration index would be frozen at its recorded value "
            "and the generated solver would run a different algorithm. Pass "
            "iterates= (the successive recorded iterate expressions) and "
            "param_values= to check, or schedules= to declare the variation.",
            UserWarning,
            # Two frames intervene on both paths in: generate_code or the CLI's
            # _extract_fom, then extract_fom_ir, then here.
            stacklevel=4,
        )
        return
    if param_values is None:
        raise ValueError("iterates= requires param_values= (keyed by parameter leaf).")
    check_stationary(ir, verifier, iterates, param_values, state=state, params=params)


def extract_fom_ir(
    verifier,
    next_state=None,
    config: CodegenConfig | None = None,
    state=None,
    params=None,
    name: str = "fom",
    schedules: dict | None = None,
    iterates=None,
    param_values: dict | None = None,
    roll: bool = True,
) -> FomIR:
    """Extract an IR by whichever of the three paths the arguments select.

    The dispatch, and the stationarity gate that goes with the rolled path, live
    here rather than inside :func:`generate_code` so that every caller reaching
    for an IR gets the same one. The CLI used to reimplement this branch for its
    ``--scaffold-only`` mode and drifted: it skipped the gate, so previewing a
    method whose coefficients vary with ``k`` reported nothing while the full run
    on the same input failed.

    See :func:`generate_code` for what each argument selects and means.
    """
    config = config or CodegenConfig()

    if schedules:
        from fomtools.codegen.extract_scheduled import extract_scheduled_ir

        if not callable(verifier):
            raise TypeError(
                "schedules= requires a build function as the first argument: a "
                "callable taking one keyword per schedule and returning a "
                "codegen.Recording."
            )
        return extract_scheduled_ir(verifier, schedules, config=config, name=name)

    if callable(verifier):
        raise TypeError(
            "a build function was passed but no schedules=; call generate_code "
            "with (verifier, next_state) for a stationary method, or with "
            "iterates= to generate the whole recorded run."
        )

    if next_state is None:
        # Unrolled: generate the run the user recorded. Rolls back up when
        # repeating one iteration is verified to reproduce it.
        from fomtools.codegen.extract_unrolled import extract_unrolled_ir

        if iterates is None:
            raise TypeError(
                "generate_code needs either iterates= (the iterate sequence your loop "
                "produced, generating that whole run) or next_state= (the iterate after "
                "one iteration, generating a rolled solver that repeats it)."
            )
        return extract_unrolled_ir(
            verifier,
            iterates,
            config=config,
            name=name,
            state=state,
            params=params,
            roll=roll,
            param_values=param_values,
        )

    ir = extract_ir(
        verifier,
        next_state,
        state=state,
        params=params,
        config=config,
        name=name,
        iterates=iterates,
    )
    _gate_stationarity(ir, verifier, iterates, param_values, state, params)
    return ir


def generate_code(
    verifier,
    next_state=None,
    out_dir: str | Path = None,
    config: CodegenConfig | None = None,
    driver: AgentDriver | None = None,
    state=None,
    params=None,
    name: str = "fom",
    code_optimization: bool = False,
    profile: bool = False,
    schedules: dict | None = None,
    iterates=None,
    param_values: dict | None = None,
    roll: bool = True,
) -> CodegenResult:
    """Generate a low-level solver for the FOM recorded in ``verifier``.

    Three ways in, selected by what you pass:

    * ``iterates=`` with no ``next_state`` generates the **whole recorded run**,
      straight from the loop you wrote to verify with. This accepts arbitrary
      per-iteration coefficients and step types, and anchored methods. It rolls
      the run back up when repeating one iteration is verified to reproduce it,
      so a stationary method costs nothing here. See
      :func:`~fomtools.codegen.extract_unrolled.extract_unrolled_ir`.
    * ``next_state=`` generates a **rolled** solver from one stationary
      iteration, repeated ``config.termination.max_iters`` times.
    * a *build function* plus ``schedules=`` generates a rolled solver whose
      declared coefficients are tabulated and read at the loop counter. See
      :func:`~fomtools.codegen.extract_scheduled.extract_scheduled_ir`.

    Parameters
    ----------
    verifier, next_state, state, params:
        Forwarded to :func:`fomtools.codegen.extract.extract_ir`.

        For a method with an iteration-varying coefficient, pass a *build
        function* as ``verifier`` and its ``schedules`` instead: the function
        records one iteration as a function of the schedule scalars and returns a
        :class:`~fomtools.codegen.extract_scheduled.Recording`. See
        :func:`~fomtools.codegen.extract_scheduled.extract_scheduled_ir`.
    out_dir:
        Directory to emit the solver into.
    config:
        Code-generation configuration.
    driver:
        The coding-agent driver. If ``None``, uses
        :func:`fomtools.codegen.agent.get_default_driver`.
    name:
        FOM name used for file naming.
    code_optimization:
        Run the code optimization phase (iterative profile-guided rewriting)
        after a successful, verified translation.
    profile:
        Collect before/after timing + a profiling breakdown (needs a C
        compiler). Always skipped silently when no compiler is available.
    schedules:
        Iteration-varying scalars, mapping name to a callable of the 0-based
        iteration index, a sequence, or a
        :class:`~fomtools.codegen.schedule.Schedule`. Selects the scheduled
        extraction path.
    iterates:
        The successive recorded iterate expressions, or an
        :class:`~fomtools.iteratetrajectory.IterateTrajectory`.

        Passed *without* ``next_state`` this selects the unrolled path and is the
        run to generate. Passed *with* ``next_state`` it instead checks the rolled
        IR against the recorded tape before anything is emitted, failing if the
        recording is not stationary — strongly recommended there, since without it
        nothing detects a method whose coefficients vary with ``k``.
    param_values:
        Parameter values, keyed by parameter *leaf*, used for whichever check the
        chosen path runs. Optional on the unrolled path, which draws its own.
    roll:
        Unrolled path only: emit the compact rolled IR when repeating one
        iteration is verified to reproduce the recording. Pass ``False`` to unroll
        regardless.
    """
    config = config or CodegenConfig()
    out_dir = Path(out_dir)

    ir = extract_fom_ir(
        verifier,
        next_state,
        config=config,
        state=state,
        params=params,
        name=name,
        schedules=schedules,
        iterates=iterates,
        param_values=param_values,
        roll=roll,
    )

    target = scaffold(ir, out_dir, config)

    if driver is None:
        from fomtools.codegen.agent import get_default_driver

        driver = get_default_driver(config)

    from fomtools.codegen.agent.prompts import (
        code_optimization_instructions,
        translation_instructions,
    )

    # The agent may edit only the solver body; restore all deterministic files
    # (Makefile, headers, numeric data, factorized solve, sparse matvecs) before
    # each verify so it cannot break the build infrastructure or the data.
    def restore_deterministic_files():
        target.scaffold(ir, out_dir)

    # In the code optimization phase the agent may also rewrite the linear solves
    # and the matrix-vector products (target.optimization_editable_paths()):
    # restore everything else, and keep what it wrote to those files.
    editable = tuple(target.optimization_editable_paths())

    def read_editable() -> dict[str, str]:
        return {rel: (out_dir / rel).read_text() for rel in editable}

    def write_editable(sources: dict[str, str]) -> None:
        for rel, text in sources.items():
            (out_dir / rel).write_text(text)

    def restore_all_but_editable():
        kept = read_editable()
        target.scaffold(ir, out_dir)
        write_editable(kept)

    # --- Translation phase ----------------------------------------------------
    translation_task = AgentTask(
        workdir=out_dir,
        solver_rel_path=target.solver_source_path(),
        verify_cmd=target.verify_command(),
        instructions=translation_instructions(ir, target),
        phase="translation",
        restore_hook=restore_deterministic_files,
    )
    translate = driver.run(translation_task)

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
        timing.c_final_us = timing.c_initial_us  # updated after code optimization
        try:
            timing.profile = run_profile(out_dir)
        except Exception:
            timing.profile = []

    # --- Code optimization phase ---------------------------------------------
    optimize = None
    optimization_diff = None
    optimizations_md = None
    kernel_checks = None
    if code_optimization and translate.success:
        translated_sources = read_editable()  # known-good fallback

        optimization_task = AgentTask(
            workdir=out_dir,
            solver_rel_path=target.solver_source_path(),
            verify_cmd=target.verify_command(),
            instructions=code_optimization_instructions(ir, target),
            phase="code_optimization",
            max_turns=config.agent_optimize_max_turns,
            bench_cmd=["make", "bench"],
            restore_hook=restore_all_but_editable,
            extra={"editable_paths": editable},
        )
        # While the agent works, `make verify` also applies the kernel rules, so it
        # sees a violation as it makes it; Gate 1 below applies them again and
        # decides. The shipped directory carries no such config.
        target.write_kernel_check_config(out_dir, ir, translated_sources)
        try:
            optimize = driver.run(optimization_task)
        finally:
            target.remove_kernel_check_config(out_dir)
        restore_all_but_editable()  # discard edits outside the editable sources
        optimized_sources = read_editable()
        changed = optimized_sources != translated_sources
        notes: list[str] = []

        # Gate 1 (correctness): the optimized sources must still verify (and, on
        # the embedded target, pass its static checks). If they broke (or never
        # verified), fall back to the known-good translated sources.
        collected = optimize.success and changed  # a rewrite to re-verify and check
        recheck = build_and_verify(out_dir, target) if collected else None
        static_issues = (
            _static_issues(target, ir, optimized_sources, translated_sources) if collected else []
        )
        correct = bool(
            optimize.success and (recheck is None or recheck.success) and not static_issues
        )
        if not correct and changed:
            write_editable(translated_sources)
            optimized_sources = translated_sources
            changed = False
            notes.append("[optimization not kept: the optimized sources failed verification]")
            if recheck and not recheck.success:
                notes.append(recheck.output)
            notes.extend(static_issues)

        # Gate 2 (performance): keep a *changed* solver only if it is at least as
        # fast as the translated solver, measured head-to-head now (timings are noisy, so a
        # small tolerance avoids discarding genuine wins). Otherwise ship the
        # faster translated solver — the code optimization phase never regresses speed.
        kept = changed and correct
        if profile and have_cc and timing is not None and changed and correct:
            c_optimized = _bench_mean_us(out_dir)
            write_editable(translated_sources)
            c_translated = _bench_mean_us(out_dir)
            if optimization_is_faster(c_optimized, c_translated):
                write_editable(optimized_sources)
                timing.c_initial_us, timing.c_final_us = c_translated, c_optimized
            else:
                kept = False
                timing.c_initial_us = timing.c_final_us = c_translated
                notes.append(
                    f"[optimization not kept: the optimized build ({c_optimized} us) was not "
                    f"faster than the translated solver ({c_translated} us)]"
                )

        if profile and have_cc and timing is not None:
            try:
                timing.profile = run_profile(out_dir)
            except Exception:
                pass

        if not changed and not notes:
            notes.append("[no source change from the code optimization phase]")

        optimize = AgentResult(
            success=kept,
            log=optimize.log + ("\n" + "\n".join(notes) if notes else ""),
            agent_log=optimize.agent_log,
            turns_used=optimize.turns_used,
            artifacts=[out_dir / rel for rel in editable],
            truncated=optimize.truncated,
        )

        # Record exactly what changed (diff translated -> final, per editable
        # source) + the agent's notes.
        final_sources = read_editable()
        # The shipped fom_ops.c's checks: the optimized one was just re-verified
        # when kept; otherwise the shipped one is the generated fom_ops.c, which
        # the translation phase's passing verify already checked.
        kernel_checks = (
            _kernel_checks(
                target,
                ir,
                out_dir,
                final_sources,
                translated_sources,
                recheck.output if (kept and recheck is not None) else translate.log,
            )
            if have_cc
            else None
        )
        optimization_diff = "".join(
            unified_solver_diff(translated_sources[rel], final_sources[rel], rel)
            for rel in editable
        )
        # The agent's own output only: the combined log interleaves compiler and
        # verify output, and carries the pipeline's bracketed notes appended above.
        agent_notes = extract_agent_notes(optimize.agent_log)
        caveats = []
        if optimize.truncated:
            caveats.append(
                "The agent was terminated at its wall-clock budget before it finished, so it "
                "may never have written a summary. The timing and diff below are still sound: "
                "the solver on disk was re-verified and benchmarked. Raise "
                "`CodegenConfig.agent_optimize_timeout_s` to give it longer."
            )
        if not agent_notes.strip():
            caveats.append(
                "No agent summary was recovered from this run, so the 'Agent notes' section "
                "is absent rather than empty-by-choice."
            )
        optimizations_md = write_optimization_report(
            out_dir,
            timing_summary=timing.summary() if timing is not None else "",
            diff=optimization_diff,
            agent_notes=agent_notes,
            optimize_succeeded=optimize.success,
            caveats=caveats,
            kernel_checks=kernel_checks,
            rule_violations=static_issues,
            rewrite_collected=collected,
        )
    # Always persist the raw agent output, code optimization phase or not.
    _write_agent_transcripts(out_dir, translate, optimize)

    # Always emit a clean, marker-free production copy of the final solver.
    production_path = None
    if translate.success:
        production_path = write_production_variant(
            out_dir,
            target.solver_source_path(),
            tuple(rel for rel in editable if rel != target.solver_source_path()),
        )

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
        kernel_checks=kernel_checks,
    )
