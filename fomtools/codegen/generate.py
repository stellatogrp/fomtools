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
4. Assemble a `TimingReport` (speedups +
   profiling) and attach it to the result.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace
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
from fomtools.codegen.kernel_rules import functions, reachable, strip_c_comments_and_strings
from fomtools.codegen.oracle import check_stationary
from fomtools.codegen.postprocess import (
    extract_agent_notes,
    extract_applied_optimizations,
    extract_rule_friction,
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
    #: Optimizations the agent reports a rule blocked (see `CodeOptimizationResult`).
    rule_friction: list[dict] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether a correct, verified solver was produced.

        The code optimization phase never fails the run: if it does not yield a
        faster, still-correct solver, the verified translated solver is shipped
        instead. Use `optimization_kept` to see whether an optimization
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
    checked data (`kernel_checks`).
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
    """Whether ``fom_apply_<name>`` (``fom_fused_<block>`` for a ``name`` of
    ``fused_<block>``) computes differently between two ``fom_ops.c``
    sources (ignoring comments and whitespace): its body, a function it reaches,
    or the file scope (macros, static data) changed. A routine that cannot be
    found in either is not the generated one, so it counts as rewritten."""

    def parse(text: str) -> tuple[dict[str, list[str]], str]:
        bodies, scope = functions(strip_c_comments_and_strings(text))
        return {fn: [" ".join(b.split()) for b in bs] for fn, bs in bodies.items()}, " ".join(
            scope.split()
        )

    (before, before_scope), (after, after_scope) = parse(translated), parse(final)
    if name == "iterate" or name.startswith(("fused_", "step_")):
        fn = f"fom_{name}"
    else:
        fn = f"fom_apply_{name}"
    if fn not in before or fn not in after or before_scope != after_scope:
        return True
    return any(before.get(g) != after.get(g) for g in reachable(after, [fn]))


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
        _, shape, nnz = target.ops_check_info(ir, check["name"])
        rewritten = _routine_rewritten(check["name"], translated[ops], final[ops])
        rows.append({**check, "shape": shape, "nnz": nnz, "rewritten": rewritten})
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
    here rather than inside `generate_code` so that every caller reaching
    for an IR gets the same one. The CLI used to reimplement this branch for its
    ``--scaffold-only`` mode and drifted: it skipped the gate, so previewing a
    method whose coefficients vary with ``k`` reported nothing while the full run
    on the same input failed.

    See `generate_code` for what each argument selects and means.
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
      `extract_unrolled_ir`.
    * ``next_state=`` generates a **rolled** solver from one stationary
      iteration, repeated ``config.termination.max_iters`` times.
    * a *build function* plus ``schedules=`` generates a rolled solver whose
      declared coefficients are tabulated and read at the loop counter. See
      `extract_scheduled_ir`.

    Parameters
    ----------
    verifier, next_state, state, params:
        Forwarded to `fomtools.codegen.extract.extract_ir`.

        For a method with an iteration-varying coefficient, pass a *build
        function* as ``verifier`` and its ``schedules`` instead: the function
        records one iteration as a function of the schedule scalars and returns a
        `Recording`. See
        `extract_scheduled_ir`.
    out_dir:
        Directory to emit the solver into.
    config:
        Code-generation configuration.
    driver:
        The coding-agent driver. If ``None``, uses
        `fomtools.codegen.agent.get_default_driver`.
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
        `Schedule`. Selects the scheduled
        extraction path.
    iterates:
        The successive recorded iterate expressions, or an
        `IterateTrajectory`.

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
    elif getattr(driver, "config", False) is None:
        # A driver built without a config (e.g. ClaudeAgentDriver()) takes this
        # run's budgets instead of its constructor default.
        driver.config = config

    from fomtools.codegen.agent.prompts import translation_instructions

    # The agent may edit only the solver body; restore all deterministic files
    # (Makefile, headers, numeric data, factorized solve, sparse matvecs) before
    # each verify so it cannot break the build infrastructure or the data.
    def restore_deterministic_files():
        target.scaffold(ir, out_dir)

    # In the code optimization phase the agent may also rewrite the linear solves
    # and the matrix-vector products (see run_code_optimization).
    editable = tuple(target.optimization_editable_paths())

    # --- Translation phase ----------------------------------------------------
    translation_task = AgentTask(
        workdir=out_dir,
        solver_rel_path=target.solver_source_path(),
        verify_cmd=target.verify_command(),
        instructions=translation_instructions(ir, target),
        phase="translation",
        restore_hook=restore_deterministic_files,
    )
    # `make verify` also holds the translated solver to the kernel rule that has
    # no baseline (it reads no matrix data: every product goes through a checked
    # fom_apply_*); the check below applies it again and decides.
    target.write_kernel_check_config(out_dir, ir, None)
    try:
        translate = driver.run(translation_task)
    finally:
        target.remove_kernel_check_config(out_dir)
    if translate.success:
        solver = (out_dir / target.solver_source_path()).read_text()
        issues = target.translation_checks(solver, ir)
        if issues:
            notes = "\n".join(f"[translation rejected: {v}]" for v in issues)
            translate = replace(translate, success=False, log=translate.log + "\n" + notes)

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
    rule_friction: list[dict] = []
    if code_optimization and translate.success:
        opt = run_code_optimization(
            ir,
            target,
            out_dir,
            driver,
            config,
            profile=profile,
            timing=timing,
            translated_verify_output=translate.log,
        )
        optimize = opt.optimize
        optimization_diff = opt.diff
        optimizations_md = opt.optimizations_md
        kernel_checks = opt.kernel_checks
        rule_friction = opt.rule_friction
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
        rule_friction=rule_friction,
    )


@dataclass
class OptimizationRound:
    """One round of the code optimization phase.

    ``us_before`` and ``us_after`` are the head-to-head ``make bench`` means of
    the best solver before the round and of the round's rewrite (``None`` when
    not measured); ``gain`` is ``(us_before - us_after) / us_before``.
    ``reason`` says, on one line, why a round was reverted or made no gain
    (later rounds' prompts carry it). ``stop_reason`` is set on the round that
    ended the phase.
    """

    index: int
    changed: bool
    correct: bool
    kept: bool
    us_before: float | None = None
    us_after: float | None = None
    gain: float | None = None
    agent_wall_s: float = 0.0
    truncated: bool = False
    reason: str | None = None
    stop_reason: str | None = None
    #: What the round's summary lists under ``Optimizations applied:`` and
    #: ``Blocked by rules:`` (`extract_applied_optimizations`,
    #: `extract_rule_friction`); later rounds' prompts carry them.
    applied: list[str] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CodeOptimizationResult:
    """Outcome of `run_code_optimization`.

    ``optimize`` combines the rounds' agent runs (logs concatenated, labelled by
    round; ``success`` is whether any round was kept). ``translated_us`` and
    ``final_us`` are head-to-head bench means of the translated and the shipped
    solver (``None`` when not measured).
    """

    optimize: AgentResult
    kept: bool
    rounds: list[OptimizationRound]
    translated_us: float | None
    final_us: float | None
    diff: str
    agent_notes: str
    static_issues: list[str]
    rewrite_collected: bool
    kernel_checks: list[dict] | None
    optimizations_md: Path | None
    #: What the agent says a rule blocked, one dict per item (``round, idea,
    #: rule, expected_gain``), from every round's summary.
    rule_friction: list[dict] = field(default_factory=list)


#: Why a round ended the phase.
STOP_NO_CHANGE = "the round left the sources unchanged"
STOP_PATIENCE = "agent_optimize_patience rounds in a row without a gain of agent_optimize_min_gain"
STOP_MAX_ROUNDS = "the last of agent_optimize_rounds rounds"
STOP_UNMEASURED = "no benchmark (profile off or no C compiler): one round only"
# Former stop reasons, now a round's ``reason`` that uses up patience instead.
STOP_FAILED = "the round's sources failed verification or the kernel rules"
STOP_NOT_FASTER = "the round's sources were not faster than the best so far"
STOP_SMALL_GAIN = "the round gained less than agent_optimize_min_gain"


def _failure_line(text: str, limit: int = 200) -> str:
    """One line of a failed verify's output that says why, cut to ``limit``.

    The first ``VERIFY FAILED`` line, else the first line naming an error (a
    compiler's), else the last non-empty line; ``""`` for empty output.
    """
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    line = next(
        (ln for ln in lines if "VERIFY FAILED" in ln),
        next((ln for ln in lines if "error" in ln.lower()), lines[-1] if lines else ""),
    )
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def run_code_optimization(
    ir: FomIR,
    target: TargetDescriptor,
    out_dir: str | Path,
    driver: AgentDriver,
    config: CodegenConfig,
    *,
    profile: bool = True,
    timing: TimingReport | None = None,
    translated_verify_output: str | None = None,
    write_report: bool = True,
) -> CodeOptimizationResult:
    """Run the code optimization phase on ``out_dir``, in rounds, and gate each one.

    ``out_dir`` must hold a scaffolded solver whose editable sources
    (``target.optimization_editable_paths()``: the solver, the linear solves and
    the matrix-vector products) already pass verify; `generate_code` calls this
    on its own output after translation, and a caller that wants to keep the
    translated solver can copy it and call this on the copy.

    Each round is a fresh agent run (with its own ``agent_optimize_timeout_s``)
    on the best solver so far. After it, everything but the editable sources is
    restored, and the round's sources must

    1. still verify and pass the kernel rules, and
    2. gain at least ``agent_optimize_min_gain`` over the best so far, head to
       head. The first round keeps a rewrite within `optimization_is_faster`'s
       2% noise tolerance, as the single-pass phase always has.

    A round that misses either gate is reverted to the best so far, and its
    ``reason`` goes into the next round's prompt with the earlier rounds'
    summaries. ``agent_optimize_patience`` rounds in a row without a gain (a
    kept first round within the tolerance counts) end the phase, as do the last
    of ``agent_optimize_rounds`` and a round that leaves the sources unchanged.
    Without a benchmark (``profile`` off or no compiler) a correct rewrite is
    kept unmeasured and only one round runs.

    ``timing``, if given, gets ``c_initial_us`` / ``c_final_us`` (the translated
    and the shipped solver, head to head) and the shipped solver's profile.
    ``translated_verify_output`` is the translated solver's verify output, used
    for the kernel-check table when no round is kept.
    """
    from fomtools.codegen.agent.prompts import (
        code_optimization_continuation,
        code_optimization_instructions,
    )

    out_dir = Path(out_dir)
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

    measure = profile and find_compiler() is not None
    translated = read_editable()  # known-good fallback
    best = translated
    best_us: float | None = None
    translated_us: float | None = None
    best_verify_output = translated_verify_output
    best_profile: list = []
    rounds: list[OptimizationRound] = []
    logs: list[str] = []
    agent_logs: list[str] = []
    notes_by_round: list[tuple[int, str]] = []
    static_issues: list[str] = []
    collected_any = False
    turns = 0
    truncated_any = False
    base_instructions = code_optimization_instructions(ir, target)
    n_rounds = config.agent_optimize_rounds

    misses = 0  # rounds in a row without a gain
    for index in range(1, n_rounds + 1):
        instructions = base_instructions
        if index > 1:
            last = rounds[-1]
            instructions += code_optimization_continuation(
                index,
                n_rounds,
                best_us,
                translated_us,
                last.gain if last.kept else None,
                best_profile,
                history=[r.as_dict() for r in rounds],
                patience=config.agent_optimize_patience,
            )
        task = AgentTask(
            workdir=out_dir,
            solver_rel_path=target.solver_source_path(),
            verify_cmd=target.verify_command(),
            instructions=instructions,
            phase="code_optimization",
            max_turns=config.agent_optimize_max_turns,
            bench_cmd=["make", "bench"],
            restore_hook=restore_all_but_editable,
            extra={"editable_paths": editable},
        )
        # While the agent works, `make verify` also applies the kernel rules, so it
        # sees a violation as it makes it; the gate below applies them again and
        # decides. The shipped directory carries no such config.
        target.write_kernel_check_config(out_dir, ir, translated)
        start = time.monotonic()
        try:
            result = driver.run(task)
        finally:
            target.remove_kernel_check_config(out_dir)
        wall = time.monotonic() - start
        restore_all_but_editable()  # discard edits outside the editable sources
        candidate = read_editable()

        header = f"\n===== code optimization round {index} =====\n"
        logs.append(header + (result.log or ""))
        agent_logs.append(header + (result.agent_log or ""))
        notes = extract_agent_notes(result.agent_log or "")
        if notes.strip():
            notes_by_round.append((index, notes))
        turns += result.turns_used or 0
        truncated_any = truncated_any or bool(result.truncated)

        changed = candidate != best
        record = OptimizationRound(
            index=index,
            changed=changed,
            correct=False,
            kept=False,
            agent_wall_s=wall,
            truncated=bool(result.truncated),
            applied=extract_applied_optimizations(notes),
            blocked=extract_rule_friction(notes),
        )
        rounds.append(record)
        if not changed:
            record.correct = bool(result.success)
            record.stop_reason = STOP_NO_CHANGE
            logs.append("[no source change from this code optimization round]")
            break

        # Gate 1 (correctness): the round's sources must still verify and pass
        # the kernel rules (and, on the embedded target, its static checks).
        # Otherwise fall back to the best sources so far.
        collected_any = collected_any or bool(result.success)
        recheck = build_and_verify(out_dir, target) if result.success else None
        issues = _static_issues(target, ir, candidate, translated) if result.success else []
        record.correct = bool(result.success and recheck.success and not issues)
        gained = False
        if not record.correct:
            write_editable(best)
            static_issues += issues
            logs.append(
                "[optimization round not kept: its sources failed verification]"
                + ("\n" + recheck.output if recheck and not recheck.success else "")
                + ("\n" + "\n".join(issues) if issues else "")
            )
            # Why, from the verify of the sources the round left: the pipeline's
            # recheck, or else the driver's last verify (not the first failure in
            # the agent's log, which it may have fixed since).
            failed_output = recheck.output if recheck else result.verify_output
            if issues and recheck.success:
                record.reason = "broke a rule: " + _failure_line(issues[0])
            elif line := _failure_line(failed_output or ""):
                record.reason = "failed verification: " + line
            else:
                record.reason = "the agent's run did not end with a passing verify"
            if result.truncated:
                record.reason += " (the agent stopped at its budget)"
        elif not measure:
            best, best_verify_output = candidate, recheck.output
            record.kept = True
            record.stop_reason = STOP_UNMEASURED
            break
        else:
            # Gate 2 (performance): head to head against the best so far,
            # measured now (timings are noisy, so both are measured back to back).
            us_after = _bench_mean_us(out_dir)
            write_editable(best)
            us_before = _bench_mean_us(out_dir)
            if translated_us is None:  # nothing kept yet: the best is the translation
                translated_us = us_before
            record.us_before, record.us_after = us_before, us_after
            if us_before and us_after:
                record.gain = (us_before - us_after) / us_before
            gained = record.gain is not None and record.gain >= config.agent_optimize_min_gain
            if gained or (index == 1 and optimization_is_faster(us_after, us_before)):
                write_editable(candidate)
                best, best_us, best_verify_output = candidate, us_after, recheck.output
                record.kept = True
            else:
                logs.append(
                    f"[optimization round not kept: its build ({us_after} us) did not gain "
                    f"{config.agent_optimize_min_gain:.1%} over the best so far ({us_before} us)]"
                )
            if not gained:
                record.reason = "not enough gain: " + (
                    f"{us_after:.3f} us vs {us_before:.3f} us best ({record.gain:+.1%})"
                    if record.gain is not None
                    else "not measured"
                )

        misses = 0 if gained else misses + 1
        if misses >= config.agent_optimize_patience:
            record.stop_reason = STOP_PATIENCE
            break
        if index == n_rounds:
            record.stop_reason = STOP_MAX_ROUNDS
            break
        if record.kept:
            try:
                best_profile = run_profile(out_dir)
            except Exception:
                best_profile = []

    write_editable(best)
    kept = best != translated

    if timing is not None and measure:
        if translated_us is None:  # no round reached the benchmark
            translated_us = timing.c_initial_us or _bench_mean_us(out_dir)
        timing.c_initial_us = translated_us
        timing.c_final_us = best_us if kept else translated_us
        try:
            timing.profile = run_profile(out_dir)
        except Exception:
            pass
    final_us = (best_us if kept else translated_us) if measure else None

    optimize = AgentResult(
        success=kept,
        log="".join(logs),
        agent_log="".join(agent_logs),
        turns_used=turns,
        artifacts=[out_dir / rel for rel in editable],
        truncated=truncated_any,
    )

    have_cc = find_compiler() is not None
    # The shipped fom_ops.c's checks: the optimized one was re-verified when
    # kept; otherwise the shipped one is the generated fom_ops.c, which the
    # translation phase's passing verify already checked.
    kernel_checks = (
        _kernel_checks(target, ir, out_dir, best, translated, best_verify_output)
        if have_cc
        else None
    )
    diff = "".join(unified_solver_diff(translated[rel], best[rel], rel) for rel in editable)
    if len(rounds) == 1:
        agent_notes = notes_by_round[0][1] if notes_by_round else ""
    else:
        agent_notes = "\n\n".join(f"### Round {i}\n\n{n.strip()}" for i, n in notes_by_round)

    rule_friction = [
        dict(round=i, **item) for i, n in notes_by_round for item in extract_rule_friction(n)
    ]

    optimizations_md = None
    if write_report:
        caveats = []
        if truncated_any:
            caveats.append(
                "The agent was terminated at its wall-clock budget before it finished (in "
                f"round {', '.join(str(r.index) for r in rounds if r.truncated)}), so it may "
                "never have written a summary. The timing and diff below are still sound: "
                "the solver on disk was re-verified and benchmarked. Raise "
                "`CodegenConfig.agent_optimize_timeout_s` to give it longer."
            )
        if not agent_notes.strip():
            caveats.append(
                "No agent summary was recovered from this run, so the 'Agent notes' section "
                "is absent rather than empty-by-choice."
            )
        if timing is not None:
            timing_summary = timing.summary()
        elif translated_us and final_us:
            timing_summary = (
                f"translated (emitted) {translated_us:9.1f} us/solve\n"
                f"optimized            {final_us:9.1f} us/solve\n"
                f"speedup              {translated_us / final_us:8.2f}x"
            )
        else:
            timing_summary = ""
        optimizations_md = write_optimization_report(
            out_dir,
            timing_summary=timing_summary,
            diff=diff,
            agent_notes=agent_notes,
            optimize_succeeded=kept,
            caveats=caveats,
            kernel_checks=kernel_checks,
            rule_violations=static_issues,
            rewrite_collected=collected_any,
            rounds=[r.as_dict() for r in rounds],
            rule_friction=rule_friction,
        )

    return CodeOptimizationResult(
        optimize=optimize,
        kept=kept,
        rounds=rounds,
        translated_us=translated_us,
        final_us=final_us,
        diff=diff,
        agent_notes=agent_notes,
        static_issues=static_issues,
        rewrite_collected=collected_any,
        kernel_checks=kernel_checks,
        optimizations_md=optimizations_md,
        rule_friction=rule_friction,
    )
