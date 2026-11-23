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

import base64
import difflib
import html
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask
from fomtools.codegen.agent.prompts import SUMMARY_BEGIN
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.harness import (
    BenchStats,
    TimingReport,
    build_and_verify,
    find_compiler,
    parse_bench,
    representative_params,
    run_bench,
    run_profile,
    time_python_oracle,
)
from fomtools.codegen.ir import FomIR
from fomtools.codegen.kernel_rules import functions, reachable, strip_c_comments_and_strings
from fomtools.codegen.manifest import remove_manifest, write_manifest
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
from fomtools.codegen.proc import DEFAULT_CHECK_TIMEOUT_S, run_check
from fomtools.codegen.progress import Progress, format_time, verify_numbers
from fomtools.codegen.report_svg import (
    REPORT_SVG_NAME,
    remove_report_svg,
    result_report_svg,
    write_report_svg,
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
    #: The rounds of the code optimization phase, in order.
    rounds: list[OptimizationRound] = field(default_factory=list)
    #: ``(samples, worst max-abs error, tolerance)`` of the shipped solver's
    #: verify, or ``None`` when it compared nothing numerically.
    verify: tuple[int, float, float | None] | None = None
    #: The report card (``report.svg``) of a run that produced a verified solver.
    report_svg: Path | None = None

    def summary(self) -> str:
        """One line: whether the run succeeded, the verify numbers and the times."""
        name = getattr(self.ir, "name", None) or "solver"
        if not self.success:
            return f"{name}: code generation FAILED, no verified solver in {self.out_dir}"
        parts = []
        if self.verify:
            parts.append(f"verified on {self.verify[0]} samples, worst error {self.verify[1]:.1e}")
        else:
            parts.append("built (verify compared nothing numerically)")
        timing = self.timing
        if timing and timing.c_final_us:
            parts.append(f"{format_time(timing.c_final_us)} per solve")
            if timing.speedup_total:
                parts.append(f"{timing.speedup_total:,.0f}x the NumPy reference")
        if self.optimize is not None:
            kept = sum(1 for r in self.rounds if r.kept)
            parts.append(
                f"{kept} of {len(self.rounds)} optimization rounds kept"
                if self.rounds
                else f"optimization {'kept' if self.optimization_kept else 'not kept'}"
            )
        return f"{name}: " + "; ".join(parts) + f". Output in {self.out_dir}"

    def _repr_html_(self) -> str:
        """The report card and `summary` for Jupyter and marimo; never raises.

        The card is embedded as an image, so its styles stay out of the page.
        """
        try:
            ok = bool(self.success)
        except Exception:  # a display hook must not break the notebook cell
            ok = False
        try:
            text = html.escape(self.summary())
        except Exception:
            text = "code generation result" if ok else "code generation FAILED"
        if not ok:
            return f"<p><strong>{text}</strong></p>"
        try:
            card = base64.b64encode(result_report_svg(self).encode("utf-8")).decode("ascii")
        except Exception:
            return f"<p>{text}</p>"
        return (
            f'<div><img alt="{text}" style="max-width:100%;height:auto" '
            f'src="data:image/svg+xml;base64,{card}"><p>{text}</p></div>'
        )

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


#: Timing runs per solver; their median is the solver's time (one run is noisy).
BENCH_RUNS = 5

#: The noise band of a head-to-head benchmark, in median absolute deviations of
#: the paired ratios. With five pairs and no real difference, the median ratio
#: leaves a band of 2 MADs on a given side in fewer than 1 comparison in 10
#: (simulated with normal, heavy-tailed and spiky noise), and a gain of three
#: times the noise of one timing is detected about 9 times in 10.
NOISE_MADS = 2.0

#: Largest loss, as a relative time, that the first round tolerates: the noise
#: band of that round is capped here. A band that only follows the measured
#: noise keeps a real regression when the timings are noisy (with a run-to-run
#: noise of 2% per timing, a 5% slower rewrite in 23% of comparisons, against 3%
#: with a fixed 2% tolerance, simulated). The cap is that old fixed tolerance.
FIRST_ROUND_MAX_LOSS = 0.02

#: Fewest pairs of successful runs that count as a measurement. The MAD of one
#: pair is 0, so the rule would decide on a single ratio, and with no real
#: difference a later round would be kept in 50% of comparisons with one pair,
#: 15% with two, 18% with three, 10% with four and 8% with all five (simulated).
MIN_PAIRS = 4


def _build_bench(
    out_dir: Path, name: str, timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S
) -> Path | None:
    """Build ``bench_bin`` from the current sources and keep a copy as ``name``.

    ``make bench`` also runs the binary once, which serves as a warmup. Each
    build and each timing run is ended after ``timeout_s`` seconds and then
    counts as failed (``None``), here and in the helpers below.
    """
    try:
        run_bench(out_dir, timeout_s=timeout_s)
        dst = Path(out_dir) / name
        shutil.copy2(Path(out_dir) / "bench_bin", dst)
        return dst
    except Exception:
        return None


def _run_bench_bin(
    path: Path, out_dir: Path, timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S
) -> float | None:
    proc = run_check([str(path)], out_dir, timeout_s)
    if proc.returncode != 0:
        return None
    try:
        return parse_bench(proc.stdout + proc.stderr).mean_us
    except ValueError:
        return None


def _bench_mean_us(
    out_dir: Path, runs: int = BENCH_RUNS, timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S
) -> float | None:
    """Per-solve time of the current sources: the median of ``runs`` bench runs."""
    path = _build_bench(out_dir, ".bench_current", timeout_s)
    if path is None:
        return None
    try:
        stats = BenchStats.of(_run_bench_bin(path, out_dir, timeout_s) for _ in range(runs))
        return stats.median if stats else None
    finally:
        path.unlink(missing_ok=True)


def _head_to_head_us(
    out_dir: Path,
    write_editable,
    candidate: dict,
    best: dict,
    runs: int = BENCH_RUNS,
    timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S,
) -> tuple[BenchStats | None, BenchStats | None, BenchStats | None]:
    """Per-solve times of ``candidate`` and ``best``, measured alternately.

    Each is built once; then the two binaries run ``runs`` times each, in
    alternating order, so slow drifts (clock speed, other load) hit both alike.
    Returns the statistics of the candidate's times, of the best's times, and
    of the paired ratios candidate/best (pair ``i`` is run ``i`` of each side,
    taken back to back; a pair with a failed run is left out). Each is ``None``
    when it has no measurement. Leaves ``best`` written.
    """
    times: dict[str, list] = {"candidate": [], "best": []}
    copies = {"candidate": ".bench_candidate", "best": ".bench_best"}
    try:  # the builds too: an interrupt during the second must not leave the first
        write_editable(candidate)
        cand_bin = _build_bench(out_dir, copies["candidate"], timeout_s)
        write_editable(best)
        best_bin = _build_bench(out_dir, copies["best"], timeout_s)
        bins = {"candidate": cand_bin, "best": best_bin}
        for i in range(runs):
            for key in ("candidate", "best") if i % 2 == 0 else ("best", "candidate"):
                if bins[key] is not None:
                    times[key].append(_run_bench_bin(bins[key], out_dir, timeout_s))
        ratios = [c / b for c, b in zip(times["candidate"], times["best"]) if c and b]
        return (
            BenchStats.of(times["candidate"]),
            BenchStats.of(times["best"]),
            BenchStats.of(ratios),
        )
    finally:
        for name in copies.values():
            (Path(out_dir) / name).unlink(missing_ok=True)


def bench_noise(ratio: BenchStats) -> float:
    """The noise band of a head-to-head benchmark, as a relative time.

    ``ratio`` holds the paired ratios candidate/best; the band is ``NOISE_MADS``
    times their median absolute deviation. A gain is measured only when it is
    larger than this. One pair gives 0, hence `MIN_PAIRS`.
    """
    return NOISE_MADS * ratio.mad


def optimization_is_faster(ratio: BenchStats | None, first_round: bool = False) -> bool:
    """Decide whether to keep a rewrite based on its head-to-head timing.

    ``ratio`` holds the paired ratios candidate/best (see `_head_to_head_us`).
    With ``gain = 1 - ratio.median`` and ``noise = bench_noise(ratio)``:

    - a rewrite is kept only if it is measurably faster, ``gain > noise``;
    - in the first round the best so far is the translated solver, and the
      rewrite is kept unless it is measurably slower,
      ``gain >= -min(noise, FIRST_ROUND_MAX_LOSS)``. The cap bounds the loss
      that a noisy measurement can let through: a rewrite whose median is more
      than 2% slower is reverted however large the noise.

    Returns ``False`` if the measurement is missing or has fewer than
    ``MIN_PAIRS`` pairs, which is too few to estimate the noise.
    """
    if ratio is None or ratio.runs < MIN_PAIRS:
        return False
    gain, noise = 1.0 - ratio.median, bench_noise(ratio)
    return gain >= -min(noise, FIRST_ROUND_MAX_LOSS) if first_round else gain > noise


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
    verbose: bool = False,
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
    verbose:
        Print progress: one line per phase and per code optimization round, and
        a final summary of the timings and the optimizations the agent kept.
    """
    progress = Progress(verbose)
    try:
        return _generate_code(
            progress,
            verifier,
            next_state,
            out_dir,
            config,
            driver,
            state,
            params,
            name,
            code_optimization,
            profile,
            schedules,
            iterates,
            param_values,
            roll,
        )
    finally:
        progress.close()


def _generate_code(
    progress: Progress,
    verifier,
    next_state,
    out_dir,
    config,
    driver,
    state,
    params,
    name,
    code_optimization,
    profile,
    schedules,
    iterates,
    param_values,
    roll,
) -> CodegenResult:
    """`generate_code`'s body; ``progress`` is closed by the caller, also on error."""
    config = config or CodegenConfig()
    out_dir = Path(out_dir)
    progress.start("extract + scaffold")

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
    progress.done(True, "data, factorizations, matvecs")

    if driver is None:
        from fomtools.codegen.agent import get_default_driver

        driver = get_default_driver(config)
    elif getattr(driver, "config", False) is None:
        # A driver built without a config (e.g. CliAgentDriver(spec)) takes this
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
        check_timeout_s=config.check_timeout_s,
    )
    # `make verify` also holds the translated solver to the kernel rule that has
    # no baseline (it reads no matrix data: every product goes through a checked
    # fom_apply_*); the check below applies it again and decides.
    target.write_kernel_check_config(out_dir, ir, None)
    progress.start("translate → C (agent)")
    try:
        translate = driver.run(translation_task)
    finally:
        target.remove_kernel_check_config(out_dir)
        remove_manifest(out_dir)  # only this run's last step may leave a record
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
        timing.c_initial_us = _bench_mean_us(out_dir, timeout_s=config.check_timeout_s)
        timing.c_final_us = timing.c_initial_us  # updated after code optimization
        try:
            timing.profile = run_profile(out_dir, timeout_s=config.check_timeout_s)
        except Exception:
            timing.profile = []
    progress.translated(translate.success, translate.log, timing and timing.c_initial_us)

    # --- Code optimization phase ---------------------------------------------
    optimize = None
    optimization_diff = None
    optimizations_md = None
    kernel_checks = None
    rule_friction: list[dict] = []
    rounds: list[OptimizationRound] = []
    # Of the solver that ships, and only what verify itself printed: the agent's
    # own text (in `translate.log`) may quote numbers that no check produced.
    verify_output = translate.verify_output
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
            progress=progress,
            report_card=True,
        )
        optimize = opt.optimize
        optimization_diff = opt.diff
        optimizations_md = opt.optimizations_md
        kernel_checks = opt.kernel_checks
        rule_friction = opt.rule_friction
        if opt.kept:
            verify_output = opt.verify_output
            progress.summary(name, timing, opt.verify_output, opt.applied)
        rounds = opt.rounds
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
        # Last, once everything a library build compiles is in place: the record
        # that these sources are the verified ones (none after a compile-only check).
        write_manifest(out_dir, target, verify_output, timing)

    result = CodegenResult(
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
        rounds=rounds,
        verify=verify_numbers(verify_output) if translate.success else None,
    )
    if translate.success:
        # With the code optimization phase, OPTIMIZATIONS.md shows this card.
        result.report_svg = write_report_svg(out_dir, result_report_svg(result))
    return result


@dataclass
class OptimizationRound:
    """One round of the code optimization phase.

    ``us_before`` and ``us_after`` are the head-to-head ``make bench`` times of
    the best solver before the round and of the round's rewrite: the medians of
    the runs summarized in ``before`` and ``after`` (``None`` when not
    measured). ``gain`` is one minus the median of the paired ratios
    after/before, which equals ``(us_before - us_after) / us_before`` when the
    runs agree, and ``noise`` is the band the gain is compared with (see
    `optimization_is_faster`). ``stop_reason`` is set on the round that ended
    the phase.
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
    stop_reason: str | None = None
    noise: float | None = None
    before: BenchStats | None = None
    after: BenchStats | None = None

    def as_dict(self) -> dict:
        return asdict(self)


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
    #: What the agent says it applied, from the kept rounds' summaries.
    applied: list[str] = field(default_factory=list)
    #: ``make verify`` output of the shipped solver.
    verify_output: str | None = None


#: Wall-clock budget, in seconds, of the follow-up that asks for a missing round summary.
SUMMARY_FOLLOWUP_TIMEOUT_S = 300


def _summary_followup(
    driver: AgentDriver, task: AgentTask, before: dict[str, str], after: dict[str, str]
) -> AgentResult | None:
    """Ask the agent for the summary of a round that ended without one.

    A fresh run is shown the round's diff and asked to print only the summary;
    the caller restores the round's sources afterwards in case it edited them.
    Its verify command is a no-op, so the driver makes one attempt, within
    ``SUMMARY_FOLLOWUP_TIMEOUT_S``. Returns ``None`` if the follow-up fails: a
    missing summary never fails a round.
    """
    from fomtools.codegen.agent.prompts import code_optimization_summary_request

    diff = "".join(
        "".join(
            difflib.unified_diff(
                before[rel].splitlines(keepends=True),
                after[rel].splitlines(keepends=True),
                fromfile=f"{rel} (before this round)",
                tofile=f"{rel} (after this round)",
            )
        )
        for rel in after
    )
    followup = replace(
        task,
        instructions=code_optimization_summary_request(diff),
        # Nothing to verify: one attempt, and no "make it pass" re-prompt.
        verify_cmd=[sys.executable, "-c", ""],
        extra={**task.extra, "summary_only": True, "timeout_s": SUMMARY_FOLLOWUP_TIMEOUT_S},
    )
    try:
        return driver.run(followup)
    except Exception as exc:  # noqa: BLE001 - the summary is a report, not a gate
        warnings.warn(f"the follow-up asking for a round summary failed: {exc}", stacklevel=2)
        return None


#: Why a round ended the phase.
STOP_NO_CHANGE = "the round left the sources unchanged"
STOP_FAILED = "the round's sources failed verification or the kernel rules"
STOP_NOT_FASTER = "the round's sources were not measurably faster than the best so far"
STOP_SMALL_GAIN = "the round gained less than agent_optimize_min_gain or than the noise"
STOP_MAX_ROUNDS = "the last of agent_optimize_rounds rounds"
STOP_UNMEASURED = "no benchmark (profile off or no C compiler): one round only"


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
    progress: Progress | None = None,
    report_card: bool = False,
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

    1. still verify and pass the kernel rules (otherwise they are reverted to the
       best so far and the phase ends), and
    2. be faster than the best so far, head to head, by more than the measured
       noise (otherwise reverted, and the phase ends). The first round only has
       to be no slower than the translated solver, within the noise and at
       most 2% (`FIRST_ROUND_MAX_LOSS`), as the single-pass phase always has:
       a rewrite of the same speed is kept. Later
       rounds must gain, so that rounds inside the noise cannot add up to a
       slower solver. See `optimization_is_faster`.

    A kept round that gains less than ``agent_optimize_min_gain``, or no more
    than the noise, ends the phase,
    as does the last of ``agent_optimize_rounds``. Without a benchmark
    (``profile`` off or no compiler) a correct rewrite is kept unmeasured and only
    one round runs.

    ``timing``, if given, gets ``c_initial_us`` / ``c_final_us`` (the translated
    and the shipped solver, head to head, with their run-to-run spread in
    ``c_initial_stats`` / ``c_final_stats``) and the shipped solver's profile.
    ``translated_verify_output`` is the translated solver's verify output, used
    for the kernel-check table when no round is kept. ``progress`` reports each
    round as it ends. ``report_card`` puts the report card (``report.svg``, which
    the caller writes) at the top of ``OPTIMIZATIONS.md``.
    """
    from fomtools.codegen.agent.prompts import (
        code_optimization_continuation,
        code_optimization_instructions,
    )

    out_dir = Path(out_dir)
    editable = tuple(target.optimization_editable_paths())
    progress = progress or Progress(False)

    def read_editable() -> dict[str, str]:
        return {rel: (out_dir / rel).read_text() for rel in editable}

    def write_editable(sources: dict[str, str]) -> None:
        for rel, text in sources.items():
            (out_dir / rel).write_text(text)

    def restore_all_but_editable():
        kept = read_editable()
        target.scaffold(ir, out_dir)
        write_editable(kept)

    def ensure_summary(index, task, result, before, after):
        """A kept round that ended (often at its time budget) before the agent
        wrote its summary gets a short follow-up asking for it, which may not edit."""
        if SUMMARY_BEGIN in (result.agent_log or ""):
            return
        followup = _summary_followup(driver, task, before, after)
        write_editable(after)
        log = followup.agent_log if followup is not None else ""
        if log and SUMMARY_BEGIN in log:
            agent_logs.append(f"\n--- round {index} summary (follow-up) ---\n" + log)
            notes_by_round[:] = [(i, n) for i, n in notes_by_round if i != index]
            notes_by_round.append((index, extract_agent_notes(log)))

    measure = profile and find_compiler() is not None
    remove_manifest(out_dir)  # the sources are about to change; the caller records the result
    remove_report_svg(out_dir)  # and draws the card of the result
    translated = read_editable()  # known-good fallback
    best = translated
    best_us: float | None = None
    translated_us: float | None = None
    best_stats: BenchStats | None = None
    translated_stats: BenchStats | None = None
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

    # A round that raises or is interrupted (Ctrl-C) must not leave its unverified
    # candidate behind: the generated files are restored and, on every way out,
    # the best verified sources are written back.
    try:
        for index in range(1, n_rounds + 1):
            if rounds:  # the previous round ended without a break
                progress.round(rounds[-1])
            instructions = base_instructions
            if index > 1:
                last = rounds[-1]
                instructions += code_optimization_continuation(
                    index, n_rounds, best_us, translated_us, last.gain, best_profile
                )
            task = AgentTask(
                workdir=out_dir,
                solver_rel_path=target.solver_source_path(),
                verify_cmd=target.verify_command(),
                instructions=instructions,
                phase="code_optimization",
                bench_cmd=["make", "bench"],
                restore_hook=restore_all_but_editable,
                extra={"editable_paths": editable},
                check_timeout_s=config.check_timeout_s,
            )
            # While the agent works, `make verify` also applies the kernel rules, so it
            # sees a violation as it makes it; the gate below applies them again and
            # decides. The shipped directory carries no such config.
            target.write_kernel_check_config(out_dir, ir, translated)
            progress.start(f"optimize · round {index} (agent)")
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
            if not record.correct:
                write_editable(best)
                static_issues = issues
                logs.append(
                    "[optimization round not kept: its sources failed verification]"
                    + ("\n" + recheck.output if recheck and not recheck.success else "")
                    + ("\n" + "\n".join(issues) if issues else "")
                )
                record.stop_reason = STOP_FAILED
                break

            if not measure:
                ensure_summary(index, task, result, best, candidate)
                best, best_verify_output = candidate, recheck.output
                record.kept = True
                record.stop_reason = STOP_UNMEASURED
                break

            # Gate 2 (performance): head to head against the best so far, measured
            # now and alternately (timings are noisy), leaving the best written.
            after, before, ratio = _head_to_head_us(
                out_dir, write_editable, candidate, best, timeout_s=config.check_timeout_s
            )
            us_after = after.median if after else None
            us_before = before.median if before else None
            if index == 1:
                translated_us, translated_stats = us_before, before
            record.before, record.after = before, after
            record.us_before, record.us_after = us_before, us_after
            if ratio:
                record.gain, record.noise = 1.0 - ratio.median, bench_noise(ratio)
            if not optimization_is_faster(ratio, first_round=index == 1):
                logs.append(
                    f"[optimization round not kept: its build ({us_after} us) was not "
                    f"measurably faster than the best so far ({us_before} us): "
                    f"gain {record.gain}, noise {record.noise}, "
                    f"{ratio.runs if ratio else 0} of {BENCH_RUNS} pairs]"
                )
                record.stop_reason = STOP_NOT_FASTER
                break
            write_editable(candidate)
            ensure_summary(index, task, result, best, candidate)
            best, best_us, best_verify_output = candidate, us_after, recheck.output
            best_stats = after
            record.kept = True
            # Only the first round can be kept with a gain inside the noise.
            if record.gain < config.agent_optimize_min_gain or record.gain <= record.noise:
                record.stop_reason = STOP_SMALL_GAIN
                break
            if index == n_rounds:
                record.stop_reason = STOP_MAX_ROUNDS
                break
            try:
                best_profile = run_profile(out_dir, timeout_s=config.check_timeout_s)
            except Exception:
                best_profile = []
    except BaseException:
        target.scaffold(ir, out_dir)
        raise
    finally:
        write_editable(best)
        remove_manifest(out_dir)  # none that an agent wrote while it worked

    if rounds:
        progress.round(rounds[-1])
    kept = best != translated

    if timing is not None and measure:
        if translated_us is None:  # no round reached the benchmark
            translated_us = timing.c_initial_us or _bench_mean_us(
                out_dir, timeout_s=config.check_timeout_s
            )
        timing.c_initial_us = translated_us
        timing.c_final_us = best_us if kept else translated_us
        timing.c_initial_stats = translated_stats
        timing.c_final_stats = best_stats if kept else translated_stats
        try:
            timing.profile = run_profile(out_dir, timeout_s=config.check_timeout_s)
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
    kept_rounds = {r.index for r in rounds if r.kept}
    applied = [
        item
        for i, n in notes_by_round
        if i in kept_rounds
        for item in extract_applied_optimizations(n)
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
            final_stats = best_stats if kept else translated_stats
            timing_summary = (
                f"translated (emitted) {format_time(translated_us):>10} per solve"
                f"{'  ' + translated_stats.spread() if translated_stats else ''}\n"
                f"optimized            {format_time(final_us):>10} per solve"
                f"{'  ' + final_stats.spread() if final_stats else ''}\n"
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
            card=REPORT_SVG_NAME if report_card else None,
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
        applied=applied,
        verify_output=best_verify_output,
    )
