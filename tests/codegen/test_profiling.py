"""Tests for timing, profiling, and the code optimization phase.

The "before vs. after codegen" timing compares the Python reference oracle
(how the FOM ran before codegen) against the generated C solver.
"""

from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.harness import (
    parse_bench,
    parse_profile,
    representative_params,
    run_bench,
    run_profile,
    time_python_oracle,
)
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"


# --- pure parsing (no compiler) ------------------------------------------
def test_parse_bench():
    line = "BENCH solves=5000 total_us=12076.0 mean_us=2.4152 min_us=0.5\n"
    r = parse_bench(line)
    assert r.solves == 5000
    assert r.mean_us == pytest.approx(2.4152)
    assert r.min_us == pytest.approx(0.5)


def test_parse_profile():
    text = (
        "PROF region=soft_threshold total_us=20246.0 calls=1000000 pct=51.21\n"
        "PROF region=state_update total_us=19288.0 calls=1000000 pct=48.79\n"
    )
    regions = parse_profile(text)
    assert [r.name for r in regions] == ["soft_threshold", "state_update"]
    assert regions[0].pct == pytest.approx(51.21)
    assert sum(r.pct for r in regions) == pytest.approx(100.0, abs=0.1)


def test_parse_bench_missing_raises():
    with pytest.raises(ValueError):
        parse_bench("no bench line here")


# --- compiled timing/profiling -------------------------------------------
def _generate(tmp_path, max_iters=200):
    case = build_ista(max_iters=max_iters)
    return generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
    ), case


def test_make_bench_runs(tmp_path, requires_compiler):
    result, _ = _generate(tmp_path)
    assert result.translate.success
    bench = run_bench(tmp_path)
    assert bench.solves > 0
    assert bench.mean_us > 0.0


def test_make_profile_reports_regions(tmp_path, requires_compiler):
    result, _ = _generate(tmp_path)
    regions = run_profile(tmp_path)
    names = {r.name for r in regions}
    assert {"soft_threshold", "state_update"} <= names
    assert sum(r.pct for r in regions) == pytest.approx(100.0, abs=1.0)
    assert all(r.calls > 0 for r in regions)


def test_timing_before_and_after_codegen(tmp_path, requires_compiler):
    """Measure execution time before (Python oracle) and after (generated C)."""
    result, _ = _generate(tmp_path)
    ir = result.ir
    sample = representative_params(ir)

    python_us = time_python_oracle(ir, sample, repeats=30)
    c_us = run_bench(tmp_path).mean_us

    assert python_us > 0.0 and c_us > 0.0
    # The compiled C solver must be substantially faster than the Python oracle.
    assert c_us < python_us
    speedup = python_us / c_us
    print(f"before(py)={python_us:.2f}us after(C)={c_us:.4f}us speedup={speedup:.0f}x")
    assert speedup > 5.0  # very conservative; typically hundreds-to-thousands x


# --- code optimization loop machinery (stub: optimize is a no-op, harness is real)
def test_code_optimization_loop_records_timing_and_reverifies(tmp_path, requires_compiler):
    case = build_ista(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
        code_optimization=True,
        profile=True,
    )
    assert result.translate.success
    assert result.success  # a verified solver was shipped regardless of optimization
    assert result.optimize is not None
    # The stub makes no source change, so no optimization is "kept".
    assert not result.optimization_kept
    t = result.timing
    assert t is not None
    assert t.python_us and t.python_us > 0
    assert t.c_initial_us and t.c_initial_us > 0
    assert t.c_final_us and t.c_final_us > 0
    assert t.speedup_codegen and t.speedup_codegen > 1.0
    assert t.profile  # profiling breakdown captured
    # The summary string is well-formed.
    s = t.summary()
    assert "Speedup from codegen" in s
    assert "Profiling breakdown" in s


def test_optimization_perf_gate_logic():
    """The keep/revert decision: keep the optimized solver only if not slower (within noise)."""
    from fomtools.codegen.generate import optimization_is_faster
    from fomtools.codegen.harness import BenchStats

    def keep(after, before, first_round=True):
        """Five equal runs per side (no noise), or the runs given as lists."""
        after = after if isinstance(after, list) else [after] * 5
        before = before if isinstance(before, list) else [before] * 5
        ratio = BenchStats.of(a / b for a, b in zip(after, before))
        return optimization_is_faster(ratio, first_round=first_round)

    # Without noise, the first round keeps a rewrite that is not slower ...
    assert keep(1.0, 2.0)  # clearly faster -> keep
    assert keep(2.0, 2.0)  # equal -> keep
    assert not keep(3.0, 2.0)  # slower -> revert to the translated solver
    assert not keep(2.1, 2.0)
    # ... and a measured 1% loss is a loss (the fixed 2% tolerance used to keep it).
    assert not keep(2.02, 2.0)
    # Later rounds must gain.
    assert keep(1.0, 2.0, first_round=False)
    assert keep(1.99, 2.0, first_round=False)
    assert not keep(2.0, 2.0, first_round=False)  # equal is not a gain
    assert not keep(2.1, 2.0, first_round=False)

    # With noise: the paired ratios below have median 0.99 and MAD 0.02, so the
    # 1% gain is inside the noise band of 2 * 0.02.
    before = [100.0] * 5
    inside = [95.0, 97.0, 99.0, 101.0, 103.0]
    assert not keep(inside, before, first_round=False)  # not measurably faster
    assert keep(inside, before)  # round 1: not measurably slower -> keep
    slower = [97.0, 99.0, 101.0, 103.0, 105.0]  # median 1.01, same spread
    assert keep(slower, before)  # round 1: inside the noise -> keep
    assert not keep(slower, before, first_round=False)
    # Round 1 tolerates at most a 2% loss, however large the measured noise: the
    # ratios below have median 1.05 and MAD 0.04, so the noise band alone (0.08)
    # would keep this 5% regression.
    noisy_loss = [97.0, 101.0, 105.0, 109.0, 113.0]
    assert not keep(noisy_loss, before)
    assert not keep([95.0, 99.0, 103.0, 107.0, 111.0], before)  # 3% loss, same spread
    assert keep([93.0, 97.0, 101.0, 105.0, 109.0], before)  # 1% loss: under the cap -> keep
    # An equal-speed rewrite within small noise is still kept (MAD 0.002, band 0.004).
    assert keep([99.6, 99.8, 100.0, 100.2, 100.4], before)
    assert keep([99.9, 100.1, 100.3, 100.5, 100.7], before)  # 0.3% loss, inside the band
    assert not keep([100.2, 100.4, 100.6, 100.8, 101.0], before)  # 0.6% loss, outside it
    # The same spread around a 10% gain or loss is outside the band.
    assert keep([86.0, 88.0, 90.0, 92.0, 94.0], before, first_round=False)
    assert not keep([106.0, 108.0, 110.0, 112.0, 114.0], before)
    # One slow run of the best solver (or of the rewrite) does not move the decision.
    assert not keep([110.0] * 5, [100.0, 100.0, 300.0, 100.0, 100.0])
    assert keep([90.0, 90.0, 270.0, 90.0, 90.0], before, first_round=False)
    # A drift that hits both sides of each pair alike cancels in the ratios.
    drift = [1.0, 1.1, 1.2, 1.3, 1.4]
    assert keep([98.0 * d for d in drift], [100.0 * d for d in drift], first_round=False)

    # Four pairs are a measurement, fewer are not (one pair has no spread at all).
    assert keep([50.0] * 4, [100.0] * 4) and keep([50.0] * 4, [100.0] * 4, first_round=False)
    for n in (1, 2, 3):
        assert not keep([50.0] * n, [100.0] * n)
        assert not keep([50.0] * n, [100.0] * n, first_round=False)
    assert not optimization_is_faster(None)  # missing measurement -> revert
    assert not optimization_is_faster(None, first_round=True)


def test_bench_stats():
    from fomtools.codegen.harness import BenchStats

    s = BenchStats.of([3.0, 1.0, 2.0, 10.0, 4.0])
    assert (s.runs, s.median, s.min, s.max) == (5, 3.0, 1.0, 10.0)
    assert s.mad == 1.0  # deviations 0, 2, 1, 7, 1: the slow run does not move it
    assert s.spread() == "± 1.000 (1.000 to 10.000, 5 runs)"
    even = BenchStats.of([1.0, 2.0, 4.0, 8.0])
    assert even.median == 3.0 and even.mad == 1.5  # deviations 2, 1, 1, 5
    # Failed runs (None) are ignored.
    failed = BenchStats.of([None, 2.0, None, 4.0])
    assert (failed.runs, failed.median, failed.min, failed.max, failed.mad) == (2, 3.0, 2, 4, 1)
    one = BenchStats.of(iter([5.0]))
    assert (one.runs, one.median, one.min, one.max, one.mad) == (1, 5.0, 5.0, 5.0, 0.0)
    assert one.spread() == "± 0.000 (5.000 to 5.000, 1 run)"
    assert BenchStats.of([]) is None and BenchStats.of([None, None]) is None


def test_head_to_head_statistics_pair_the_runs(tmp_path, monkeypatch):
    """Run i of each side forms a pair; a pair with a failed run is left out."""
    from fomtools.codegen import generate as gen

    def build(out_dir, name, timeout_s):
        return out_dir / name

    runs = {
        ".bench_candidate": iter([90.0, None, 99.0, 110.0, 96.0]),
        ".bench_best": iter([100.0, 100.0, 110.0, 100.0, None]),
    }
    monkeypatch.setattr(gen, "_build_bench", build)
    monkeypatch.setattr(gen, "_run_bench_bin", lambda path, out_dir, t: next(runs[path.name]))
    after, before, ratio = gen._head_to_head_us(tmp_path, lambda sources: None, {}, {})
    assert (after.runs, after.median) == (4, 97.5)
    assert (before.runs, before.median) == (4, 100.0)
    assert ratio.runs == 3 and ratio.median == pytest.approx(0.9)  # 0.9, 0.9, 1.1
    assert gen.bench_noise(ratio) == pytest.approx(0.0)
    # Three pairs are too few to estimate the noise: the 10% gain is not kept.
    assert not gen.optimization_is_faster(ratio) and not gen.optimization_is_faster(ratio, True)

    # No run of the candidate succeeds: no ratio, so the round is not kept.
    runs[".bench_candidate"], runs[".bench_best"] = iter([None] * 5), iter([100.0] * 5)
    after, before, ratio = gen._head_to_head_us(tmp_path, lambda sources: None, {}, {})
    assert after is None and before.runs == 5 and ratio is None
    assert not gen.optimization_is_faster(ratio, first_round=True)

    # A build that failed (no binary): that side has no runs.
    monkeypatch.setattr(gen, "_build_bench", lambda out_dir, name, t: None)
    assert gen._head_to_head_us(tmp_path, lambda sources: None, {}, {}) == (None, None, None)


def test_timing_report_speedup_math():
    from fomtools.codegen.harness import ProfRegion, TimingReport

    t = TimingReport(python_us=1000.0, c_initial_us=10.0, c_final_us=5.0)
    assert t.speedup_codegen == pytest.approx(100.0)
    assert t.speedup_optimization == pytest.approx(2.0)
    assert t.speedup_total == pytest.approx(200.0)
    t.profile = [ProfRegion("matvec", 80.0, 100, 80.0), ProfRegion("relu", 20.0, 100, 20.0)]
    assert "matvec" in t.summary()


# --- benchmark length and the head-to-head gate (#168) -------------------
def test_bench_times_a_fast_solver_for_its_full_budget(tmp_path, requires_compiler):
    # A ~1 us solve used to stop at 5000 solves, about 5 ms of timed work.
    _generate(tmp_path)
    bench = run_bench(tmp_path)
    assert bench.solves > 5000
    assert bench.total_us >= 0.9e5  # FOM_BENCH_BUDGET_NS = 100 ms


def test_head_to_head_alternates_two_builds_and_leaves_best(
    tmp_path, monkeypatch, requires_compiler
):
    from fomtools.codegen import generate as gen

    _generate(tmp_path)
    solver = tmp_path / "src" / "fom_solver.c"
    best = {"src/fom_solver.c": solver.read_text()}
    # Same code plus a comment: a separate build of the same speed.
    candidate = {"src/fom_solver.c": best["src/fom_solver.c"] + "\n/* candidate */\n"}

    def write(sources):
        for rel, text in sources.items():
            (tmp_path / rel).write_text(text)

    order = []
    real_run = gen._run_bench_bin

    def recording_run(path, out_dir, timeout_s):
        order.append(path.name)
        return real_run(path, out_dir, timeout_s)

    monkeypatch.setattr(gen, "_run_bench_bin", recording_run)
    after, before, ratio = gen._head_to_head_us(tmp_path, write, candidate, best, runs=3)
    assert after.median > 0 and before.median > 0
    assert after.runs == before.runs == ratio.runs == 3
    assert after.min <= after.median <= after.max and after.mad >= 0
    assert ratio.median == pytest.approx(1.0, rel=0.5)  # the same code, built twice
    c, b = ".bench_candidate", ".bench_best"
    assert order == [c, b, b, c, c, b]  # alternating, so drifts hit both alike
    assert solver.read_text() == best["src/fom_solver.c"]  # best left in place
    assert not list(tmp_path.glob(".bench_*"))  # copies removed


# --- a check that never returns (no compiler) ------------------------------
def _hanging_bin(tmp_path):
    path = tmp_path / "bench_bin"
    path.write_text("#!/bin/sh\necho partial\nsleep 60\n")
    path.chmod(0o755)
    return path


def test_a_hung_bench_binary_counts_as_a_failed_run(tmp_path):
    import time

    from fomtools.codegen import generate as gen

    start = time.monotonic()
    assert gen._run_bench_bin(_hanging_bin(tmp_path), tmp_path, timeout_s=0.3) is None
    assert time.monotonic() - start < 10


# A hand-written directory: `build_lib` says it has no verification record.
@pytest.mark.filterwarnings("ignore:.*has no verification record:UserWarning")
def test_hung_make_targets_fail_instead_of_hanging(tmp_path):
    import shutil
    import time

    from fomtools.codegen import generate as gen
    from fomtools.codegen.config import CodegenConfig
    from fomtools.codegen.harness import build_and_verify, build_lib
    from fomtools.codegen.targets import get_target

    if shutil.which("make") is None:
        pytest.skip("no make available")
    _hanging_bin(tmp_path)
    (tmp_path / "Makefile").write_text("verify bench profile lib:\n\t./bench_bin\n")
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="benchmark build/run failed(.|\n)*exceeded its 0.3s"):
        run_bench(tmp_path, timeout_s=0.3)
    with pytest.raises(RuntimeError, match="profile build/run failed(.|\n)*partial"):
        run_profile(tmp_path, timeout_s=0.3)
    with pytest.raises(RuntimeError, match="library build failed"):
        build_lib(tmp_path, timeout_s=0.3)
    assert gen._bench_mean_us(tmp_path, timeout_s=0.3) is None
    target = get_target(CodegenConfig().model_copy(update={"check_timeout_s": 0.3}))
    outcome = build_and_verify(tmp_path, target)
    assert not outcome.success and "exceeded its 0.3s limit" in outcome.output
    assert time.monotonic() - start < 20
