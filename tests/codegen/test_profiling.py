"""Tests for timing, profiling, and the performance-optimization phase.

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


# --- phase-B loop machinery (stub: optimize is a no-op, harness is real) --
def test_phase_b_loop_records_timing_and_reverifies(tmp_path, requires_compiler):
    case = build_ista(max_iters=200)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="ista",
        run_phase_b=True,
        profile=True,
    )
    assert result.translate.success
    assert result.optimize is not None
    assert result.optimize.success  # re-verify correctness gate passed
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


def test_timing_report_speedup_math():
    from fomtools.codegen.harness import ProfRegion, TimingReport

    t = TimingReport(python_us=1000.0, c_initial_us=10.0, c_final_us=5.0)
    assert t.speedup_codegen == pytest.approx(100.0)
    assert t.speedup_optimization == pytest.approx(2.0)
    assert t.speedup_total == pytest.approx(200.0)
    t.profile = [ProfRegion("matvec", 80.0, 100, 80.0), ProfRegion("relu", 20.0, 100, 20.0)]
    assert "matvec" in t.summary()
