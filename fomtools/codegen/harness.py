"""Build / verify harness for emitted solvers.

Thin wrappers around the target's build commands plus a compiler probe, so the
rest of the pipeline (and tests) need not shell out directly.
"""

from __future__ import annotations

import os
import re
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from fomtools.codegen.manifest import check_manifest
from fomtools.codegen.proc import DEFAULT_CHECK_TIMEOUT_S, run_check
from fomtools.codegen.progress import format_time
from fomtools.codegen.targets import TargetDescriptor


def find_compiler() -> str | None:
    """The C compiler the generated Makefiles build with, or ``None``.

    That is ``$CC`` when it is set, else ``cc`` (``CC ?= cc``). No other
    compiler is looked for, since ``make`` would not use it.
    """
    words = os.environ.get("CC", "").split() or ["cc"]
    return shutil.which(words[0])


@dataclass
class VerifyResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def build_and_verify(workdir: str | Path, target: TargetDescriptor) -> VerifyResult:
    """Run the target's verify command in ``workdir``.

    A run still going after the ``check_timeout_s`` of the target's config is
    ended and reported as a failure (see `fomtools.codegen.proc.run_check`).
    """
    proc = run_check(target.verify_command(), workdir, target.config.check_timeout_s)
    return VerifyResult(
        success=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


@dataclass
class BenchResult:
    """Parsed result of ``make bench``."""

    solves: int
    total_us: float
    mean_us: float
    min_us: float
    raw: str = ""


@dataclass
class BenchStats:
    """Summary of repeated timings of one solver (per-solve times, in microseconds).

    ``mad`` is the median absolute deviation from the median: the robust
    run-to-run spread. With the usual five runs one or two slow runs (another
    process, a clock-speed change) are common; they inflate a standard deviation
    or a range but leave the MAD alone, and unlike an interquartile range the
    MAD of five values does not depend on an interpolation convention.

    The same shape summarizes the paired ratios candidate/best of a head-to-head
    benchmark (see `fomtools.codegen.generate.optimization_is_faster`).
    """

    runs: int
    median: float
    min: float
    max: float
    mad: float

    @classmethod
    def of(cls, values) -> "BenchStats | None":
        """Statistics of ``values``; failed runs (``None``) are left out.

        Returns ``None`` when no run succeeded. One run has ``mad == 0``.
        """
        values = [v for v in values if v]
        if not values:
            return None
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values)
        return cls(len(values), median, min(values), max(values), mad)

    def spread(self) -> str:
        """``± MAD (min to max, n runs)``, to print next to the median, in its unit."""
        runs = "1 run" if self.runs == 1 else f"{self.runs} runs"

        def fmt(value, unit=False):
            return format_time(value, like=self.median, digits=4, unit=unit)

        return f"± {fmt(self.mad)} ({fmt(self.min)} to {fmt(self.max, True)}, {runs})"


@dataclass
class ProfRegion:
    """One row of the profiling breakdown."""

    name: str
    total_us: float
    calls: int
    pct: float


@dataclass
class TimingReport:
    """Before/after timing summary for a code-generation run.

    All C times are per-solve means in microseconds; the Python time is the
    per-solve median of the reference oracle (the pre-codegen way to run the FOM).
    ``c_initial_stats`` and ``c_final_stats`` hold the run-to-run spread of the
    two C times when they come from a head-to-head benchmark of the code
    optimization phase (``None`` otherwise).
    """

    python_us: float | None = None
    c_initial_us: float | None = None
    c_final_us: float | None = None
    profile: list[ProfRegion] = field(default_factory=list)
    c_initial_stats: BenchStats | None = None
    c_final_stats: BenchStats | None = None

    @property
    def speedup_codegen(self) -> float | None:
        """Python (before codegen) vs. the initial generated C solver."""
        if self.python_us and self.c_initial_us:
            return self.python_us / self.c_initial_us
        return None

    @property
    def speedup_optimization(self) -> float | None:
        """Translated generated C vs. the C after the code optimization phase."""
        if self.c_initial_us and self.c_final_us:
            return self.c_initial_us / self.c_final_us
        return None

    @property
    def speedup_total(self) -> float | None:
        """Python vs. the final optimized C solver."""
        if self.python_us and self.c_final_us:
            return self.python_us / self.c_final_us
        return None

    def summary(self) -> str:
        """A human-readable speedup + profiling summary.

        The optimized solver and its speedups are listed only when the code
        optimization phase changed the time: with the phase off, or no round
        kept, the translated solver is the one that ships.
        """
        optimized = self.c_final_us is not None and self.c_final_us != self.c_initial_us
        lines = ["Timing summary (per solve):"]
        if self.python_us is not None:
            lines.append(f"  Python reference (before codegen): {format_time(self.python_us):>10}")
        for label, us, stats, shown in (
            ("Generated C (translated):", self.c_initial_us, self.c_initial_stats, True),
            ("Generated C (optimized): ", self.c_final_us, self.c_final_stats, optimized),
        ):
            if us is not None and shown:
                spread = f"  {stats.spread()}" if stats else ""
                lines.append(f"  {label}          {format_time(us):>10}{spread}")
        if self.speedup_codegen:
            lines.append(f"  Speedup from codegen (Py -> C):    {self.speedup_codegen:8.1f}x")
        if optimized and self.speedup_optimization:
            lines.append(f"  Speedup from code optimization:   {self.speedup_optimization:8.2f}x")
        if optimized and self.speedup_total:
            lines.append(f"  Total speedup (Py -> optimized C): {self.speedup_total:8.1f}x")
        if self.profile:
            lines.append("Profiling breakdown (hottest first):")
            for r in self.profile:
                lines.append(
                    f"  {r.name:<24} {format_time(r.total_us):>10}  {r.pct:6.2f}%  ({r.calls} calls)"
                )
        return "\n".join(lines)


_BENCH_RE = re.compile(
    r"BENCH\s+solves=(\d+)\s+total_us=([\d.eE+-]+)\s+mean_us=([\d.eE+-]+)\s+min_us=([\d.eE+-]+)"
)
_PROF_RE = re.compile(
    r"PROF\s+region=(\S+)\s+total_us=([\d.eE+-]+)\s+calls=(\d+)\s+pct=([\d.eE+-]+)"
)


def parse_bench(text: str) -> BenchResult:
    m = _BENCH_RE.search(text)
    if not m:
        raise ValueError(f"could not parse BENCH line from output:\n{text}")
    return BenchResult(
        solves=int(m.group(1)),
        total_us=float(m.group(2)),
        mean_us=float(m.group(3)),
        min_us=float(m.group(4)),
        raw=text,
    )


def parse_profile(text: str) -> list[ProfRegion]:
    regions = []
    for m in _PROF_RE.finditer(text):
        regions.append(
            ProfRegion(
                name=m.group(1),
                total_us=float(m.group(2)),
                calls=int(m.group(3)),
                pct=float(m.group(4)),
            )
        )
    return regions


def run_bench(
    workdir: str | Path,
    make_target: str = "bench",
    timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S,
) -> BenchResult:
    """Build and run the benchmark; return the parsed result.

    A run still going after ``timeout_s`` seconds is ended and raises like any
    other failure.
    """
    proc = run_check(["make", make_target], workdir, timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark build/run failed:\n{proc.stdout}\n{proc.stderr}")
    return parse_bench(proc.stdout + proc.stderr)


def run_profile(
    workdir: str | Path,
    make_target: str = "profile",
    timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S,
) -> list[ProfRegion]:
    """Build with profiling enabled, run, and return the region breakdown.

    A run still going after ``timeout_s`` seconds is ended and raises like any
    other failure.
    """
    proc = run_check(["make", make_target], workdir, timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"profile build/run failed:\n{proc.stdout}\n{proc.stderr}")
    return parse_profile(proc.stdout + proc.stderr)


def time_python_oracle(ir, param_values: dict, repeats: int = 50) -> float:
    """Median per-solve time (microseconds) of the Python reference oracle.

    This is the "before codegen" baseline: running the FOM in Python.
    """
    from fomtools.codegen.oracle import ReferenceOracle

    oracle = ReferenceOracle(ir)
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        oracle.solve(param_values=param_values)
        times.append((time.perf_counter() - t0) * 1.0e6)
    return statistics.median(times)


def representative_params(ir) -> dict:
    """A representative parameter set (finite-bound midpoints) for timing."""
    import numpy as np

    pvals = {}
    for p in ir.params:
        lo = np.where(np.isfinite(p.lb), p.lb, -1.0)
        hi = np.where(np.isfinite(p.ub), p.ub, 1.0)
        pvals[p.name] = 0.5 * (lo + hi)
    return pvals


def build_lib(
    workdir: str | Path,
    make_target: str = "lib",
    timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S,
    allow_unverified: bool = False,
) -> Path:
    """Build the shared library (``make lib``) and return its path.

    The sources are first checked against the directory's verification record
    (`fomtools.codegen.manifest.check_manifest`): a directory without one
    warns, and one whose sources changed since they were verified raises unless
    ``allow_unverified`` is set. A build still going after ``timeout_s`` seconds
    is ended and raises like any other failure.
    """
    workdir = Path(workdir)
    check_manifest(workdir, allow_unverified)
    proc = run_check(["make", make_target], workdir, timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"library build failed:\n{proc.stdout}\n{proc.stderr}")
    for name in ("libfomsolver.dylib", "libfomsolver.so"):
        cand = workdir / name
        if cand.exists():
            return cand
    raise RuntimeError("library build reported success but no library file was found")
