"""Build / verify harness for emitted solvers.

Thin wrappers around the target's build commands plus a compiler probe, so the
rest of the pipeline (and tests) need not shell out directly.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from fomtools.codegen.targets import TargetDescriptor


def find_compiler() -> str | None:
    """Return an available C compiler executable, or ``None``."""
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


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
    """Run the target's verify command in ``workdir``."""
    proc = subprocess.run(
        target.verify_command(),
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
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
    """

    python_us: float | None = None
    c_initial_us: float | None = None
    c_final_us: float | None = None
    profile: list[ProfRegion] = field(default_factory=list)

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
        """A human-readable speedup + profiling summary."""
        lines = ["Timing summary (per solve):"]
        if self.python_us is not None:
            lines.append(f"  Python reference (before codegen): {self.python_us:10.3f} us")
        if self.c_initial_us is not None:
            lines.append(f"  Generated C (translated):         {self.c_initial_us:10.3f} us")
        if self.c_final_us is not None:
            lines.append(f"  Generated C (optimized):          {self.c_final_us:10.3f} us")
        if self.speedup_codegen:
            lines.append(f"  Speedup from codegen (Py -> C):    {self.speedup_codegen:8.1f}x")
        if self.speedup_optimization:
            lines.append(f"  Speedup from code optimization:   {self.speedup_optimization:8.2f}x")
        if self.speedup_total:
            lines.append(f"  Total speedup (Py -> optimized C): {self.speedup_total:8.1f}x")
        if self.profile:
            lines.append("Profiling breakdown (hottest first):")
            for r in self.profile:
                lines.append(
                    f"  {r.name:<24} {r.total_us:12.3f} us  {r.pct:6.2f}%  ({r.calls} calls)"
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


def run_bench(workdir: str | Path, make_target: str = "bench") -> BenchResult:
    """Build and run the benchmark; return the parsed result."""
    proc = subprocess.run(["make", make_target], cwd=str(workdir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark build/run failed:\n{proc.stdout}\n{proc.stderr}")
    return parse_bench(proc.stdout + proc.stderr)


def run_profile(workdir: str | Path, make_target: str = "profile") -> list[ProfRegion]:
    """Build with profiling enabled, run, and return the region breakdown."""
    proc = subprocess.run(["make", make_target], cwd=str(workdir), capture_output=True, text=True)
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


def build_lib(workdir: str | Path, make_target: str = "lib") -> Path:
    """Build the shared library (``make lib``) and return its path."""
    workdir = Path(workdir)
    proc = subprocess.run(
        ["make", make_target],
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"library build failed:\n{proc.stdout}\n{proc.stderr}")
    for name in ("libfomsolver.dylib", "libfomsolver.so"):
        cand = workdir / name
        if cand.exists():
            return cand
    raise RuntimeError("library build reported success but no library file was found")
