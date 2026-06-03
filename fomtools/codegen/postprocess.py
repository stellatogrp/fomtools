"""Post-processing of generated solvers.

Two outputs after a run:

* A **production variant** of the solver with the profiling markers (and the
  ``fom_profile.h`` include) stripped out. The markers are already no-ops in the
  shipped build, so this is purely a cleaner, dependency-free source for
  embedding into other codebases — behaviour and performance are identical.
* An **optimization report** (``OPTIMIZATIONS.md``) recording the timing summary,
  the agent's own notes, and a unified diff of the solver from the initial
  (phase A) to the optimized (phase B) version — a durable record of exactly
  what changed.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

_PROFILE_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"fom_profile\.h"\s*$')
# A line that is solely a FOM_PROF_BEGIN(...)/FOM_PROF_END(...) statement.
_PROFILE_CALL_RE = re.compile(r"^\s*FOM_PROF_(?:BEGIN|END)\s*\([^;]*\)\s*;\s*$")


def strip_profiling(source: str) -> str:
    """Return ``source`` with profiling markers and the profiler include removed.

    Only removes lines that are *solely* the profiler include or a
    ``FOM_PROF_BEGIN/END(...)`` statement, so it never touches real logic. The
    result is functionally identical (the markers are no-ops in production).
    """
    out = []
    for line in source.splitlines():
        if _PROFILE_INCLUDE_RE.match(line) or _PROFILE_CALL_RE.match(line):
            continue
        out.append(line)
    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    return text


def has_profiling_markers(source: str) -> bool:
    return bool(_PROFILE_INCLUDE_RE.search(source) or "FOM_PROF_BEGIN" in source)


def write_production_variant(out_dir: str | Path, solver_rel_path: str) -> Path:
    """Write a marker-free production copy of the solver under ``release/``.

    Returns the path to the emitted production source.
    """
    out_dir = Path(out_dir)
    src = (out_dir / solver_rel_path).read_text()
    clean = strip_profiling(src)

    release = out_dir / "release"
    release.mkdir(parents=True, exist_ok=True)
    name = Path(solver_rel_path).name
    dest = release / name
    dest.write_text(clean)
    (release / "README.md").write_text(
        "# Production solver\n\n"
        f"`{name}` is the marker-free production build of `{solver_rel_path}`: the\n"
        "`FOM_PROF_*` profiling markers and the `fom_profile.h` include have been\n"
        "stripped. Behaviour and performance are identical to the instrumented\n"
        "source (the markers are no-ops outside `make profile`); this copy is a\n"
        "clean, dependency-free drop-in for embedding.\n\n"
        "Build it with `make release` (from the parent directory), or compile\n"
        f"`release/{name}` together with `src/fom_data.c` against `include/`.\n"
    )
    return dest


def unified_solver_diff(before: str, after: str, filename: str) -> str:
    """Unified diff of the solver source (phase A -> optimized)."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{filename} (phase A)",
        tofile=f"{filename} (optimized)",
    )
    return "".join(diff)


def extract_agent_notes(log: str, max_chars: int = 4000) -> str:
    """Best-effort extraction of the agent's own optimization summary.

    Prefers an explicit 'FINAL SUMMARY' section; otherwise returns the tail of
    the log (where any concluding summary lives).
    """
    if not log:
        return ""
    lower = log.lower()
    for marker in ("final summary", "## summary", "summary of optimizations"):
        idx = lower.rfind(marker)
        if idx != -1:
            return log[idx:][:max_chars].strip()
    return log[-max_chars:].strip()


def write_optimization_report(
    out_dir: str | Path,
    timing_summary: str,
    diff: str,
    agent_notes: str,
    optimize_succeeded: bool,
) -> Path:
    """Write ``OPTIMIZATIONS.md`` recording timing, agent notes, and the diff."""
    out_dir = Path(out_dir)
    lines = ["# Optimization report", ""]
    if timing_summary:
        lines += ["## Timing", "", "```", timing_summary, "```", ""]
    status = (
        "kept (re-verified correct)" if optimize_succeeded else "not kept (reverted to phase A)"
    )
    lines += [f"Optimization outcome: **{status}**.", ""]
    lines += ["## What the agent changed", ""]
    if diff.strip():
        lines += ["```diff", diff.rstrip(), "```", ""]
    else:
        lines += ["_No changes to the solver source (it was already optimal or unchanged)._", ""]
    if agent_notes.strip():
        lines += ["## Agent notes", "", "```", agent_notes.strip(), "```", ""]
    path = out_dir / "OPTIMIZATIONS.md"
    path.write_text("\n".join(lines))
    return path
