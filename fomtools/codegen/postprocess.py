"""Post-processing of generated solvers.

Two outputs after a run:

* A **production variant** of the solver with the profiling markers (and the
  ``fom_profile.h`` include) stripped out. The markers are already no-ops in the
  shipped build, so this is purely a cleaner, dependency-free source for
  embedding into other codebases — behaviour and performance are identical.
* An **optimization report** (``OPTIMIZATIONS.md``) recording the timing summary,
  the agent's own notes, and a unified diff of the solver from the translated
  version to the one after the code optimization phase — a durable record of exactly
  what changed.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from fomtools.codegen.agent.prompts import SUMMARY_BEGIN, SUMMARY_END

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
    """Unified diff of the solver source (translated -> optimized)."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{filename} (translated)",
        tofile=f"{filename} (optimized)",
    )
    return "".join(diff)


def _looks_like_prompt_echo(text: str) -> bool:
    """Whether ``text`` is something other than the agent's own prose.

    Two things get mistaken for a summary. The prompt, when a log ends up
    containing it — a subprocess exception that stringifies its argv is one way,
    the re-prompt on a failed attempt another; since the prompt is what *asks*
    for a summary, a naive marker search returns the request instead of the
    reply. And the harness's own bracketed diagnostics, which describe the agent
    rather than quoting it. Neither belongs in a report section headed "Agent
    notes". The phrases below appear only in
    :mod:`fomtools.codegen.agent.prompts` or in harness messages.
    """
    lowered = text.lower()
    tells = (
        # the prompt, echoed back
        "--append-system-prompt",
        "--allowedtools",
        "--permission-mode",
        "--output-format",
        "...your summary here...",
        "the markers are how your summary is lifted",
        "print a final summary",
        "emit them literally",
        # the harness talking about the agent, rather than the agent talking
        "[agent invocation raised:",
        "[fomtools:",
    )
    return any(tell in lowered for tell in tells)


def extract_agent_notes(log: str, max_chars: int = 4000) -> str:
    """Extract the agent's own optimization summary from its output.

    Pass the agent-only log (:attr:`~fomtools.codegen.agent.driver.AgentResult.agent_log`),
    not the combined one: the combined log interleaves build and verify output,
    so the fallback branches below would otherwise return compiler noise.

    Resolution order:

    1. the block the agent was asked to bracket with explicit markers, which is
       exact;
    2. a recognizable summary heading, as a fallback for a backend that ignored
       the markers;
    3. the tail of the log.

    Returns ``""`` rather than something misleading when the only candidate is
    our own prompt echoed back (see :func:`_looks_like_prompt_echo`) — an empty
    notes section is honest, whereas the prompt masquerading as a summary is not.
    """
    if not log:
        return ""

    begin_idx = log.rfind(SUMMARY_BEGIN)
    if begin_idx != -1:
        body = log[begin_idx + len(SUMMARY_BEGIN) :]
        end_idx = body.find(SUMMARY_END)
        if end_idx != -1:
            body = body[:end_idx]
        candidate = body.strip()
        if candidate and not _looks_like_prompt_echo(candidate):
            # Not truncated: the agent delimited this block itself, so its extent
            # is known rather than guessed, and cutting it would land mid-sentence
            # in the one section a reader most needs whole. The heuristic branches
            # below stay capped precisely because their extent is a guess.
            return candidate

    lower = log.lower()
    for marker in ("final summary", "## summary", "summary of optimizations"):
        idx = lower.rfind(marker)
        while idx != -1:
            candidate = log[idx:][:max_chars].strip()
            if not _looks_like_prompt_echo(candidate):
                return candidate
            idx = lower.rfind(marker, 0, idx)

    tail = log[-max_chars:].strip()
    return "" if _looks_like_prompt_echo(tail) else tail


def write_optimization_report(
    out_dir: str | Path,
    timing_summary: str,
    diff: str,
    agent_notes: str,
    optimize_succeeded: bool,
    caveats: list[str] | None = None,
) -> Path:
    """Write ``OPTIMIZATIONS.md`` recording timing, agent notes, and the diff.

    ``caveats`` records anything a reader must know to interpret the rest — most
    importantly that the agent was cut off before it could summarize. The timing
    and the diff stay trustworthy in that case (the solver on disk is re-verified
    and benchmarked either way), but an absent or partial notes section must not
    be mistaken for the agent having nothing to say.
    """
    out_dir = Path(out_dir)
    lines = ["# Optimization report", ""]
    if timing_summary:
        lines += ["## Timing", "", "```", timing_summary, "```", ""]
    status = (
        "kept (re-verified correct)"
        if optimize_succeeded
        else "not kept (reverted to the translated solver)"
    )
    lines += [f"Optimization outcome: **{status}**.", ""]
    if caveats:
        lines += ["## Caveats", ""]
        lines += [f"- {c}" for c in caveats]
        lines += [""]
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
