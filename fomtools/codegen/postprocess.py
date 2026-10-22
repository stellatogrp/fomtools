"""Post-processing of generated solvers.

Two outputs after a run:

* A **production variant** of the solver with the profiling markers (and the
  ``fom_profile.h`` include) stripped out. The markers are already no-ops in the
  shipped build, so this is purely a cleaner, dependency-free source for
  embedding into other codebases — behaviour and performance are identical.
* An **optimization report** (``OPTIMIZATIONS.md``) recording the timing summary,
  the agent's own notes, and a unified diff of the solver from the translated
  version to the one after the code optimization phase — a durable record of exactly
  what changed — and, where the matrix-vector products are editable, how each
  shipped ``fom_apply_<name>`` was checked against its matrix.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from fomtools.codegen.agent.prompts import BLOCKED_HEADING, SUMMARY_BEGIN, SUMMARY_END

# One line per matrix from verify's direct check of fom_apply_<name>.
_OPSCHECK_RE = re.compile(
    r"^OPSCHECK name=(?P<name>\S+) mode=(?P<mode>\w+) probes=(?P<probes>\d+) "
    r"max_err=(?P<max_err>\S+)\s*$",
    re.MULTILINE,
)

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


def write_production_variant(
    out_dir: str | Path, solver_rel_path: str, extra_rel_paths: tuple[str, ...] = ()
) -> Path:
    """Write a marker-free production copy of the solver under ``release/``.

    ``extra_rel_paths`` are other sources the code optimization phase may have
    rewritten (the linear solves, the matrix-vector products); each gets a
    marker-free copy beside the solver, which ``make release`` compiles.

    Returns the path to the emitted production source.
    """
    out_dir = Path(out_dir)
    release = out_dir / "release"
    release.mkdir(parents=True, exist_ok=True)
    for rel in extra_rel_paths:
        (release / Path(rel).name).write_text(strip_profiling((out_dir / rel).read_text()))

    src = (out_dir / solver_rel_path).read_text()
    clean = strip_profiling(src)
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
    """Unified diff of one source file (translated -> optimized)."""
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
    `fomtools.codegen.agent.prompts` or in harness messages.
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

    Pass the agent-only log (`agent_log`),
    not the combined one: the combined log interleaves build and verify output,
    so the fallback branches below would otherwise return compiler noise.

    Resolution order:

    1. the block the agent was asked to bracket with explicit markers, which is
       exact;
    2. a recognizable summary heading, as a fallback for a backend that ignored
       the markers;
    3. the tail of the log.

    Returns ``""`` rather than something misleading when the only candidate is
    our own prompt echoed back (see `_looks_like_prompt_echo`) — an empty
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


def parse_ops_checks(verify_output: str) -> list[dict]:
    """The per-matrix ``OPSCHECK`` lines of a ``make verify`` run, in order."""
    return [
        {
            "name": m["name"],
            "mode": m["mode"],
            "probes": int(m["probes"]),
            "max_err": float(m["max_err"]),
        }
        for m in _OPSCHECK_RE.finditer(verify_output)
    ]


def _kernel_check_lines(
    kernel_checks: list[dict],
    rule_violations: list[str],
    rewrite_collected: bool,
    optimize_succeeded: bool,
) -> list[str]:
    lines = [
        "## Kernel checks",
        "",
        "Each `fom_apply_<name>` in the shipped `fom_ops.c`, checked by `make verify` "
        "against the package's reference over the same `fom_data.c` matrix. *basis* "
        "probes every unit vector, which recovers the routine's matrix column by "
        "column (an exact check), plus random vectors; *random* uses random vectors "
        "only (the matrix is over the probe budget). A `fused_<block>` row is a "
        "`fom_fused_<block>`, checked against the sum of its members: *basis* there "
        "probes every unit vector of each source with every schedule value at 0 and "
        "with each alone at 1, which recovers every member exactly.",
        "",
    ]
    if kernel_checks:
        lines += [
            "| matrix | shape | nnz | rewritten | check | probes | max error |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in kernel_checks:
            shape = "x".join(str(d) for d in c["shape"])
            lines.append(
                f"| `{c['name']}` | {shape} | {c['nnz']} | "
                f"{'yes' if c['rewritten'] else 'no'} | {c['mode']} | {c['probes']} | "
                f"{c['max_err']:.3e} |"
            )
    else:
        lines.append("_Not available: the final verify did not report the checks._")
    lines += ["", "### Rule violations", ""]
    if not rewrite_collected:
        lines.append(
            "_Not checked: the code optimization phase returned no changed, verified "
            "rewrite, so there was nothing to collect._"
        )
    elif rule_violations:
        lines += ["The collected rewrite broke these rules and was discarded:", ""]
        lines += [f"- {v}" for v in rule_violations]
    elif optimize_succeeded:
        lines.append("_None: the collected rewrite passed the rules and was kept._")
    else:
        lines.append(
            "_None: the collected rewrite passed the rules but was not kept "
            "(see the outcome above)._"
        )
    return lines + [""]


def _round_lines(rounds: list[dict]) -> list[str]:
    """The report's per-round table of the code optimization phase."""

    def us(value) -> str:
        return f"{value:.3f}" if value is not None else "-"

    lines = [
        "## Rounds",
        "",
        "| round | before (us) | after (us) | gain | kept | agent (s) | outcome |",
        "|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for r in rounds:
        gain = f"{100 * r['gain']:.1f}%" if r.get("gain") is not None else "-"
        outcome = r.get("stop_reason") or "continued"
        if r.get("truncated"):
            outcome += " (agent stopped at its budget)"
        lines.append(
            f"| {r['index']} | {us(r.get('us_before'))} | {us(r.get('us_after'))} | {gain} "
            f"| {'yes' if r.get('kept') else 'no'} | {r.get('agent_wall_s', 0):.0f} | {outcome} |"
        )
    return lines + [""]


def extract_rule_friction(notes: str) -> list[dict]:
    """The optimizations the agent reports a rule blocked, from its summary.

    Reads the lines under the ``Blocked by rules:`` heading the optimization
    prompt asks for, each ``- <idea> | <rule> | <expected gain>``, until the
    first blank or non-list line. ``- none`` (or no section) gives ``[]``. A line
    without separators is kept whole as the idea.
    """
    lines = notes.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().strip("*#").strip() == BLOCKED_HEADING),
        None,
    )
    if start is None:
        return []
    found = []
    for line in lines[start + 1 :]:
        item = line.strip()
        if not item.startswith(("-", "*")):
            if found or item:
                break
            continue
        item = item[1:].strip()
        if not item or item.lower().rstrip(".") == "none":
            continue
        parts = [part.strip() for part in item.split("|")]
        found.append(
            dict(
                idea=parts[0],
                rule=parts[1] if len(parts) > 1 else "",
                expected_gain=parts[2] if len(parts) > 2 else "",
            )
        )
    return found


def write_optimization_report(
    out_dir: str | Path,
    timing_summary: str,
    diff: str,
    agent_notes: str,
    optimize_succeeded: bool,
    caveats: list[str] | None = None,
    kernel_checks: list[dict] | None = None,
    rule_violations: list[str] | None = None,
    rewrite_collected: bool = True,
    rounds: list[dict] | None = None,
    rule_friction: list[dict] | None = None,
) -> Path:
    """Write ``OPTIMIZATIONS.md`` recording timing, agent notes, and the diff.

    ``caveats`` records anything a reader must know to interpret the rest — most
    importantly that the agent was cut off before it could summarize. The timing
    and the diff stay trustworthy in that case (the solver on disk is re-verified
    and benchmarked either way), but an absent or partial notes section must not
    be mistaken for the agent having nothing to say.

    ``kernel_checks`` (one dict per matrix: ``name, shape, nnz, rewritten, mode,
    probes, max_err``) adds the per-routine check table, with the
    ``rule_violations`` found when the rewrite was collected; ``None`` (a target
    whose products are not editable) leaves the section out.
    ``rewrite_collected`` is False when the phase returned no changed, verified
    rewrite, so the rules never ran on one. ``rounds`` (one dict per round, see
    `fomtools.codegen.generate.OptimizationRound`) adds a per-round table when
    the phase ran more than one. ``rule_friction`` (see `extract_rule_friction`,
    with a ``round``) adds the table of optimizations the agent says a rule
    blocked.
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
    if rounds and len(rounds) > 1:
        lines += _round_lines(rounds)
    if caveats:
        lines += ["## Caveats", ""]
        lines += [f"- {c}" for c in caveats]
        lines += [""]
    if kernel_checks is not None:
        lines += _kernel_check_lines(
            kernel_checks, rule_violations or [], rewrite_collected, optimize_succeeded
        )
    if rule_friction:
        lines += [
            "## Blocked by rules",
            "",
            "Optimizations the agent considered but did not make because a rule forbade "
            "them, in its own words.",
            "",
            "| round | idea | rule | expected gain |",
            "|---|---|---|---|",
        ]
        lines += [
            "| {} | {} | {} | {} |".format(
                f.get("round", ""),
                *(str(f.get(k, "")).replace("|", "/") for k in ("idea", "rule", "expected_gain")),
            )
            for f in rule_friction
        ]
        lines += [""]
    lines += ["## What the agent changed", ""]
    if diff.strip():
        lines += ["```diff", diff.rstrip(), "```", ""]
    else:
        lines += ["_No changes to the generated sources (already optimal or unchanged)._", ""]
    if agent_notes.strip():
        lines += ["## Agent notes", "", "```", agent_notes.strip(), "```", ""]
    path = out_dir / "OPTIMIZATIONS.md"
    path.write_text("\n".join(lines))
    return path
