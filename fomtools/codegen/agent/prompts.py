"""Prompt construction for the coding-agent phases (translation and code optimization)."""

from __future__ import annotations

from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets import TargetDescriptor

# Delimiters the code optimization agent wraps its summary in, so
# fomtools.codegen.postprocess can lift it out exactly rather than guessing at a
# phrase. Imported there; keep the two in step.
_SUMMARY_BEGIN = "FOMTOOLS-SUMMARY-BEGIN"
_SUMMARY_END = "FOMTOOLS-SUMMARY-END"
SUMMARY_BEGIN = _SUMMARY_BEGIN
SUMMARY_END = _SUMMARY_END
#: Heading of the summary section where the agent lists what the rules blocked.
BLOCKED_HEADING = "Blocked by rules:"
#: Heading of the summary section where the agent lists what it changed.
APPLIED_HEADING = "Optimizations applied:"


def _objective_block(ir: FomIR, objective_src: str) -> str:
    """The objective-termination contract, shared by both phases."""
    if not ir.termination.uses_objective:
        return ""
    where = (
        "once, after the single pass"
        if ir.unrolled
        else "at the end of every iteration `it` with `(check > 0 && (it + 1) % check == 0) "
        "|| it + 1 == max_iter`, where `check = work->settings.check_termination` and "
        "`max_iter = work->settings.max_iter`, after computing `new_state` and before it "
        "replaces `state`"
    )
    return f"""
This solver STOPS ON THE VERIFICATION OBJECTIVE (termination kind `objective`).
`{objective_src}` provides `fom_objective(...)` (declared in `include/fom_ops.h`), which
computes the certified metric r^k. Call it {where}; report its value in
`solution.residual` and return `FOM_SOLVED` as soon as it is
`<= work->settings.tol`. Never compute the metric yourself and never stop on any
other test. `tol`, `max_iter` and `check_termination` are RUNTIME settings: set
their defaults (`FOM_TOL`, `FOM_MAX_ITER`, `FOM_CHECK_TERMINATION`) in
`fom_set_default_settings` and read them from `work->settings`. SPEC.md gives the
exact protocol; `make verify` checks the stopping iteration, objective value and
status against the oracle at several runtime settings.
"""


def translation_instructions(ir: FomIR, target: TargetDescriptor) -> str:
    """Instructions for the translation phase: a faithful translation of the spec."""
    verify = " ".join(target.verify_command())
    src = target.solver_source_path()
    ops_src = target.ops_source_path()
    linsolve_src = target.linsolve_source_path()
    notes = target.agent_notes()
    notes_block = f"\n{notes}\n" if notes else ""
    ops_rule = f"""
  The one exception: never read the data of a matrix you multiply by (any
  matrix with a `fom_apply_<name>` in `{ops_src}`); multiply only through
  `fom_apply_<name>`, which `{verify}` checks against the matrix. `{verify}`
  reports a breach as `VERIFY FAILED: kernel rule: ...`."""
    shape_block = (
        """
This specification is UNROLLED: its body is the whole recorded run, not one
iteration to repeat. Execute the steps once, in order, then apply the final
state update. Do not write a loop over iterations and do not read
`work->settings.max_iter` in `fom_solve` — see SPEC.md for why.
"""
        if ir.unrolled
        else ""
    )
    shape_block += _objective_block(ir, target.objective_source_path())
    return f"""\
You are generating a high-performance, embedded implementation of a verified
first-order optimization method named `{ir.name}`.
{shape_block}

Write the file `{src}` ONLY. Everything else in this directory is already
generated and must not be changed:

- `include/fom_solver.h` defines the ABI you must implement exactly:
  `fom_set_default_settings`, `fom_setup`, `fom_update_params`, `fom_solve`,
  `fom_cleanup`.
- `include/fom_workspace.h` defines the workspace struct and dimension macros.
- `include/fom_data.h` / `src/fom_data.c` hold ALL numeric coefficients and
  activation constants (matrices `FOM_MAT_*`, `FOM_STATE_INIT`, soft-threshold
  `FOM_*_LAMBDA`, clip-bound vectors `FOM_MAT_*_lo`/`_hi`). Do NOT hardcode
  numbers; read them from these symbols.{ops_rule}
- `include/fom_linsolve.h` / `{linsolve_src}` provide the linear solve for
  every affine operator: for a step that SPEC marks with a solve, assemble its
  right-hand side from `linear_op`, then call `fom_lin_solve_<name>(rhs, out)`.
  NEVER form, store, or use a matrix inverse — the solve is already implemented
  for you (a precomputed sparse factorization, or conjugate gradient).
- `include/fom_ops.h` / `{ops_src}` provide every matrix-vector product as an
  accumulating helper `fom_apply_<matrix>(source, out)` (out += M @ source),
  implemented sparse (CSR) where the matrix is structured. Assemble each
  `linear_op` by zeroing the output and calling these — do NOT write your own
  dense matvec loops, and do not hardcode the matrices.
- The algorithm is specified in `SPEC.md`.
{notes_block}
Hard constraints for the embedded target:
- No dynamic memory: no malloc/calloc/realloc/free, no variable-length arrays.
- Division-free: do not use the `/` operator anywhere in the code. The affine
  inverse has been folded into the coefficient matrices precisely so that no
  division is needed.
- Use only the statically-sized buffers already declared in the workspace.

Each solve must start from `FOM_STATE_INIT` (reset the state in `fom_setup` and
`fom_update_params`) so results match the reference exactly.

Run `{verify}`; it builds your code and checks it against the reference oracle
over many samples. Iterate until it prints `VERIFY OK` and exits 0. Strive for
the fastest correct implementation.
"""


def _exact_literals_note(exact: tuple[str, ...]) -> str:
    """The exception to the literal rule: routines checked on every unit vector."""
    if not exact:
        return ""
    names = ", ".join(f"`{fn}`" for fn in exact)
    return f"""
  Exception: {names} are checked on every unit vector, which recovers the matrix
  each applies exactly, so inside these routines (and helpers only they call) you
  may write coefficients as literals: unroll the product over its fixed sparsity
  with the values inlined, drop zero and unit coefficients, and keep constant
  tables in `static const` arrays inside the routine. The values must be exactly
  the matrix's: verify compares each entry with the package's data to a
  tolerance far below anything a perturbed coefficient could hide in."""


def _kernel_rules_block(
    src: str,
    ops_src: str,
    verify: str,
    ops_editable: bool,
    fused: bool = False,
    exact: tuple[str, ...] = (),
) -> str:
    """The kernel rules (`fomtools.codegen.kernel_rules`), on every target."""
    if ops_editable and fused:
        own_data = f""" In `{ops_src}`, each `fom_apply_<name>` (one
  definition) and the helpers it calls read only its own matrix's data, and each
  `fom_fused_<block>`, `fom_step_<block>` (and its `__rest`) only its members'
  data, `fom_iterate` that of the whole iteration; no other code reads matrix data,
  every function but these entry points is `static`, and nothing in
  `{ops_src}` reads the schedule (`fom_sched_*`, `FOM_SCHED_*`): a routine
  takes its schedule values as arguments and must be right for any value."""
    elif ops_editable:
        own_data = f""" In `{ops_src}`, each `fom_apply_<name>` (one
  definition) and the helpers it calls read only its own matrix's data; no other
  code reads matrix data, and every function but the `fom_apply_<name>` entry
  points is `static`."""
    else:
        own_data = ""
    files = f"`{src}` and `{ops_src}`" if ops_editable else f"`{src}`"
    same_routine = f", in the same routine of `{ops_src}`," if ops_editable else ""
    return f"""

Three rules keep every coefficient inside the checked `fom_apply_<name>`
routines; `{verify}` reports a breach as `VERIFY FAILED: kernel rule: ...`, and a
rewrite that breaks any of them is discarded when it is collected:
- `{src}` must not read the `FOM_MAT_<name>` data of any matrix it multiplies
  by; call `fom_apply_<name>` instead.{own_data}
- {files} must not add floating-point constants other than `0.0`,
  `0.5`, `1.0` and `2.0`, whether written as decimal or hex float literals or as
  integers cast to a floating type (literals the translated file already
  had{same_routine} are fine).{_exact_literals_note(exact)}
- `fom_setup` and `fom_update_params` must not add calls to `fom_apply_*`{" or `fom_fused_*`, `fom_step_*`, `fom_iterate`" if fused else ""}: apply
  matrices in the iteration, never probe them to rebuild a matrix."""


def code_optimization_instructions(ir: FomIR, target: TargetDescriptor) -> str:
    """Instructions for the code optimization phase (profile-guided rewriting)."""
    verify = " ".join(target.verify_command())
    src = target.solver_source_path()
    ops_src = target.ops_source_path()
    linsolve_src = target.linsolve_source_path()
    notes = target.agent_notes()
    notes_block = f"\n{notes}\n" if notes else ""
    editable = target.optimization_editable_paths()
    ops_data = target.ops_data_path()
    fused = target.fused_routines(ir) if ops_src in editable else {}
    selections = target.selection_matrices(ir)
    steps = target.step_routines(ir) if ops_src in editable else {}
    iterate = target.iterate_routine(ir) if ops_src in editable else False
    selection_rows = target.selection_rows(ir) if ops_src in editable else {}
    rests = target.rest_routines(ir) if ops_src in editable else []
    if linsolve_src in editable:
        both = ops_src in editable
        others = f"`{linsolve_src}` (the linear solves, usually the hottest code)"
        if both:
            others += f" and `{ops_src}` (the matrix-vector products)"
        fixed = "" if both else f", and `{ops_src}`"
        scope = f"""\
You may edit `{src}` and {others}. Everything else — the Makefile, build flags,
headers (including the `fom_lin_solve_<name>` and `fom_apply_<name>` signatures
in `include/`), numeric data in `src/fom_data.c`{fixed} — is fixed and restored
automatically before every check, so changes to it are discarded (don't waste
turns editing it).

`{linsolve_src}` holds each solve's precomputed factorization (permutation,
unit-lower factor, reciprocal diagonal) as static data plus the routine that
applies it. You may restructure both — e.g. a dense or fully unrolled
triangular solve for a small or dense factor, a blocked/supernodal layout,
folding the permutation into the factor, fusing the gather/scatter — but each
`fom_lin_solve_<name>` must still solve the SAME system `M x = b` by applying
that factorization: never form, store, or use a matrix inverse. `{verify}` also
checks every `fom_lin_solve_<name>` directly against its reference solve. To
profile inside it, `#include "fom_profile.h"` there."""
        if both:
            scope += f"""

`{ops_src}` implements each `fom_apply_<name>(x, out)` (out += M @ x) over the
matrix data in `{ops_data}`. Put any specialized matrix-vector kernel HERE,
inside the `fom_apply_<name>` it replaces — e.g. exploiting Toeplitz, Hadamard,
banded or block structure, unrolling a small block, or a better sparse loop — and
call it from `{src}`; do not write matrix-vector loops in `{src}`. Read the
coefficients from `{ops_data}` (or values derived from them at setup); never
hardcode them. `{verify}` checks every `fom_apply_<name>` directly against its
matrix, column by column (probing each unit vector) and on random vectors, so a
kernel must apply exactly `M` for every input and must keep accumulating into
`out`."""
        if both and fused:
            names = ", ".join(f"`fom_fused_{b}`" for b in fused)
            scope += f"""

`{ops_src}` also implements one FUSED routine per output block ({names};
prototypes and formulas in `include/fom_ops.h`): `fom_fused_<block>(sources...,
schedule values..., out)` adds every per-iteration product of that block in one
call, `out += sum_k s_k * (M_k @ src_k)`. Parameter terms are left out so you
can still compute them once per solve. Calling the fused routine in place of
the separate `fom_apply_*` calls, and rewriting it, is how to merge the products
into one pass and exploit relations BETWEEN its member matrices: one member
equal to a rearrangement or a signed copy of another's block, a shared block,
members that cancel, so `A x + g D x` collapses to a single smaller product. A
fused routine may read every member's data (and only those). `{verify}` checks
it directly against its members: every unit vector of every source, with every
schedule value at 0 and with each alone at 1 (which recovers each member
exactly), and on random sources and schedule values. So it must be exact for
ANY schedule value, not just the ones the solver passes."""
        if both and steps:
            names = ", ".join(f"`{fn}`" for fn in steps)
            scope += f"""

It also implements one STEP routine per output block ({names}; prototypes in
`include/fom_ops.h`): `fom_step_<block>(sources..., schedule values..., bias,
act, out)` computes the whole block, `out = act(bias + sum_k s_k * (M_k @
src_k))`, and OVERWRITES `out`. `bias` is everything outside the per-iteration
products (the constant plus the parameter terms, which you still compute once
per solve) and `act` the activation's parameters (pass `&FOM_ACT_<block>`;
blocks without an activation take none). Use it to fold the activation and the
bias into the product pass, write the result straight into its destination, and
drop the zeroing and copy passes. It must apply whatever `act` it is passed:
`{verify}` checks it with the activation switched off (which recovers every
member exactly, unit vector by unit vector, at every schedule setting), at the
package parameters, and at random parameters that make it active on about half
the rows. Never hardcode a bound or threshold, and never read `FOM_ACT_*` or the
constant and bound vectors in `{ops_src}`."""
        if both and iterate:
            scope += f"""

`fom_iterate(state, schedule values..., bias_<block>..., act_<block>...,
t_<name>..., new_state)` runs one whole iteration: every body step in order,
writing each temporary `t_<name>` (the solver and `fom_objective` still need
them), then the state update into `new_state`. By default it calls the step
routines; rewrite it to compute the iteration in one pass, keep values in
registers between blocks, and share intermediates between them. `{verify}`
checks every temporary and the new state against the chained reference steps:
unit vectors of the state at every schedule setting with the activations off,
then random states, biases and activation parameters. It may read the data of
every matrix of the iteration."""
        if both and selection_rows:
            items = "\n".join(f"  - `{name}`: {what}" for name, what in selection_rows.items())
            rest = ", ".join(f"`{fn}`" for fn in rests) or "none"
            scope += f"""

These output blocks have SELECTION ROWS, rows that are a plain copy of one
source entry (see SPEC.md; `FOM_SELROWS_<block>_ROW/SRC/COL` in
`include/fom_data.h`):
{items}
You may write those rows in `{src}` yourself, for example a step block's row as
`out[row] = act(bias[row] + src[col])` straight into its destination, and call
the block's `__rest` routine ({rest}) for the other rows: it computes every row
but the selection rows, which it must leave untouched (`{verify}` checks both).
Copies you write are checked only by the end-to-end `{verify}`, so get the
indices exactly right."""
        if selections:
            items = "\n".join(f"  - `{name}`: {what}" for name, what in selections.items())
            scope += f"""

These products are 0/1 SELECTIONS (at most one 1.0 per row, see SPEC.md):
{items}
You may apply them in `{src}` yourself, as copies or by writing a result
straight into its destination, instead of calling `fom_apply_<name>`; the index
map is in `include/fom_data.h` as `FOM_SEL_<name>_SRC` (-1 for an empty row).
Such a copy is checked only by the end-to-end `{verify}`, so get the indices
exactly right. This is the one exception to never writing a product in `{src}`."""
    else:
        scope = f"""\
Edit ONLY `{src}`: the Makefile, build flags, headers, numeric data,
`{linsolve_src}`, and `{ops_src}` are fixed and restored automatically before
every check, so changes to them are discarded (don't waste turns editing them)."""
    exact = tuple(target.exact_routines(ir)) if ops_src in editable else ()
    scope += _kernel_rules_block(src, ops_src, verify, ops_src in editable, bool(fused), exact)
    objective_src = target.objective_source_path()
    objective_block = _objective_block(ir, objective_src)
    if objective_block:
        objective_block += f"""Keep calling `fom_objective` exactly where it is called now: the
checks it gates are part of the verified behavior, and their cost is already
amortized over the check interval. It lives in `{objective_src}`, which stays fixed
whatever else you may edit; do not reimplement or inline the metric anywhere.
"""
    return f"""\
The solver `{src}` already passes `{verify}`. Now make it as FAST as possible
WITHOUT ever breaking correctness. {scope}

Work as a tight, profile-guided loop:

1. PROFILE. Run `make profile`. It prints, per instrumented region, a line
   `PROF region=<name> total_us=<t> calls=<n> pct=<%>` sorted hottest-first, and
   a `BENCH ... mean_us=<...>` line for the whole solve. Use this to identify the
   current bottleneck. Wrap the hot sections of the iteration in
   `FOM_PROF_BEGIN(name)` / `FOM_PROF_END(name)` macros (from `fom_profile.h`,
   already included) so you get a meaningful breakdown — these are no-ops in the
   normal build, so they cost nothing in production.
2. OPTIMIZE the hottest region with one careful, well-understood change, e.g.:
   - hoist iteration-invariant work out of the loop (precompute once in
     `fom_setup` / `fom_update_params` into workspace scratch);
   - exploit matrix structure (sparsity, symmetry, identity/zero blocks — see
     `spec.json`);
   - improve memory access / enable vectorization; reduce redundant passes.
3. VERIFY. Run `{verify}` — it MUST still print `VERIFY OK`. If it fails, revert
   that change and try a different one. Never keep a change that breaks
   correctness.
4. MEASURE. Run `make bench` and compare `mean_us` against the previous best.
   Keep the change only if it is faster (or clearly enables a later win).
5. Repeat from step 1 until further changes give no meaningful improvement
   (e.g. two rounds with < ~2% gain) or the code is clearly optimal.

Keep the target constraints throughout, in every file you edit (embedded: no
dynamic allocation, division-free).
{objective_block}{notes_block}
In your FINAL version, KEEP `FOM_PROF_BEGIN/END` markers around
the main computational regions (they are no-ops in production) so that
`make profile` still reports a per-region breakdown after optimization.

When done, print a summary as the last thing you output, bracketed by these two
marker lines exactly, each alone on its own line:

{_SUMMARY_BEGIN}
...your summary here...
{_SUMMARY_END}

Between the markers, give:
- the initial vs. final whole-solve `mean_us` and the overall speedup factor;
- the profiling breakdown of the FINAL solver (region, time, %), hottest first;
- a section headed exactly `{APPLIED_HEADING}` listing the optimizations you
  applied, one per line as `- <the change> | <the structure it exploits>`, so a
  reader (and the next round) can tell what made it valid here;
- a section headed exactly `{BLOCKED_HEADING}` listing every optimization you
  considered but did not make because a rule forbade it (the kernel rules, the
  files you may edit, or the target constraints), one per line as
  `- <the idea> | <the rule that blocked it> | <expected gain, or unknown>`,
  or the single line `- none`. Be specific: these reports decide which rules
  are revisited.

The markers are how your summary is lifted into the generated `OPTIMIZATIONS.md`
report, so emit them literally, and only once.
"""


def code_optimization_continuation(
    round_index: int,
    rounds: int,
    current_us: float | None,
    translated_us: float | None,
    last_gain: float | None,
    profile: list | None = None,
    history: list[dict] | None = None,
    patience: int | None = None,
) -> str:
    """The block appended to `code_optimization_instructions` for a later round.

    Each round is a fresh agent run on the solver the previous rounds left (the
    best one so far, re-verified and benchmarked by the pipeline), so the agent
    has to be told where it stands: which round this is, the current and the
    translated whole-solve times, and what the last round gained. ``history``
    (the earlier rounds, see `fomtools.codegen.generate.OptimizationRound`)
    adds what each one applied and what a rule blocked, and why a round was
    reverted (`_history_lines`). ``patience`` is ``agent_optimize_patience``.
    """
    lines = [
        "",
        f"This is optimization round {round_index} of at most {rounds}. The editable "
        "sources already hold the fastest verified solver from the earlier rounds; "
        "start from it, not from the translation.",
    ]
    if current_us is not None:
        line = f"It currently takes {current_us:.3f} us per solve (`make bench` mean_us)"
        if translated_us is not None and current_us > 0:
            line += (
                f", against {translated_us:.3f} us for the translated solver "
                f"({translated_us / current_us:.2f}x so far)"
            )
        lines.append(line + ".")
    if last_gain is not None:
        lines.append(f"The previous round made it {100 * last_gain:.1f}% faster.")
    if profile:
        lines.append("Its current profile (`make profile`), hottest first:")
        lines += [f"  {r.name:<24} {r.total_us:12.1f} us  {r.pct:6.2f}%" for r in profile]
    if history:
        lines += _history_lines(history)
    stop = (
        f"after {patience} rounds in a row that fail, gain too little or change nothing"
        if patience and patience > 1
        else "at the first round that fails, gains too little or changes nothing"
    )
    lines += [
        "Look for further speedups, including structure the earlier rounds did not "
        "exploit. Do not repeat an approach the earlier rounds show was reverted "
        "unless you can fix what made it fail; a reverted round is a reason to try a "
        "different idea, not to stop. If you find nothing that helps, leave the "
        "sources unchanged and say so in your summary. The pipeline reverts a round "
        "that fails or gains too little, and stops " + stop + ".",
        "",
    ]
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    """``text`` on one line, cut to ``limit`` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _history_lines(history: list[dict], item_chars: int = 160, max_chars: int = 3000) -> list[str]:
    """The earlier rounds, oldest first, for a later round's prompt.

    Each round is a header (kept or reverted, its gain, and the one-line
    ``reason`` it was reverted or made no gain) and what its summary listed under
    `APPLIED_HEADING` (for a reverted round, what was tried) and
    `BLOCKED_HEADING`. Timings and profiles
    are left out: the prompt already gives the current ones. Each item is cut to
    ``item_chars``; past ``max_chars`` in all, the oldest rounds shrink to their
    header, then are dropped, so the newest stays whole.
    """
    entries = []
    for r in history:
        if r.get("kept"):
            status = "kept"
        elif r.get("changed", True):
            status = "reverted"
        else:
            status = "no change"
        head = f"Round {r['index']}: {status}"
        # A reason marks a round without a gain (a kept first round within the
        # noise tolerance may even be slower), so only a round without one has
        # a gain to print.
        if r.get("kept") and r.get("gain") is not None and not r.get("reason"):
            head += f", {100 * r['gain']:.1f}% faster"
        if r.get("reason") and r.get("changed", True):
            head += f" ({_clip(r['reason'], 2 * item_chars)})"
        applied = [_clip(a, item_chars) for a in r.get("applied") or []]
        blocked = [
            _clip(" | ".join(p for p in (b.get("idea"), b.get("rule")) if p), item_chars)
            for b in r.get("blocked") or []
        ]
        detail = []
        if applied:
            detail.append("  applied: " + "; ".join(applied))
        if blocked:
            detail.append("  blocked: " + "; ".join(blocked))
        if not detail:
            detail.append("  (no list of changes recovered from its summary)")
        entries.append([head, detail])

    def size() -> int:
        return sum(len(h) + 1 + sum(len(d) + 1 for d in ds) for h, ds in entries)

    omitted = 0
    for entry in entries[:-1]:
        if size() <= max_chars:
            break
        entry[1] = []
    while size() > max_chars and len(entries) > 1:
        entries.pop(0)
        omitted += 1

    lines = ["Earlier rounds, oldest first (what each applied or tried, and what a rule blocked):"]
    if omitted:
        lines.append(f"  ({omitted} earlier round{'s' if omitted > 1 else ''} omitted)")
    for head, detail in entries:
        lines += [head, *detail]
    return lines
