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


def translation_instructions(ir: FomIR, target: TargetDescriptor) -> str:
    """Instructions for the translation phase: a faithful translation of the spec."""
    verify = " ".join(target.verify_command())
    src = target.solver_source_path()
    ops_src = target.ops_source_path()
    linsolve_src = target.linsolve_source_path()
    notes = target.agent_notes()
    notes_block = f"\n{notes}\n" if notes else ""
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
  numbers; read them from these symbols.
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


def code_optimization_instructions(ir: FomIR, target: TargetDescriptor) -> str:
    """Instructions for the code optimization phase (profile-guided rewriting)."""
    verify = " ".join(target.verify_command())
    src = target.solver_source_path()
    ops_src = target.ops_source_path()
    linsolve_src = target.linsolve_source_path()
    notes = target.agent_notes()
    notes_block = f"\n{notes}\n" if notes else ""
    return f"""\
The solver `{src}` already passes `{verify}`. Now make it as FAST as possible
WITHOUT ever breaking correctness. Edit ONLY `{src}`: the Makefile, build flags,
headers, numeric data, `{linsolve_src}`, and `{ops_src}` are fixed and restored
automatically before every check, so changes to them are discarded (don't waste
turns editing them). Work as a tight, profile-guided loop:

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

Keep the target constraints throughout (embedded: no dynamic allocation,
division-free).
{notes_block}
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
- a short list of the optimizations you applied and why. For each one, name the
  structure you exploited, so a reader can tell what made it valid here.

The markers are how your summary is lifted into the generated `OPTIMIZATIONS.md`
report, so emit them literally, and only once.
"""
