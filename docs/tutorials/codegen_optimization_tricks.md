# Code Generation Optimization Tricks

This file summarizes the C optimization tricks used by the code generation speedup examples. 

The generated C timing is the result just after the first code generation pass. The optimized C timing is after the live C-editing iteration changed `src/fom_solver.c`, re-ran verification, and kept only changes that stayed correct.

## Overall Pattern

| Trick | Why it helps | Used in |
|---|---|---|
| Replace generic helper calls with scalar code for fixed-size problems | Removes tiny loops, indirect indexing, repeated zeroing, and function-call overhead | Lasso, network flow, nonnegative least squares, portfolio, optimal control |
| Hoist values that do not change inside a solve | Avoids recomputing parameter terms and constant offsets on every iteration | Lasso, nonnegative least squares, network flow, optimal control |
| Keep state in local scalar variables during the hot loop | Reduces repeated loads from and stores to the workspace arrays | Lasso, network flow, portfolio |
| Inline small linear solves directly into `src/fom_solver.c` | Avoids generic solve-call overhead when the factorization size is fixed | Optimal control, portfolio |
| Replace sparse state-update helpers with direct assignments | Uses known identity and sparse structure instead of general matrix-vector code | Nonnegative least squares, network flow, optimal control, portfolio |
| Move bookkeeping out of the hot loop when it is only needed at the end | Reduces per-iteration writes to `solution.iter`, `solution.residual`, or final residual variables | Nonnegative least squares, optimal control, portfolio |
| Specialize fixed active sets after verification | Removes clamp branches once the trajectory is known to stay on a stable bound pattern | Portfolio |
| Force inlining when the compiler does not inline a hot helper | Removes an unexpected call from every iteration | Portfolio |
| Reject correct but slower edits | Verification alone is not enough; every accepted edit also has to improve timing | All examples |

## Per-Example Notes

### Lasso

Timing: generated C `7.335 us`, optimized C `1.546 us`, C optimization speedup `4.75x`.

Kept changes:

- Replaced generic matrix helper calls with the explicit four-variable recurrence.
- Hoisted the parameter-dependent affine terms once per solve.
- Replaced the generic soft-thresholding loop with direct scalar threshold calls.
- Kept the four state variables in local scalars and wrote the workspace arrays once at the end.
- Kept profiling markers around the affine update and state update regions.

TLDR: for tiny fixed-size Lasso, the cost was mostly helper-call and loop overhead, not arithmetic.

### Optimal Control

Timing: generated C `153.596 us`, optimized C `29.704 us`, C optimization speedup `5.17x`.

Kept changes:

- Inlined the fixed-size linear solve instead of calling the generic solve path.
- Removed avoidable branch and indexing overhead inside the solve loops.
- Replaced the state-update helper calls with direct assignments using the known structure.
- Replaced the projection step with direct bound logic where the bounds were fixed.
- Folded known constants into the solve right-hand side.
- Added final-only state commit and residual handling where it preserved the reported result.

Rejected or not kept:

- A countdown loop version verified but benchmarked slower, so it was reverted.
- An in-place update version verified but benchmarked slower, so it was reverted.

TLDR:  the biggest win came from specializing the fixed linear solve and direct state update.

### Network Flow

Timing: generated C `10.613 us`, optimized C `2.593 us`, C optimization speedup `4.09x`.

Kept changes:

- Scalarized the whole 11-state recurrence.
- Replaced generic clipping and nonnegative projection helpers with direct scalar clamp and ReLU code.
- Split the final residual-producing step from the non-final loop, removing a branch from most iterations.
- Fused final state and solution stores where it was neutral or slightly positive.
- Kept profiling markers around the three main regions.

Rejected or not kept:

- Hoisting a couple of loop-invariant products verified but made the code slower, likely because it increased register pressure.
- Algebraic factorization of repeated coefficients verified but was slower, so it was reverted.
- A default-iteration fast path verified but was slower, so it was reverted.

TLDR: fewer arithmetic operations is not always faster; register pressure and compiler scheduling can dominate very small kernels.

### Nonnegative Least Squares

Timing: generated C `9.352 us`, optimized C `1.459 us`, C optimization speedup `6.41x`.

Kept changes:

- Precomputed the parameter contribution to the affine update once per solve.
- Inlined the fixed three-by-three state multiply.
- Replaced the identity state-update helper with direct assignment.
- Moved `solution.iter` and `solution.residual` writes out of the per-iteration loop when the final values were the only observable values.

Rejected or not kept:

- Keeping state entirely in local variables verified but benchmarked slower.
- Removing `new_state` writes from every iteration verified but benchmarked slower.
- Computing residual only on the final iteration verified but did not beat the simpler loop.

MTLDR: the useful win was removing generic helper overhead; several seemingly obvious memory-write reductions were already handled well by the compiler or hurt scheduling.

### Portfolio

Timing: generated C `14.831 us`, optimized C `6.351 us`, C optimization speedup `2.34x`.

Kept changes:

- Replaced the `t1` right-hand-side build with direct scalar formulas.
- Replaced sparse state-update helpers with direct scalar assignments.
- Replaced the projection helper with direct clamps.
- Inlined the fixed four-variable linear solve.
- Changed the step to update state in place while preserving final scratch arrays.
- Forced inlining of a hot helper after checking that the compiler had emitted it as a real function.
- Added a fixed-active-set path after the first few startup iterations, while keeping the generic path for early iterations and other iteration counts.

Rejected or not kept:

- Removing one fixed `t0[4]` load verified but did not materially improve timing.
- A more aggressive default-iteration fast path was tested only after verification; it was not the final kept result.

TLDR: this problem needed several small structural wins. The largest came from inlining the solve, in-place state updates, and the verified active-set specialization.
