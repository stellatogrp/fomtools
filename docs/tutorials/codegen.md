# Code generation

These notebooks turn a verified first-order method into an embedded C solver.
Each one builds a FOM on a [`Verifier`](../api/verifier.md), calls
[`codegen_rolled`](../api/verifier.md#fomtools.Verifier.codegen_rolled) on a
single stationary iteration to emit a self-contained C project, and times the
generated solver against direct Python.
Each has a companion write-up covering the problem formulation and the update
rule it uses.

## Examples

* **Lasso** — [notebook](codegen/lasso.html), [write-up](codegen/lasso.md) —
  ISTA on a small $\ell_1$-regularized least-squares problem, built from a
  soft-threshold step.
* **Optimal control** — [notebook](codegen/control.html),
  [write-up](codegen/control.md) — a finite-horizon linear-quadratic control
  problem with input bounds, solved by ADMM with a saturated linear step.
* **Network flow** — [notebook](codegen/network.html),
  [write-up](codegen/network.md) — a capacitated min-cost network-flow problem,
  solved by PDHG.
* **Nonnegative least squares** — [notebook](codegen/nnls.html),
  [write-up](codegen/nnls.md) — projected gradient descent, whose projection
  onto the nonnegative orthant is a ReLU step.
* **Portfolio** — [notebook](codegen/portfolio.html),
  [write-up](codegen/portfolio.md) — a mean–variance portfolio problem, solved
  by ADMM.

The [summary table](codegen/speedup_results.md) collects the timings of all five
side by side; the recorded runs range from 18× to 222× faster than direct NumPy
before the optimization pass, and 95× to 910× after it.

## Going further

The [code generation guide](../user_guide/codegen.md) is the reference for the
pipeline these notebooks exercise: how a recorded run rolls back up into a
compact loop, how to declare an iteration-varying coefficient such as FISTA's
momentum, how to stop early on a tolerance, what the emitted project contains,
and how the profile-guided optimization phase works. Once a solver is generated,
the [CVXPYgen integration](cvxpygen.md) tutorial shows how to drive it from a
CVXPY model.
