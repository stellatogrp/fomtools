# Verification

These notebooks verify the worst-case performance of first-order methods on five
example problems. Each one sets up a [`Verifier`](../api/verifier.md), declares
the problem data and the initial condition, writes the algorithm out of
[step operators](../api/steps.md), and solves the resulting verification problem
exactly as a mixed-integer linear program (`method='mip'`).

## Examples

* **[Lasso](verification/lasso.html)** — ISTA on the $\ell_1$-regularized least-squares
  problem, verified with a soft-threshold step.
* **[Optimal control](verification/control.html)** — a linear-quadratic control problem with
  input bounds, verified with a saturated linear step.
* **[Network flow](verification/network.html)** — a capacitated min-cost
  network-flow linear program, solved by an operator-splitting method.
* **[Nonnegative least squares](verification/nnls.html)** — projected gradient
  descent, whose projection onto the nonnegative orthant is a ReLU step.
* **[Portfolio](verification/portfolio.html)** — a mean–variance portfolio
  problem with box bounds and a budget constraint, solved by ADMM.

Once a method is verified, the [code generation](codegen.md) tutorial shows how
to turn the same `Verifier` object into an embedded C solver, and the
[code generation guide](../user_guide/codegen.md) documents that pipeline in full.
