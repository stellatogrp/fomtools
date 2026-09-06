# Tutorials

These tutorials show many different ways to use FOMTools, following the same path
a project usually takes: verify a first-order method, generate a fast
implementation of it, and then drive that implementation from a CVXPY model.

## Sections

* **[Verification](verification.md)** contains five notebooks that
  verify the worst-case performance of first-order methods on example problems —
  lasso, optimal control, network flow, nonnegative least squares, and portfolio
  optimization.
* **[Code generation](codegen.md)** explains how FOMTools turns a
  verified first-order method into an embedded C solver, and works through the
  same five problems end to end with timings against direct Python.
* **[CVXPYgen integration](cvxpygen.md)** models a parametric problem
  in CVXPY and solves it with a generated FOM, combining CVXPYgen's
  canonicalization code with a FOMTools solver in one embedded C project.
