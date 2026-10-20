# Tutorials

These tutorials show many different ways to use FOMTools, following the same path
a project usually takes: verify a first-order method, generate a fast
implementation of it, and then drive that implementation from a CVXPY model.
The notebooks are [marimo](https://marimo.io) files; their sources are in
[`docs/tutorials` on GitHub](https://github.com/stellatogrp/fomtools/tree/main/docs/tutorials).

## Sections

* **[Verification](verification.md)** contains five notebooks that
  verify the worst-case performance of first-order methods on example problems —
  lasso, optimal control, network flow, nonnegative least squares, and portfolio
  optimization.
* **[Code generation](codegen.md)** works through the same five problems end to
  end, turning each verified method into an embedded C solver and timing it
  against direct Python. The [code generation guide](../user_guide/codegen.md) is the
  reference for the pipeline itself.
* **[CVXPYgen integration](cvxpygen.md)** models a parametric problem
  in CVXPY and solves it with a generated FOM, combining CVXPYgen's
  canonicalization code with a FOMTools solver in one embedded C project.
