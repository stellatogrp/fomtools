# API and modules

The FOMTools API is organized around a single entry point, the
[`Verifier`](verifier.md), and a set of modules that describe the algorithm
to be verified, the function class it is run on, and the numerical backend used
to solve the resulting verification problem.

A typical workflow is: declare the problem data and iterates on a `Verifier`,
build up the algorithm from [step operators](steps.md) and
[proxable functions](proxablefunctions.md), state the assumptions and the
quantity of interest as [linear expressions](linearexpression.md) and
[constraints](constraint.md), and then let a
[canonicalizer](canonicalizer.md) translate everything into a mixed-integer
or semidefinite program that a [solver](solvers.md) solves to obtain a
worst-case bound.

The source code of every module is in the
[`fomtools` package on GitHub](https://github.com/stellatogrp/fomtools/tree/main/fomtools).

## Modules

* **[Verifier](verifier.md)** holds the
  optimization variables and their bounds, applies step operations, propagates
  interval bounds, and drives the canonicalizers and OBBT handlers.
* **[Linear expressions](linearexpression.md)** are linear combinations of
  iterates and function values, with arithmetic and `@` support, used to state
  objectives and performance measures.
* **[Constraints](constraint.md)** are equalities and inequalities over
  linear expressions, used to encode assumptions on the problem data and the
  initial conditions.
* **[Step operators](steps.md)** are the building blocks of a first-order
  method, covering affine steps, ReLU, saturated linear, and soft thresholding.
* **[Canonicalizers](canonicalizer.md)** translate a verification problem
  into a solver-ready formulation (MIP or SDP), including optimization-based
  bound tightening (OBBT).
* **[Solvers](solvers.md)** are thin wrappers that put the numerical
  backends, such as Gurobi and HiGHS, behind a common interface.
* **[Proxable functions](proxablefunctions.md)** describe the function class
  an algorithm is verified over, each supplying the proximal operator and the
  representation needed by the verification problem.
* **[Code generation](codegen.md)** emits low-level solver code for a
  verified first-order method, together with a small Python wrapper.
