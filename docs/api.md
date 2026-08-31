# API and modules

The FOMTools API is organized around a single entry point, the
[`Verifier`](api/verifier.md), and a set of modules that describe the algorithm
to be verified, the function class it is run on, and the numerical backend used
to solve the resulting verification problem.

A typical workflow is: declare the problem data and iterates on a `Verifier`,
build up the algorithm from [step operators](api/steps.md) and
[proxable functions](api/proxablefunctions.md), state the assumptions and the
quantity of interest as [linear expressions](api/linearexpression.md) and
[constraints](api/constraint.md), and then let a
[canonicalizer](api/canonicalizer.md) translate everything into a mixed-integer
or semidefinite program that a [solver](api/solvers.md) solves to obtain a
worst-case bound.

## Modules

* **[Verifier](api/verifier.md)** holds the
  optimization variables and their bounds, applies step operations, propagates
  interval bounds, and drives the canonicalizers and OBBT handlers.
* **[Linear expressions](api/linearexpression.md)** are linear combinations of
  iterates and function values, with arithmetic and `@` support, used to state
  objectives and performance measures.
* **[Constraints](api/constraint.md)** are equalities and inequalities over
  linear expressions, used to encode assumptions on the problem data and the
  initial conditions.
* **[Step operators](api/steps.md)** are the building blocks of a first-order
  method, covering affine steps, ReLU, saturated linear, and soft thresholding.
* **[Canonicalizers](api/canonicalizer.md)** translate a verification problem
  into a solver-ready formulation (MIP or SDP), including optimization-based
  bound tightening (OBBT).
* **[Solvers](api/solvers.md)** are thin wrappers that put the numerical
  backends, such as Gurobi and HiGHS, behind a common interface.
* **[Proxable functions](api/proxablefunctions.md)** describe the function class
  an algorithm is verified over, each supplying the proximal operator and the
  representation needed by the verification problem.
* **[Code generation](api/codegen.md)** emits low-level solver code for a
  verified first-order method, together with a small Python wrapper.
