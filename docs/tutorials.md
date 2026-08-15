## Tutorials

These tutorials show many different ways to use FOMTools:

### Verification

These notebooks verify the performance of first-order methods on five example
problems.

* [Lasso](tutorials/verification/lasso.html)
* [Optimal control](tutorials/verification/control.html)
* [Network flow](tutorials/verification/network.html)
* [Nonnegative least squares](tutorials/verification/nnls.html)
* [Portfolio](tutorials/verification/portfolio.html)

### Code generation

Read the [code generation tutorial](tutorials/codegen.md) for an explanation of
how FOMTools turns a verified first-order method into an embedded C solver.

#### Examples

Each of these notebooks builds a FOM, generates an embedded C solver for it, and
times the generated solver against direct Python. The
[summary table](tutorials/codegen/speedup_results.md) collects the results of all
five side by side.

* [Lasso](tutorials/codegen/lasso.html)
* [Optimal control](tutorials/codegen/control.html)
* [Network flow](tutorials/codegen/network.html)
* [Nonnegative least squares](tutorials/codegen/nnls.html)
* [Portfolio](tutorials/codegen/portfolio.html)

### CVXPYgen integration

After the FOMTools code-generation tutorials, see the
[CVXPYgen integration tutorial](tutorials/cvxpygen.md) and the
[CVXPYgen MPC example](tutorials/cvxpygen_mpc.html).
