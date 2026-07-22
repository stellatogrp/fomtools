## Tutorials

These tutorials show many different ways to use FOMTools:

### Verification

These notebooks verify the performance of first-order methods on five example
problems.

* [Lasso](tutorials/lasso_verification.html)
* [Optimal control](tutorials/control_verification.html)
* [Network flow](tutorials/network_verification.html)
* [Nonnegative least squares](tutorials/nnls_verification.html)
* [Portfolio](tutorials/portfolio_verification.html)

### Code generation

Read the [code generation tutorial](tutorials/codegen.md) for an explanation of
how FOMTools turns a verified first-order method into an embedded C solver.

#### Examples

Each of these notebooks builds a FOM, generates an embedded C solver for it, and
times the generated solver against direct Python. The
[summary table](tutorials/codegen_speedup_results.md) collects the results of all
five side by side.

* [Lasso](tutorials/lasso_codegen.html)
* [Optimal control](tutorials/control_codegen.html)
* [Network flow](tutorials/network_codegen.html)
* [Nonnegative least squares](tutorials/nnls_codegen.html)
* [Portfolio](tutorials/portfolio_codegen.html)

### CVXPYgen integration

After the FOMTools code-generation tutorials, see the
[CVXPYgen integration tutorial](tutorials/cvxpygen.md) and the
[CVXPYgen MPC example](tutorials/cvxpygen_mpc.html).
