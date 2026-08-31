# Tutorials

These tutorials show many different ways to use FOMTools.

## Verification

Formulating a first-order method as a verification problem and proving bounds on its behavior.

* [Nonnegative least squares](tutorials/nnls_verification.html)

## Code generation

Turning a verified first-order method into an embedded solver, and integrating with CVXPYgen.

* [Code generation](tutorials/codegen.md)
* [CVXPYgen integration](tutorials/cvxpygen.md)
* [CVXPYgen MPC example](tutorials/cvxpygen_mpc.html)

### Speedups

Each of these notebooks builds a FOM, generates an embedded C solver for it, and
times the generated solver against direct Python. The
[summary table](tutorials/codegen_speedup_results.md) collects the results of all
five side by side.

* [Lasso](tutorials/lasso.html)
* [Optimal control](tutorials/control.html)
* [Network flow](tutorials/network.html)
* [Nonnegative least squares](tutorials/nnls.html)
* [Portfolio](tutorials/portfolio.html)
