## Tutorials
These tutorials show many different ways to use FOMTools. The notebooks are
[marimo](https://marimo.io) files; their sources are in
[`docs/tutorials` on GitHub](https://github.com/stellatogrp/fomtools/tree/main/docs/tutorials).

* [NNLS](tutorials/nnls_verification.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/nnls_verification.py))
* [Code generation](tutorials/codegen.md)
* [CVXPYgen integration](tutorials/cvxpygen.md)
* [CVXPYgen MPC example](tutorials/cvxpygen_mpc.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/cvxpygen_mpc.py))

### Code generation speedups
Each of these notebooks builds a FOM, generates an embedded C solver for it, and
times the generated solver against direct Python. The
[summary table](tutorials/codegen_speedup_results.md) collects the results of all
five side by side.

* [Lasso](tutorials/lasso.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/lasso.py))
* [Optimal control](tutorials/control.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/control.py))
* [Network flow](tutorials/network.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/network.py))
* [Nonnegative least squares](tutorials/nnls.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/nnls.py))
* [Portfolio](tutorials/portfolio.html) ([source](https://github.com/stellatogrp/fomtools/blob/main/docs/tutorials/portfolio.py))
