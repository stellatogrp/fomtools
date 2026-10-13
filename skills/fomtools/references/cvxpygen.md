# CVXPYgen Integration Reference

## Overview

`fomtools.cvxpygen` connects FOMTools codegen to CVXPYgen: model a parametric
problem in CVXPY, pick a first-order method, and get a single embedded C project
where CVXPYgen handles canonicalization and FOMTools generates the solver.

## Requirements

```bash
uv pip install 'fomtools[cvxpygen]'   # cvxpy, cvxpygen, osqp
```

Building the Python wrapper needs `cmake` and `pip`. Generating the solver
body needs `FOMTOOLS_CODEGEN_AGENT=1` or an explicit `driver=`.

## Usage

```python
import cvxpy as cp
import numpy as np
import fomtools.cvxpygen as fcpg
from fomtools.codegen import CodegenConfig, TerminationSpec

# 1. Model a parametric QP in CVXPY
n, m = 3, 2
A = np.random.randn(m, n)
q = cp.Parameter(n, name="q")
q.value = np.zeros(n)
x = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
    [A @ x == np.zeros(m), x <= 1.0, -1.0 <= x],
)

# 2. Generate canonicalization + solver
result = fcpg.generate_solver(
    problem,
    "out_qp",
    fom="admm",
    param_bounds={q: (-1.0, 1.0)},
    config=CodegenConfig(termination=TerminationSpec(max_iters=2000)),
)

# 3. Solve
q.value = np.array([0.5, -0.2, 0.1])
problem.solve(method="CPG")
```

Use a *relative* `code_dir` -- CVXPYgen imports `<code_dir>.cpg_solver`.

## First-Order Methods

| `fom=` | Description | Options (`fom_options=`) |
|--------|-------------|------------------------|
| `"admm"` | ADMM (default). Sparse LDLt + projection. Works for any QP. | `rho`, `sigma`, `alpha`, `rho_eq_scale` |
| `"pdhg"` | PDHG / Chambolle-Pock. SpMV only, no factorization. For LPs (zero quadratic term). | `step_scale` |
| Custom callable | `f(qp, **fom_options) -> FomSpec` | User-defined |

Custom builders receive a `QPData` (canonical P, q, A, equality values b,
inequality bounds u_ineq, parametric flags, propagated bounds).

## Parameter Bounds

Every runtime parameter needs bounds for verification:

```python
param_bounds={param_or_name: (lb, ub)}
```

Bounds broadcast against the parameter shape. Parameters without bounds fall
back to `value +/- bound_margin` (default 1.0) with a warning.

## Generated Output

```
out_qp/
├── cpg_solver.py             # cpg_solve; problem.solve(method='CPG')
├── c/
│   ├── src/cpg_solve.c       # CVXPYgen canonicalization
│   ├── src/cpg_workspace.c   # canonical data, settings, results
│   └── solver_code/          # FOMTools embedded solver
│       ├── src/fom_solver.c
│       ├── src/cpg_fom_bridge.c   # Canon_Params -> FomParams -> fom_solve
│       └── test/
```

### C API

```c
cpg_update_q(values);     // update parameter
cpg_solve();              // run canonicalization + FOM
// results in CPG_Result
cpg_set_solver_max_iter(N);
cpg_set_solver_tol(tol);
```

## Limitations (v1)

- Parameters may enter only canonical **vectors** (q, l, u), not P or A matrices.
  Parameter-dependent P/A raises `NotImplementedError`.
- The FOM must be **stationary** (one iteration repeated). Schedules not yet wired.
- Embedded C target only (float64).
- One solver instance per binary.
- Residuals are `||x_k - x_{k-1}||_inf`, not primal/dual constraint violations.
- Each `cpg_solve()` cold-starts the FOM.
