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

Other `generate_solver` keywords: `fom_options`, `driver`, `bound_margin`,
`allow_unrolled`, `prefix`, `wrapper`, `enable_settings`, `code_optimization`,
`auto_equilibrate` (default `True`).

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
cpg_set_solver_tol(tol);                 // INF_NORM_CHANGE / OBJECTIVE kinds only
cpg_set_solver_check_termination(N);     // likewise
```

### Stopping on OSQP's residuals

The built-in builders set `max(||Ax - clamp(Ax, l, u)||_inf, ||Px + q + A'y||_inf)`
(in original units under `auto_equilibrate`, via `QPData.unscale = (D, E, c)` set
by `scale_qp`) as the verification objective,
so `TerminationKind.OBJECTIVE` is OSQP's test with `eps_abs = tol`, `eps_rel = 0`,
`scaled_termination = False`:

```python
config = CodegenConfig(
    termination=TerminationSpec(kind=TerminationKind.OBJECTIVE, max_iters=4000, tol=1e-6)
)  # checked every 25 iterations (default) and at max_iter
result = fcpg.generate_solver(problem, "out_qp", fom="admm", config=config)
problem.solve(method="CPG", tol=1e-5, check_termination=10)
```

The bridge's `pri_res` / `dua_res` are the two halves of the objective at the stop.
Custom builders can call `builders.set_residual_objective(v, qp, exprs, x, y)`.

## Limitations (v1)

- Parameters may enter only canonical **vectors** (q, l, u), not P or A matrices.
  Parameter-dependent P/A raises `NotImplementedError`.
- Varying coefficients go through a scheduled builder (`relaxed_admm`) or an
  unrolled one (`allow_unrolled=True`).
- Embedded C and large-scale OpenMP / CUDA targets, float64 only.
- One solver instance per binary.
- Reported residuals are OSQP's primal / dual residuals, computed by the bridge.
- Each `cpg_solve()` cold-starts the FOM.
