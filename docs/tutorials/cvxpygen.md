# CVXPYgen integration: model in CVXPY, solve with a generated FOM

`fomtools.cvxpygen` connects the FOMTools code generator to
[CVXPYgen](https://github.com/cvxgrp/cvxpygen): you model a parametric problem
in CVXPY, pick (or define) a first-order method in FOMTools, and get a single
embedded C project in which

- **CVXPYgen** generates the fast canonicalization code — updating a CVXPY
  `Parameter` triggers only the sparse affine update of the canonical QP data
  (`P, q, A, l, u`), exactly as it does for its other solver backends, and
- **FOMTools** generates the solver — a verified, statically allocated,
  division-free embedded C implementation of the chosen FOM, produced by the
  LLM coding agent and checked against the Python reference oracle.

CVXPYgen is extended from the outside (no fork): the integration registers a
solver named `'FOMTOOLS'` in CVXPYgen's dispatch at runtime.

## Requirements

```bash
pip install 'fomtools[cvxpygen]'   # cvxpy, cvxpygen, osqp
```

Building the Python wrapper additionally needs `cmake` and `pip` available in
the environment (CVXPYgen builds the extension module with
`python -m pip install`). Generating the solver body requires a coding-agent
driver: set `FOMTOOLS_CODEGEN_AGENT=1` to use the Claude agent, or pass an
explicit `driver=`.

## The seamless flow

```python
import cvxpy as cp
import numpy as np
import fomtools.cvxpygen as fcpg
from fomtools.codegen import CodegenConfig, TerminationSpec

# 1. Model a parametric problem in CVXPY (must reduce to a QP).
n, m = 3, 2
A = np.random.randn(m, n)
q = cp.Parameter(n, name="q")
q.value = np.zeros(n)
x = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
    [A @ x == np.zeros(m), x <= 1.0, -1.0 <= x],
)

# 2. Generate canonicalization (CVXPYgen) + solver (FOMTools) code.
result = fcpg.generate_solver(
    problem,
    "out_qp",                          # CVXPYgen layout: c/, cpg_solver.py, ...
    fom="admm",                        # or "pdhg", or a custom builder
    param_bounds={q: (-1.0, 1.0)},     # verified parameter range
    config=CodegenConfig(termination=TerminationSpec(max_iters=2000)),
)
assert result.success

# 3. Solve through the generated code (registered automatically).
q.value = np.array([0.5, -0.2, 0.1])
problem.solve(method="CPG")
```

Note: use a *relative* `code_dir` — CVXPYgen registers the wrapper by importing
`<code_dir>.cpg_solver` relative to the working directory.

## Choosing the first-order method

- `fom="admm"` (default): ADMM. One sparse LDL^T factorization of
  the KKT matrix is computed at generation time; each iteration is a
  triangular solve plus a projection. Works for any QP. Options (via
  `fom_options=...`): `rho`, `sigma`, `alpha`, `rho_eq_scale`.
- `fom="pdhg"`: PDHG / Chambolle–Pock. Matrix-vector products only — no
  factorization — for problems with a zero quadratic term (LPs). Option:
  `step_scale`.
- **Custom**: pass any callable `f(qp, **fom_options) -> FomSpec`. It receives
  a `fomtools.cvxpygen.qpdata.QPData` (canonical `P, q, A`, equality values
  `b`, inequality bounds `u_ineq`, and which pieces are parametric, with
  propagated bounds) and records one iteration in a `Verifier` — see
  `fomtools/cvxpygen/builders.py` for the two built-ins as templates.

## Parameter bounds

FOMTools verifies the generated code by sampling parameters, so every runtime
parameter needs bounds. Pass `param_bounds={param_or_name: (lb, ub)}`; bounds
broadcast against the parameter shape. Parameters without bounds fall back to
their current value ± `bound_margin` (default 1.0) with a warning. Bounds are
propagated to the canonical vectors exactly through CVXPYgen's affine
parameter maps.

## What gets generated

```
out_qp/
├── cpg_solver.py            # cpg_solve; registered as problem.solve(method='CPG')
├── c/
│   ├── src/cpg_solve.c      # CVXPYgen: canonicalization + parameter updates
│   ├── src/cpg_workspace.c  # CVXPYgen: canonical data, settings, results
│   └── solver_code/         # FOMTools: the embedded solver
│       ├── src/fom_solver.c       # the verified FOM loop
│       ├── src/fom_{data,ops,linsolve}.c
│       ├── src/cpg_fom_bridge.c   # Canon_Params -> FomParams -> fom_solve
│       └── test/                  # spec + verification harness
└── ...
```

In C, `cpg_update_<param>(...)` then `cpg_solve()` runs the canonicalization
update and the embedded FOM; results land in `CPG_Result` as with any CVXPYgen
solver. `cpg_set_solver_max_iter(...)` and `cpg_set_solver_tol(...)` adjust
the FOM termination at runtime; the check interval
(`TerminationSpec(check_termination=...)`: check every N iterations,
0 = disabled) is fixed at code-generation time.

## Limitations (v1)

- Parameters may enter only the canonical **vectors**: the objective's linear
  term and constraint right-hand sides. Parameter-dependent `P`/`A` entries
  raise `NotImplementedError` (FOMTools bakes matrices into the generated
  code).
- The FOM must be **stationary**: the builder behind `fom=` records one
  iteration, which the generated loop then repeats. A coefficient that changed
  with the iteration index would be frozen at its recorded value. The two
  built-in builders (`admm`, `pdhg`) are stationary by construction — their step
  sizes and relaxation factor are computed once, offline — and a custom builder
  must be too. Schedules (see the [codegen tutorial](codegen.md)) are not yet
  wired through this front end.
- Embedded C target only (`Version.EMBEDDED`, `Backend.C`, `float64`).
- One solver instance per binary (the bridge globals are unprefixed, as in
  CVXPYgen's statically-allocated backends).
- The reported primal/dual residuals are both the FOM termination metric
  `||x_k - x_{k-1}||_inf`, not primal/dual constraint-violation residuals.
- Each `cpg_solve()` cold-starts the FOM from its initial state
  (`fom_update_params` resets the iterate).
