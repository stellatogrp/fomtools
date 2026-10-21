---
name: fomtools
description: |-
  Reference and guide for building, verifying, and code-generating first-order optimization methods with FOMTools.
  Use this skill whenever: the user mentions FOMTools, fomtools, FOM verification, first-order methods, convergence verification, ISTA, Douglas-Rachford, PDHG, ADMM, OSQP, proximal operators, projected gradient, codegen for optimization solvers, CVXPYgen integration, embedded C solver generation, or asks about verifying iterative algorithms, worst-case analysis of optimization methods, MIP/SDP verification of FOMs, or generating low-level solvers from Python.
  Also use when the user is working in a fomtools codebase (imports from `fomtools`, uses `Verifier`, `LinearExpression`, step operators, or the codegen pipeline).
license: Apache-2.0
---

# FOMTools

FOMTools is a Python toolbox for **performance analysis of first-order optimization methods** and **code generation of embedded solvers**. You define a FOM as a sequence of steps, bound finite-iteration worst-case performance via MIP or SDP, and optionally generate a fast C/CUDA solver from it.

## When to Use This Skill

- Writing or modifying code that uses FOMTools
- Setting up a verification problem for a first-order method
- Generating an embedded C solver from a verified FOM
- Integrating FOMTools with CVXPYgen
- Understanding the architecture or API of FOMTools
- Debugging verification or codegen issues

## Quick Start

Follow the executable example in `docs/index.md`, which is tested directly.

```python
import numpy as np
import fomtools as ft

K, n, eta = 5, 3, 0.01
A = np.random.default_rng(0).normal(size=(n, n))
P = A @ A.T

vp = ft.Verifier()                      # HiGHS by default
q = vp.parameter(n, lb=-1, ub=1)        # problem parameter
z = vp.iterate(n, lb=0, ub=0)           # z^0 = 0
for k in range(K):
    z_prev, z = z, vp.relu(z - eta * (P @ z + q))
vp.maximize(ft.norm(z - z_prev, np.inf))
bound = vp.solve()                      # q.value is the worst case
```

Use `vp.iterate`/`vp.parameter` (`add_iterate`/`add_param` are deprecated aliases),
`vp.maximize(ft.norm(expr, 1 | np.inf))` or `vp.maximize(ft.sum_squares(expr))` for the
squared two-norm (an MIQP with `method="mip"`, so it needs `solver="gurobi"`;
`ft.norm(expr, 2)` raises), `expr.value` for the worst case found by a MIP solve, and
`solve(convex_hull_cuts=True)` for MIP hull cuts. Solvers are silent by default. Pass `verbose=True` to
`Verifier` or `solve` for solver logs.
MIP returns a finite best upper bound and raises when none is available.
SDP accepts only accurate optimal termination. Numerical tolerances still apply.
Use `post_process` only with certified bounds for all successive infinity-norm
residuals from reference to target; a final-step bound alone is insufficient.

## Architecture Overview

The verification pipeline follows: **Step -> StepDecomposer -> Canonicalizer -> Solver**.

### Verification Methods

| `method=` | Canonicalizer | Solvers | Use case |
|-----------|--------------|---------|----------|
| `"mip"` | `MIP` | HiGHS (default), Gurobi | Exact formulations for verification via mixed-integer programming |
| `"sdp"` | `SDP` | Clarabel, CVXPY/Clarabel, cuClarabel | Semidefinite relaxation (supports RLT/triangle cuts) |
| `"simulator"` | Simulator | (none) | Sampled residuals and distributional risk bounds |

### Step Operators

Call as `vp.<name>(arg, ...)` (`vp.step.<name>` is deprecated). Options after `arg` are keyword-only:

| Step | Method | Description |
|------|--------|-------------|
| `relu` | `vp.relu(x)` | ReLU activation / projection onto non-negative orthant |
| `soft_threshold` | `vp.soft_threshold(x, lambd=t)` | Soft thresholding (proximal of L1 norm) |
| `satlin` | `vp.satlin(x, l=l, u=u)` | Box projection / clipping |
| `implicit_linear` | `vp.implicit_linear(x, output_mat=M)` | Implicit linear solve (used in ADMM/KKT) |

### Key Classes

- **`Verifier`**: Main orchestrator -- manages iterates, params, steps, constraints, objectives, solving, and codegen.
- **`LinearExpression`**: Symbolic linear combination of variables/params. Supports `@` (matrix multiply), arithmetic, and `stack`. Every iterate and param is a `LinearExpression`.
- **`Constraint`**: `EqualityConstraint`, `InequalityConstraint`, `ComplementarityConstraint`, `QuadraticConstraint`.

For detailed API signatures and parameters, read `references/api.md`.

## Code Generation

FOMTools generates OSQP-style embedded C solvers from verified FOMs. An LLM coding agent writes the solver and iterates against a Python reference oracle until checked on sampled inputs.

### Basic Codegen

```python
vp = ft.Verifier(method="simulator", obbt_solver=None)
b_param = vp.parameter(m, lb=b - 1.0, ub=b + 1.0)
z = vp.iterate_trajectory(K + 1)
z[0] = vp.iterate(n, lb=2, ub=2)
for k in range(1, K + 1):
    z[k] = vp.soft_threshold(At @ z[k - 1] + Bt @ b_param, lambd=t)

result = vp.codegen("out/ista", iterates=z, name="ista")
```

### Rolled vs Unrolled

- `vp.codegen(...)` records the full run, then automatically rolls it back up if repeating one iteration reproduces the recording. Use `roll=False` to force unrolling.
- `vp.codegen_rolled(...)` takes a single stationary iteration and a configurable horizon. Use when you want a stopping rule via `TerminationSpec`.

### Iteration-Varying Coefficients (Schedules)

For accelerated methods (FISTA, Nesterov) with coefficients that change per iteration:

1. **Record the run** with `codegen()` -- works directly but emits one block per iteration.
2. **Declare a schedule** via `generate_code(build_fn, None, out_dir, schedules={"gamma": momentum_fn})` -- emits a single rolled loop with a lookup table.

### Stopping Criteria

```python
from fomtools.codegen import CodegenConfig, TerminationKind, TerminationSpec

config = CodegenConfig(
    termination=TerminationSpec(
        kind=TerminationKind.INF_NORM_CHANGE,
        max_iters=2000,
        tol=1e-6,
        check_termination=1,  # check every N iterations; 0 disables early exit
    ),
)
result = vp.codegen_rolled("out/ista", next_state=z[1], config=config)
```

To stop on the verified metric itself, set the objective on the recording and use
`TerminationKind.OBJECTIVE` (checked every 25 iterations by default and at the
last one; `ord=2` is the squared norm, as in verification):

```python
vp.maximize(ft.norm(z[K] - z[K - 1], np.inf))
config = CodegenConfig(
    termination=TerminationSpec(kind=TerminationKind.OBJECTIVE, max_iters=K, tol=1e-6),
)
result = vp.codegen("out/ista", iterates=z, config=config)  # stops by K if r^K <= tol is verified
```

The objective may read `w^K`, `w^{K-1}`, parameters, and ReLU steps recorded after
the loop (`max(Ax - u, 0)`); anything else raises `ObjectiveExtractionError` with
the fix. See `references/codegen.md`.

### Performance Profiling

```python
result = vp.codegen("out/ista", iterates=z, code_optimization=True, profile=True)
print(result.timing.summary())
```

### Targets

| Target | `version=` | `backend=` | Notes |
|--------|-----------|-----------|-------|
| Embedded C | `"embedded"` | `"c"` | Default. Statically sized, malloc-free, division-free |
| OpenMP | `"large_scale"` | `"openmp"` | Multithreaded CPU |
| CUDA | `"large_scale"` | `"cuda"` | GPU kernels |

### Generated Output Structure

```
out/
├── include/   fom_solver.h, fom_workspace.h, fom_data.h
├── src/       fom_solver.c, fom_data.c, fom_ops.c, fom_linsolve.c
├── test/      SPEC.md, spec.json, data.npz, verify_main.c
├── python/    fom_solver.py (ctypes wrapper)
└── Makefile   (make verify / make lib / make bench / make profile)
```

### Using Generated Solvers

**From C (OSQP-style ABI):**
```c
FomWorkspace work; FomSettings s;
fom_set_default_settings(&s);
fom_setup(&work, &s);
fom_update_params(&work, &params);
fom_solve(&work);
// work.solution.x holds the result
```

**From Python:**
```python
from fom_solver import Solver
solver = Solver()
out = solver.solve(p0=b)  # returns {"x", "iter", "residual", "status"}
```

For detailed codegen documentation including accumulating methods, schedules, linear solve strategies, and the CLI, read `references/codegen.md`.

## CVXPYgen Integration

Model in CVXPY, solve with a generated FOM:

```python
import cvxpy as cp
import fomtools.cvxpygen as fcpg
from fomtools.codegen import CodegenConfig, TerminationSpec

x = cp.Variable(n)
q = cp.Parameter(n, name="q")
problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + q @ x), [...])

result = fcpg.generate_solver(
    problem, "out_qp",
    fom="admm",  # or "pdhg", or custom builder
    param_bounds={q: (-1.0, 1.0)},
    config=CodegenConfig(termination=TerminationSpec(max_iters=2000)),
)
problem.solve(method="CPG")
```

For full CVXPYgen integration docs, read `references/cvxpygen.md`.

## Installation

```bash
uv pip install -e .               # basic
uv pip install -e ".[dev]"        # development (pytest, ruff)
uv pip install -e ".[codegen]"    # code generation
uv pip install -e ".[cvxpygen]"   # CVXPYgen integration
uv pip install -e ".[docs]"       # documentation (mkdocs)
```

Requires Python 3.12+. Core dependencies include `numpy`, `scipy`, `highspy`, and `cvxpy`. Gurobi is optional and requires a suitable license. `torch`/`jax` are optional neural-network dependencies.

## CLI

```bash
fomtools-codegen --example ista --out out/ista --driver agent
fomtools-codegen --example mpc  --out out/mpc  --driver agent
fomtools-codegen --example dr   --out out/dr   --scaffold-only
fomtools-codegen --module mymodule.py --out out/custom --driver agent
```

## Key Invariants

1. **Never form or store a matrix inverse.** Affine solves use sparse LDLt (SPD/quasidefinite), LU (general), or CG (indirect, SPD).
2. **The oracle is the arbiter.** Generated code must match `ReferenceOracle` to tolerance over random samples.
3. **The agent owns exactly one file** (`src/fom_solver.c[.cu]`). Everything else is regenerated.
4. **Embedded target is malloc-free and division-free.**

## Common Patterns

### Projected Gradient Descent
```python
z[k] = vp.relu(z[k-1] - step_size * (P @ z[k-1] + q))
```

### ADMM / Douglas-Rachford
```python
u = vp.implicit_linear(z[k-1] - q, output_mat=kkt)
ut = vp.relu(2*u - z[k-1], proj_ranges=(n, dim))
z[k] = z[k-1] + ut - u
```

### Soft Thresholding (ISTA/Lasso)
```python
z[k] = vp.soft_threshold(At @ z[k-1] + Bt @ b_param, lambd=t)
```

### Box-Constrained QP
```python
z[k] = vp.satlin(z[k-1] - step_size * grad, l=lb, u=ub)
```

## Source Code

GitHub: [stellatogrp/fomtools](https://github.com/stellatogrp/fomtools)
