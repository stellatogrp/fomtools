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
- `fom="relaxed_admm"`: the same operator with an over-relaxation that *varies
  with the iteration index*, ramping from `alpha_start` to `alpha_final` over
  `ramp_iters`. Generated down the scheduled path below. Ramping is a
  heuristic, not an acceleration guarantee — it helps on some problems and not
  others — but any sequence inside `(0, 2)` converges, and the solver is the
  same size and carries the same state as the plain builder.
- `fom="pdhg"`: PDHG / Chambolle–Pock. Matrix-vector products only — no
  factorization — for problems with a zero quadratic term (LPs). Option:
  `step_scale`.
- **Custom**: pass any callable `f(qp, **fom_options) -> spec`. It receives
  a `fomtools.cvxpygen.qpdata.QPData` (canonical `P, q, A`, equality values
  `b`, inequality bounds `u_ineq`, and which pieces are parametric, with
  propagated bounds) and records the method in a `Verifier` — see
  `fomtools/cvxpygen/builders.py` for the built-ins as templates.

## Equilibration

By default (`auto_equilibrate=True`) the canonical QP is Ruiz-equilibrated
before the FOM is built, as OSQP does: diagonal scalings `D` (variables) and
`E` (constraint rows) from ten passes over the KKT matrix `[[P, A'], [A, 0]]`,
and a cost scaling `c`. The solver iterates on `c D P D`, `c D q`, `E A D`,
`E l`, `E u`; the bridge scales the parameters in and the solution out
(`x = D x~`, `y = E y~ / c`), so `problem.solve(method='CPG')` returns the
original problem's primal and dual solution, and the reported residuals are
those of the original problem. The parameter bounds the generated code is
verified on are scaled exactly. Pass `auto_equilibrate=False` to build the FOM
on the canonical QP unchanged (`fomtools.cvxpygen.scaling` has the routines).

The cost scaling `c` is fixed at generation time, so it is computed from the
largest `|q|` the parameter bounds allow, not from the current `q`. Very wide
`param_bounds` (or a large `bound_margin`) on parameters entering `q` can
therefore shrink `c` and flatten `P`: the solution is still correct, but
convergence can be slower than without equilibration. The parameter bounds now
affect conditioning as well as verification, so keep them as tight as the
application allows.

## Code optimization

`code_optimization=True` runs the code optimization phase after translation
(see the code generation tutorial): the coding agent rewrites the translated
solver for speed, and the rewrite is kept only if it passes verification again
and is benchmarked faster. Its budget comes from `config`
(`agent_optimize_timeout_s`, `agent_optimize_max_turns`).

## The three generation paths

A builder declares which path it wants with `@fom_builder(...)`, mirroring the
dispatch in [`generate_code`](codegen.md). Undecorated callables are `rolled`,
so builders written against the earlier API keep working unchanged.

| Path | Builder returns | Emits | Runtime settings |
|---|---|---|---|
| `rolled` (default) | `FomSpec` with `next_state` | one iteration, repeated `max_iters` times | `max_iter`, `tol`, `check_termination` |
| `scheduled` | `ScheduledFomSpec` | one iteration plus a table per varying coefficient | same, with `max_iter` **capped** at the table length |
| `unrolled` | `FomSpec` with `next_state=None` and `iterates` | the whole recorded run as one pass | none |

**Rolled** is the stationary case: the builder records one iteration and the
generated loop repeats it. Pass `iterates` (the successive recorded iterates)
and `param_values` as well, and generation *checks* that repeating that one
iteration really does reproduce the recording — both built-in stationary
builders do, so a coefficient that secretly varied with `k` is caught rather
than silently frozen at its first recorded value.

**Scheduled** is for a coefficient that varies on purpose. The builder becomes a
function of its varying scalars, extraction probes it at several values and
recovers each coefficient block as an affine function of them, and the scalars
are emitted as tables read at the loop counter. The solver stays one block long
however far it runs. Two restrictions are worth knowing before you write one:
the scalar must enter the recorded expressions **linearly**, and it may not
scale a **constant offset** — so a method that pulls each iterate toward a fixed
anchor cannot be expressed this way whenever a canonical constant (for instance
a non-parametric equality right-hand side) reaches the state update.

**Unrolled** generates the run exactly as recorded, which accepts arbitrary
per-iteration coefficients, a step type that changes with the iteration, and
anchored methods. It needs `allow_unrolled=True`, because the emitted program
grows with the horizon — large for an embedded target — and the iteration count
is fixed at generation time. Generation rolls the run back up automatically when
repeating one iteration is *verified* to reproduce it, so writing a stationary
method as a loop costs nothing. Record `iterates` either way you like: the
recorded iterations alone, or those with the initial iterate kept in slot 0, as
an `IterateTrajectory` holds them. A leading initial iterate is recognized and
dropped, so the emitted run and the iteration count reported through
`solver_stats` are the recorded iterations in both cases.

```python
from fomtools.codegen.extract_scheduled import Recording
from fomtools.cvxpygen.builders import ScheduledFomSpec, fom_builder

@fom_builder("scheduled")
def my_fom(qp, **options):
    def build(step):                     # one keyword per schedule
        ...                              # record one iteration using `step`
        return Recording(v, next_state=..., state=..., params=...)

    return ScheduledFomSpec(
        build=build,
        schedules={"step": lambda k: 1.0 / (k + 2.0)},   # 0-based counter
        name="my_fom", primal_offset=0, primal_dim=qp.n,
        dual_offset=qp.n + qp.m, dual_dim=qp.m,
        param_copy=("q",),
    )
```

`build_relaxed_admm_qp` in `fomtools/cvxpygen/builders.py` is the same thing
filled in, and `tests/codegen/test_cvxpygen_scheduled.py` shows how to check
such a builder against CVXPY without generating any C.

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
│       ├── SPEC.md, spec.json, data.npz   # the IR the agent worked from
│       └── test/                  # verification harness
└── ...
```

In C, `cpg_update_<param>(...)` then `cpg_solve()` runs the canonicalization
update and the embedded FOM; results land in `CPG_Result` as with any CVXPYgen
solver. The reported residuals are the OSQP-style
`||Ax - clamp(Ax, l, u)||_inf` (primal) and `||Px + q + A'y||_inf` (dual),
computed by the bridge from the canonical data and the returned solution.

### Runtime settings

Which `cpg_set_solver_*` functions exist depends on the generation path, because
a solver only advertises settings it can actually honor:

- **Rolled**: `cpg_set_solver_max_iter`, `cpg_set_solver_tol` and
  `cpg_set_solver_check_termination` (check every N iterations, 0 = disabled).
  All three are read by the bridge on every solve, so they can be tuned without
  regenerating.
- **Scheduled**: the same three, except that `max_iter` is capped in the bridge
  at the length of the coefficient tables. Past their end the emitted accessors
  clamp to the last entry, which would freeze the coefficients rather than
  continue the method, so raising `max_iter` beyond the generated horizon has no
  effect instead of quietly running a different algorithm.
- **Unrolled**: none. The body is the whole run executed once, so there is no
  iteration count to bound, no per-iteration residual to compare against a
  tolerance, and no interval to check at. CVXPYgen emits no `Canon_Settings`
  struct at all for such a solver, and `CPG_Info.iter` reports the number of
  iterations the straight-line body performs.

## Limitations

- Parameters may enter only the canonical **vectors**: the objective's linear
  term and constraint right-hand sides. Parameter-dependent `P`/`A` entries
  raise `NotImplementedError` (FOMTools bakes matrices into the generated
  code).
- Scheduled coefficients must enter the recorded expressions linearly and may
  not scale a constant offset; see the three paths above.
- Targets: the embedded C target and the large-scale OpenMP / CUDA targets, all
  with `dtype='float64'` (the canonical CVXPYgen data is double precision).
- One solver instance per binary (the bridge globals are unprefixed, as in
  CVXPYgen's statically-allocated backends).
- With equilibration on, a runtime `tol` compares successive iterates of the
  *scaled* problem; the reported residuals are always those of the original
  problem.
- Each `cpg_solve()` cold-starts the FOM from its initial state
  (`fom_update_params` resets the iterate), so an anchored method's anchor is
  always the zero vector.
