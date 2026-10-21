# FOMTools Code Generation Reference

## Overview

FOMTools generates OSQP-style embedded C solvers from verified first-order
methods. An LLM coding agent translates each step into C and iterates against a
Python reference oracle until numerically verified.

Entry point: `generate_code(verifier, next_state, out_dir, config=...)`
or `Verifier.codegen(...)` / `Verifier.codegen_rolled(...)`.

## CodegenConfig

```python
from fomtools.codegen import CodegenConfig, TerminationSpec, TerminationKind

config = CodegenConfig(
    version="embedded",        # "embedded" or "large_scale"
    backend="c",               # "c", "openmp", or "cuda"
    solve_method=None,         # force "LDLT"/"LU"/"CG"; None = auto
    termination=TerminationSpec(
        kind=TerminationKind.INF_NORM_CHANGE,  # or MAX_ITERS, or OBJECTIVE (below)
        max_iters=1000,
        tol=1e-6,
        check_termination=1,   # check every N iters; 0 disables early exit
    ),                         # (default 1; 25 for OBJECTIVE)
    tol_verify=1e-6,           # float64 verification tolerance
    float_type="float64",      # "float64" or "float32" (tol_verify=1e-4)
)
```

## generate_code

```python
from fomtools.codegen import generate_code

# From a full recording (auto-rolls when stationary):
result = generate_code(verifier, None, "out/ista",
    config=config, iterates=z, name="ista",
    code_optimization=False, profile=False, roll=True)

# From a single stationary iteration:
result = generate_code(verifier, next_state, "out/ista",
    config=config, name="ista")

# With scheduled (iteration-varying) coefficients:
result = generate_code(build_fn, None, "out/fista",
    schedules={"gamma": momentum_fn}, config=config)
```

### Stopping on the verification objective

`TerminationKind.OBJECTIVE` stops at the first checked iteration `k` (every
`check_termination`-th, default 25, and always the last) with `r^k <= tol`, where
`r^k` is the objective set with `Verifier.set_objective_norm(expr, ord)` in that
norm (`ord=2` is the squared two-norm). With `max_iters = K` and `r^K <= tol`
verified over the parameter set, the solver is guaranteed to stop by `K` with
`FOM_SOLVED`; `solution.residual` is `r^k` at the stop.

- The objective is written at iteration `K` over `w^K`, `w^{K-1}`. It is mapped
  exactly onto the loop's new state, current state and parameters; written on the
  extracted iteration (`next_state` frame) it may also read that iteration's step
  outputs. One-sided terms are ReLU steps recorded after the loop
  (`v.step.relu(A @ x_K - u)`), one ReLU per row and nothing else in that row.
- An objective written after the loop needs the tape: `codegen(iterates=...)`, or
  `codegen_rolled(..., iterates=...)`.
- Not representable (raises `ObjectiveExtractionError` with the fix): an iterate
  older than `w^{K-1}` (carry it in the state), a non-leaf accumulated iterate at
  `K` (use `iterate_trajectory`), a parameter the iteration never reads, nested or
  mixed ReLU rows, other nonlinear steps.
- Scheduled builds set the objective inside `build`; coefficients that follow the
  schedule are fitted like any other. An unrolled run evaluates it once at the end.
- Generated as `fom_objective(it, state, new_state[, params][, temps])` in its own
  non-editable `src/fom_objective.c` (`fom_ops.cu` on CUDA); `make verify` checks the stopping iteration, value and status, also
  at `check_termination` 1 and 7 and a capped `max_iter`.
- CLI: `--tol 1e-6 --objective [--check-termination N]`.

### Recording vs Rolled vs Scheduled

| Path | Input | Result |
|------|-------|--------|
| Recording | `verifier` + `iterates` (list of all z[k]) | Auto-rolls if possible; unrolled otherwise |
| Rolled | `verifier` + `next_state` (one iteration) | Always rolled; horizon from config |
| Scheduled | `build_fn` + `schedules` dict | Rolled with lookup table for varying coefficients |

## CodegenResult

```python
result.success      # bool: did the agent produce a verified solver?
result.ir           # FomIR: the intermediate representation
result.ir.unrolled  # bool: True if the body has >1 block
result.ir.body      # list of iteration blocks
result.timing       # TimingReport (if profile=True)
result.timing.summary()  # formatted timing string
```

## Accumulating Methods

Douglas-Rachford / ADMM carry state through expressions, not step outputs:

```python
z = vp.iterate_trajectory(K + 1)  # handles expression collapse automatically
z[0] = vp.add_iterate(n, lb=z0, ub=z0)
for k in range(1, K + 1):
    u = vp.step.implicit_linear(z[k-1] - q, output_mat=kkt)
    ut = vp.step.relu(2*u - z[k-1], proj_ranges=(n, dim))
    z[k] = z[k-1] + ut - u
```

`iterate_trajectory` with `collapse_expr="auto"` detects growing expression
support and collapses it. Without it, leaf support grows as 3, 5, 7, 9...
and generating the run takes quadratic matrix-vector products.

## Schedules (Iteration-Varying Coefficients)

For FISTA momentum, Nesterov coefficients, Halpern anchors:

```python
def momentum(k):
    """0-based loop index. k=0 is the first iteration."""
    beta = 1.0
    for _ in range(k + 1):
        beta_next = (1 + np.sqrt(1 + 4*beta**2)) / 2
        gamma, beta = (beta - 1) / beta_next, beta_next
    return gamma

def build(gamma):
    vp = Verifier(method="simulator", obbt_solver=None)
    b_param = vp.add_param(m, lb=b-1, ub=b+1)
    y_prev = vp.add_iterate(n, lb=2, ub=2)
    y_curr = vp.add_iterate(n, lb=2, ub=2)
    extrap = y_curr + gamma * (y_curr - y_prev)
    y_next = vp.step.soft_threshold(At @ extrap + Bt @ b_param, lambd=t)
    return Recording(vp, next_state=[y_curr, y_next], state=[y_prev, y_curr])

generate_code(build, None, "out/fista", schedules={"gamma": momentum}, config=config)
```

The build function must produce expressions **affine** in the schedule scalar.
FOMTools probes at several values to recover M(gamma) = M0 + gamma * M1.
A non-affine dependency raises `ScheduleFitError`.

### What schedules cannot vary

- Factorized system matrices (e.g., ADMM rho that changes with k)
- Activation scalars (soft-threshold level, saturated-linear bounds)
- Constant offsets

## Linear Solves

Never form or store a matrix inverse. Codegen uses:

- **Sparse LDLt**: for SPD/quasidefinite systems (ADMM/OSQP KKT).
  Uses reverse Cuthill-McKee fill-reducing ordering.
- **LU**: for general non-symmetric systems.
- **CG**: indirect method for large/GPU problems (SPD only).

Generated code uses **division-free** triangular solves (diagonals stored as
reciprocals). Override with `CodegenConfig(solve_method="LDLT"/"LU"/"CG")`.

## Agent System

- `AgentDriver`: interface for the LLM coding agent.
- `StubAgentDriver`: drops a known-good golden solver for CI testing (no LLM needed).
- Real agents: Claude, Codex, Pi, or custom CLI agents from `~/.config/fomtools/agents.yaml`.
- Enable real agent: `FOMTOOLS_CODEGEN_AGENT=1`
- Select backend: `FOMTOOLS_CODEGEN_BACKEND=claude`

### Agent Phases

- **Translation phase**: translate IR to C -> verify against oracle (gate 1)
- **Code optimization phase** (optional, `code_optimization=True, profile=True`): profile -> optimize hottest region -> re-verify -> repeat until gains level off, in up to `agent_optimize_rounds` (5) rounds, each a fresh agent run on the best solver so far. Each round's rewrite is kept only if it passes the same check (gate 2) and is faster head to head; a round that fails, is slower, or gains less than `agent_optimize_min_gain` (2%) ends the phase, reverting to the best solver so far. `run_code_optimization` runs the phase on a copy of a translated solver.

## CLI Reference

```bash
fomtools-codegen --help

# Built-in examples:
fomtools-codegen --example ista      --out out/ista  --driver agent
fomtools-codegen --example dr        --out out/dr    --scaffold-only
fomtools-codegen --example mpc       --out out/mpc   --driver agent
fomtools-codegen --example boxqp     --out out/boxqp --scaffold-only
fomtools-codegen --example cgqp      --out out/cgqp  --scaffold-only
fomtools-codegen --example pdhg_lp   --out out/pdhg  --scaffold-only
fomtools-codegen --example fista     --out out/fista  --scaffold-only
fomtools-codegen --example scheduled --out out/sched  --scaffold-only

# Custom FOM from a module:
fomtools-codegen --module mymodule.py --out out/custom --driver agent

# Options:
#   --scaffold-only    Emit everything except the solver body
#   --no-roll          Keep a recorded run unrolled
```

### Custom module format

The module defines `build_fom()` returning a dict:

| Keys | Path |
|------|------|
| `verifier`, `iterates` | full recorded run, auto-rolled |
| `verifier`, `next_state` | rolled solver from one iteration |
| `build`, `schedules` | rolled with iteration-varying coefficients |

Optional keys: `name`, `state`, `params`.

## FomIR (Intermediate Representation)

```python
from fomtools.codegen import extract_ir, extract_unrolled_ir, extract_scheduled_ir

ir = extract_ir(verifier, next_state)
ir = extract_unrolled_ir(verifier, iterates)
ir = extract_scheduled_ir(build_fn, schedules, config)
```

`FomIR` contains: `body` (list of iteration blocks), `params`, `state_dim`,
`unrolled` flag, and metadata about factorizations and sparse matrices.

## ReferenceOracle

The NumPy ground truth for verification:

```python
from fomtools.codegen import ReferenceOracle

oracle = ReferenceOracle(ir)
result = oracle.run(param_values, initial_state, max_iters)
```

Generated code must match oracle output to `config.tol_verify`.
