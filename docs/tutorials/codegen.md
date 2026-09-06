# Code generation: from a verified FOM to an embedded C solver

Once you have verified a first-order method with a `Verifier`, FOMTools can
generate a fast, low-level implementation of it — in the spirit of
[OSQP codegen](https://osqp.org/docs/codegen/index.html) — together with a tiny
Python wrapper. The translation is performed by an LLM coding agent that writes
the target code and iterates against a Python reference oracle until the output
is numerically verified.

## Examples

Each of these notebooks builds a FOM, generates an embedded C solver for it, and
times the generated solver against direct Python. The
[summary table](codegen/speedup_results.md) collects the results of all five side
by side. Each notebook has a companion write-up covering the problem formulation
and the update rule it uses.

* Lasso — [notebook](codegen/lasso.html), [write-up](codegen/lasso.md)
* Optimal control — [notebook](codegen/control.html), [write-up](codegen/control.md)
* Network flow — [notebook](codegen/network.html), [write-up](codegen/network.md)
* Nonnegative least squares — [notebook](codegen/nnls.html), [write-up](codegen/nnls.md)
* Portfolio — [notebook](codegen/portfolio.html), [write-up](codegen/portfolio.md)

## Defining the FOM

Write the algorithm as a loop, exactly as you would to verify it, and keep the
iterate sequence. An [`iterate_trajectory`](#accumulating-methods) is the tidiest
way to hold it, but any list works:

```python
import numpy as np
from fomtools.verifier import Verifier

n, m = 2, 4
rng = np.random.default_rng(0)
A = rng.normal(size=(m, n))
b = rng.normal(size=(m,))
t = 1.0 / np.max(np.real(np.linalg.eigvals(A.T @ A)))
At = np.eye(n) - t * A.T @ A
Bt = t * A.T
K = 200

vp = Verifier(method="simulator", obbt_solver=None)
b_param = vp.add_param(m, lb=b - 1.0, ub=b + 1.0)

z = vp.iterate_trajectory(K + 1)
z[0] = vp.add_iterate(n, lb=2, ub=2)
for k in range(1, K + 1):
    z[k] = vp.step.soft_threshold(At @ z[k - 1] + Bt @ b_param, lambd=t)
```

Verify it however you normally would, then hand the same object to `codegen`:

```python
result = vp.codegen("out/ista", iterates=z, name="ista")
```

There is nothing to restructure: the program you generate from is the program you
verified. Each iteration carries its own coefficients, so the recording may do
whatever it likes with the iteration index — a decaying step size, momentum, a
step whose *type* changes with `k`, or a reference to an earlier iterate held
across the whole run, as in a Halpern anchor.

### Rolling back up

Generating the run literally would emit `K` blocks, which for a method that does
the same thing every iteration is wasteful. So before unrolling anything,
extraction pulls out iteration one on its own and asks whether repeating it
reproduces the recording. When it does, that compact IR is what gets emitted —
one block, whatever `K` was:

```python
result = vp.codegen("out/ista", iterates=z)
result.ir.unrolled     # False: the run rolled back up
len(result.ir.body)    # 1
```

Rolling changes the artifact and never the computation. The rolled solver keeps
the **recorded** horizon rather than any horizon you configure, and the check that
decides to roll is the proof it was safe: it replays the recorded tape and
compares it against the rolled loop, iteration by iteration, over several
parameter draws. Pass `roll=False` to emit the unrolled body regardless.

`result.ir.unrolled` tells you which you got.

### Generating from a single iteration instead

If you would rather supply one stationary iteration and choose the loop count
yourself, `codegen_rolled` does that:

```python
next_state = z[1]
result = vp.codegen_rolled(
    "out/ista", next_state,
    iterates=[z[k] for k in range(1, K + 1)],   # optional, but see below
    param_values={b_param: b},                  # any value in the parameter set
)
```

Here the iteration **must** be stationary, because extraction reads it once and
emits a loop that repeats it. A coefficient that changes with the iteration index
would be frozen at whatever value the recorded iteration happened to use, and the
generated solver would run a different algorithm than the one you verified.
Passing `iterates` and `param_values` has that checked rather than assumed; omit
them and you get a warning, because nothing else in the pipeline can catch the
mistake — the reference oracle is itself a rolled loop, so it agrees with a
wrongly rolled iteration by construction.

`codegen` needs no such argument: it generates the recording itself, and gates the
result against the tape using parameter values it draws from the declared box.

## Accumulating methods

Douglas–Rachford and ADMM carry their iterate through an expression rather than a
step output:

```python
for k in range(1, K + 1):
    u = vp.step.implicit_linear(z[k - 1] - q, output_mat=kkt)
    ut = vp.step.relu(2 * u - z[k - 1], proj_ranges=(n, dim))
    z[k] = z[k - 1] + ut - u        # not a step output
```

Written that way, each iterate references every leaf the previous one did plus the
new step outputs, so the expression grows down the trajectory — leaf support runs
3, 5, 7, 9, … — and generating the run would need a number of matrix-vector
products that grows with the *square* of the iteration count.

`iterate_trajectory` handles this for you. Its default `collapse_expr="auto"`
notices the growth after two consecutive increases and keeps each subsequent
iterate as its own variable, which restores linear growth. A method that merely
*refers* to an earlier iterate — a Halpern anchor — holds its support flat instead,
so it is left alone. Nothing to configure either way.

Forcing `collapse_expr=True` opts out of that, and generating the resulting
recording raises `UnrolledBlowupError` with the measured counts rather than
emitting a solver nobody wants. (A stationary accumulating method is unaffected
either way: it rolls back up before size matters.)

## Generating the solver

```python
result = vp.codegen("out/ista", iterates=z, name="ista")
assert result.success
```

### Stopping early on a tolerance

A solver that exits once the iterate stops moving is a property of the *rolled*
form, so it comes from `codegen_rolled`:

```python
from fomtools.codegen import CodegenConfig, TerminationKind, TerminationSpec

config = CodegenConfig(
    termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=2000, tol=1e-6),
)

result = vp.codegen_rolled("out/ista", next_state=next_state, config=config, name="ista")
assert result.success
```

Generating from a recorded run runs that run, so there is no horizon to stop short
of and `codegen` rejects a tolerance-based criterion rather than quietly ignoring
it. Reach for `codegen_rolled` when you want a stopping rule, or `codegen` when
you want the algorithm you recorded.

`TerminationSpec` also takes `check_termination` (named after OSQP's setting):
the residual is computed and tested only every `check_termination` iterations,
and `0` disables early exit entirely. The default is 1 — the generated check is
an infinity norm fused into the state copy, measured at 1–2.5% of iteration
time for direct-method FOMs, and checking less often overshoots the stopping
point by up to `check_termination - 1` full iterations. Raise it only for
matrix-free methods on large problems, where a single iteration is cheap
relative to a pass over the state.

This requires a coding agent. Enable the real Claude Agent SDK driver with
`FOMTOOLS_CODEGEN_AGENT=1` (requires authentication), or pass an explicit driver
(for example a `StubAgentDriver` wrapping a known-good source) for fully
deterministic, offline runs.

The emitted directory (`out/ista`) contains:

```
include/   fom_solver.h, fom_workspace.h, fom_data.h   (ABI + dims + data decls)
src/       fom_solver.c (the agent's solver), fom_data.c (all numeric data)
test/      SPEC.md, spec.json, data.npz, verify_main.c, verify_data.h
python/    fom_solver.py   (ctypes wrapper)
Makefile   (make verify / make lib)
```

## Using the generated solver

From C, via the OSQP-style ABI:

```c
FomWorkspace work; FomSettings s;
fom_set_default_settings(&s);
fom_setup(&work, &s);
fom_update_params(&work, &params);   /* set params.p0[...] */
fom_solve(&work);                    /* work.solution.x holds the solution */
```

Or from Python:

```python
import subprocess, sys
subprocess.run(["make", "lib"], cwd="out/ista", check=True)
sys.path.insert(0, "out/ista/python")
from fom_solver import Solver

solver = Solver()
out = solver.solve(p0=b)          # {"x", "iter", "residual", "status"}
```

## Performance: timing, profiling, and the optimization loop

Pass `run_phase_b=True, profile=True` (the default for the `fomtools-codegen`
CLI with `--driver agent`) to (a) measure the speed of the generated solver
against the Python reference, and (b) enter an iterative, profile-guided
optimization phase:

```python
result = vp.codegen(
    "out/ista", iterates=z, name="ista",
    run_phase_b=True, profile=True,
)
print(result.timing.summary())
```

After Phase A verifies, the orchestrator records a baseline (Python reference
time, the initial generated C time, and a profiling breakdown). Then the agent
runs a loop: **profile → optimize the hottest region → `make verify` (correctness
is re-checked every round) → `make bench` → repeat** until gains level off. A
change that breaks verification is never kept, and if the whole phase leaves the
solver regressed, the verified Phase-A solver is restored automatically.

`result.timing` (a `TimingReport`) holds `python_us`, `c_initial_us`,
`c_final_us`, the speedup factors (`speedup_codegen`, `speedup_optimization`,
`speedup_total`), and the final `profile` breakdown, and `.summary()` formats
them, e.g.:

```
Timing summary (per solve):
  Python reference (before codegen):   1328.250 us
  Generated C (initial, phase A):         1.753 us
  Generated C (optimized, phase B):       1.640 us
  Speedup from codegen (Py -> C):         757.7x
  Speedup from optimization (A -> B):       1.07x
  Total speedup (Py -> optimized C):      810.0x
Profiling breakdown (hottest first):
  soft_threshold              20.246 us   51.21%  (1000000 calls)
  state_update                19.288 us   48.79%  (1000000 calls)
```

The generated directory also exposes `make bench` and `make profile` directly.
The portable region profiler (`include/fom_profile.h`) is compiled in only for
`make profile` (`-DFOM_PROFILE`); its `FOM_PROF_BEGIN/END` macros are no-ops in
the shipped `make verify` / `make lib` builds, so profiling never affects the
production solver.

### What changed, and a marker-free production build

When Phase B runs, two extra artifacts are written:

- `OPTIMIZATIONS.md` — a durable record of the run: the timing summary, the
  agent's own notes, and a **unified diff of the solver** from the initial
  (phase A) version to the optimized one, so you can see exactly what was
  changed. Also available programmatically as `result.optimization_diff`.
- `release/fom_solver.c` — a **production copy with the profiling markers and
  the `fom_profile.h` include stripped**. The markers are already no-ops outside
  `make profile`, so this is behavior- and performance-identical; it is just a
  clean, dependency-free source for embedding. Build it with `make release`
  (which links the shared library from `release/`), or drop it in for
  `src/fom_solver.c`.

## Linear solves: factorized, never inverted

Affine operators (e.g. the KKT solve inside ADMM/OSQP) are **never** turned into
a matrix inverse. At codegen time each system `M w = rhs` is factorized once —
**sparse LDLᵀ** for symmetric systems (the ADMM/OSQP KKT is symmetric
quasidefinite), **LU** otherwise — and the generated solver applies
**division-free** triangular solves (the factor diagonals are stored as
reciprocals). The factors live in `src/fom_linsolve.c` and are called as
`fom_lin_solve_<name>(rhs, out)`; the iteration code assembles the right-hand
side and calls them. `CodegenConfig(solve_method=...)` can force `LDLT`/`LU`;
`CG` (an indirect method for large/GPU problems) is reserved for a future
backend. The choice can depend on problem type and the CPU/GPU target — but a
matrix inverse is never formed or stored. The LDLᵀ factorization uses a
fill-reducing (reverse Cuthill–McKee) ordering so the factor stays sparse (on the
MPC KKT, 552 vs 19600 dense entries).

Likewise, every `matrix @ vector` product in the iteration is emitted as a
**sparse CSR** operation (`fom_apply_<name>` in `fom_ops.c`) when the matrix is
structured — the OSQP/MPC selection, dynamics, and state-update blocks are highly
sparse — and dense only when the matrix genuinely is. Together these make the
*initial* generated solver fast natively (no reliance on the optimization phase
to recover structure): on the MPC example the un-optimized C solver already runs
~29× faster than the NumPy reference.

## Iteration-varying coefficients

Accelerated methods use a coefficient that changes every iteration: FISTA's
momentum `γ_k = (β_{k-1} - 1)/β_k`, Nesterov's `(k-1)/(k+2)`, a Halpern anchor
weight. Verification handles these for free — the tape is unrolled, so each
iteration just gets its own number.

Generating the recorded run keeps that property, so **nothing special is needed**.
Write the coefficient in Python, where the rest of your algorithm already lives:

```python
vp = Verifier(method="simulator", obbt_solver=None)
b_param = vp.add_param(m, lb=b - 1.0, ub=b + 1.0)
y0 = vp.add_iterate(n, lb=2, ub=2)

y_prev, y_curr, beta = y0, y0, 1.0
iterates = []
for _ in range(K):
    beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
    gamma = (beta - 1.0) / beta_next          # ordinary Python, evaluated while recording
    extrap = y_curr + gamma * (y_curr - y_prev)
    y_next = vp.step.soft_threshold(At @ extrap + Bt @ b_param, lambd=t)
    y_prev, y_curr, beta = y_curr, y_next, beta_next
    iterates.append(y_next)

vp.codegen("out/fista", iterates=iterates, name="fista")
```

Because the momentum genuinely varies, this run will not roll back up: each
iteration keeps its own coefficient blocks. The cost is one emitted block per
iteration, which is fine at a horizon of tens and impractical at a horizon of
hundreds.

A **schedule** buys back the compact loop. You declare which scalars vary,
FOMTools tabulates them at generation time, and the emitted loop reads the table
at its counter — one block total whatever the horizon, still division-free and
with no `libm` call. Two things differ: you supply the recording as a *function*
of the schedule scalars rather than a finished tape, and the state carries both
iterates the method needs:

```python
import numpy as np
from fomtools.codegen import CodegenConfig, Recording, TerminationSpec, generate_code
from fomtools.verifier import Verifier

def momentum(k):
    """FISTA's γ in the 0-based index of the emitted loop.

    Note the `k + 1`: γ_0 is the coefficient the *first* iteration applies, and it
    is 0. Stopping one step short is the easy mistake, and it silently shifts the
    whole schedule.
    """
    beta = 1.0
    for _ in range(k + 1):
        beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
        gamma, beta = (beta - 1.0) / beta_next, beta_next
    return gamma

def build(gamma):
    vp = Verifier(method="simulator", obbt_solver=None)
    b_param = vp.add_param(m, lb=b - 1.0, ub=b + 1.0)
    y_prev = vp.add_iterate(n, lb=2, ub=2)
    y_curr = vp.add_iterate(n, lb=2, ub=2)
    extrap = y_curr + gamma * (y_curr - y_prev)
    y_next = vp.step.soft_threshold(At @ extrap + Bt @ b_param, lambd=t)
    # the state shifts: (y_prev, y_curr) <- (y_curr, y_next)
    return Recording(vp, next_state=[y_curr, y_next], state=[y_prev, y_curr])

generate_code(
    build, None, "out/fista",
    schedules={"gamma": momentum},
    config=CodegenConfig(termination=TerminationSpec(max_iters=200)),
)
```

A schedule is a callable of the **0-based** loop index, a sequence, or a
`Schedule`. Textbook formulas are written in 1-based `k`, so shifting them is your
job: what a coefficient *is* belongs to the method, not to the code generator, and
nothing in the package knows any particular one. `as_schedule` calls a callable
with increasing indices, so a coefficient defined by a recurrence may cache it.

### Why a build function

A schedule scalar enters the recorded expression linearly, so every extracted
coefficient block is *affine* in it: `M(γ) = M₀ + γ M₁`. FOMTools evaluates
`build` at a few values of `γ`, recovers `M₀` and `M₁` by finite differences, and
holds one probe back to check the fit. The generated code then computes
`M₀ @ state + γ_k * (M₁ @ state)`, one extra matrix-vector product per varying
term.

That validation is what makes this safe rather than merely convenient. If a
coefficient does *not* depend affinely on the scalar — a product of two
schedules, or a scalar used inside a nonlinearity — the held-out probe fails and
you get a `ScheduleFitError` instead of a wrong solver.

### Declaring the state

Written over a single iterate, the dependency cone of iteration `k` reaches back
through every earlier step, so extraction returns the unrolled prefix rather than
a repeatable body. On the `codegen` path that is the intended answer. On the
scheduled and `codegen_rolled` paths it is not what you asked for, and a longer
body is still a valid body, so declaring `state=[y_prev, y_curr]` closes the cone
after one step. Worth remembering for any method that looks back more than one
iterate.

### What is not supported

The varying quantity has to be a linear-operator coefficient. These each raise a
clear error rather than lowering something wrong:

- a **factorized system matrix** that varies (an ADMM `ρ` that changes with `k`),
  since the factorization is computed once at generation time;
- a varying **activation scalar** — a soft-threshold level or a saturated-linear
  bound that changes with `k`;
- a varying **constant offset**;
- a scalar the build function ignores, which is nearly always a typo in its
  keyword name.

The table is sized to `max_iters`. Raising `settings.max_iter` past that at
runtime holds the last value rather than reading off the end; the generated
accessor `fom_sched_<name>(it)` clamps, and the static checker rejects any solver
body that indexes `FOM_SCHED_*` directly.

## Targets

`CodegenConfig` selects what to emit:

- **Embedded** (`version="embedded"`, `backend="c"`): statically sized, no
  dynamic allocation, division-free (matrix inverses are folded into the
  coefficients offline). The default.
- **Large-scale** (`version="large_scale"`): multithreaded CPU (`backend="openmp"`)
  or GPU (`backend="cuda"`). These are generated and compile-checked; runtime
  verification of the CUDA backend requires a GPU.

## CLI

```
fomtools-codegen --example ista      --out out/ista  --driver agent
fomtools-codegen --example dr        --out out/dr    --scaffold-only
fomtools-codegen --example mpc       --out out/mpc   --driver agent   # OSQP MPC, larger
fomtools-codegen --example fista     --out out/fista --scaffold-only  # a recorded run
fomtools-codegen --example scheduled --out out/sched --scaffold-only  # a declared coefficient
```

The built-in `mpc` example is a model-predictive-control QP solved with **OSQP's
ADMM** — a constant KKT solve (the inverse is folded into the coefficients at
codegen time) followed by a box projection (saturated-linear) and the ADMM
recombinations. It is the largest built-in example (state dimension in the
hundreds), so it is a good showcase for the Python-vs-C and the optimization-loop
speedups.

`fista` and `scheduled` are the same accelerated lasso reached the two ways.
`fista` records it as an ordinary loop, so scaffolding it shows an unrolled
`SPEC.md`: one block per recorded iteration, `FOM_UNROLLED 1`, and no table.
`scheduled` declares the momentum instead, so it shows a single block, the
`FOM_SCHED_gamma` table in `fom_data.c`, and the clamping accessor. Scaffolding
both side by side is the quickest way to see what declaring a coefficient buys.

`--scaffold-only` emits everything except the solver body (useful to inspect the
spec and data before invoking the agent). `--no-roll` keeps a recorded run
unrolled even when it could be rolled back up.

A custom FOM can be supplied with `--module mymodule.py`, where the module defines
`build_fom()` returning a dict. Which key you provide selects the path:

| keys | generates |
|---|---|
| `verifier`, `iterates` | the whole recorded run, rolled back up when that is verified to be equivalent |
| `verifier`, `next_state` | a rolled solver from one stationary iteration |
| `build`, `schedules` | a rolled solver with tabulated iteration-varying coefficients |

`name`, `state` and `params` are optional in all three.
