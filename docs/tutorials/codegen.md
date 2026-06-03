# Code generation: from a verified FOM to an embedded C solver

Once you have verified a first-order method with a `Verifier`, FOMTools can
generate a fast, low-level implementation of it — in the spirit of
[OSQP codegen](https://osqp.org/docs/codegen/index.html) — together with a tiny
Python wrapper. The translation is performed by an LLM coding agent that writes
the target code and iterates against a Python reference oracle until the output
is numerically verified.

## Defining the FOM

Build the algorithm exactly as you would for verification, and keep a handle on
the iterate produced after **one** iteration (`next_state`):

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

vp = Verifier(method="simulator", obbt_solver=None)
b_param = vp.add_param(m, lb=b - 1.0, ub=b + 1.0)
z = vp.add_iterate(n, lb=2, ub=2)

next_state = vp.step.soft_threshold(At @ z + Bt @ b_param, lambd=t)  # one iteration
for _ in range(4):  # a few more, for context (optional)
    z = next_state
    next_state = vp.step.soft_threshold(At @ z + Bt @ b_param, lambd=t)
```

## Generating the solver

```python
from fomtools.codegen import CodegenConfig, TerminationKind, TerminationSpec

config = CodegenConfig(
    termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=2000, tol=1e-6),
)

result = vp.codegen("out/ista", next_state=next_state, config=config, name="ista")
assert result.success
```

This requires a coding agent. Enable the real Claude Agent SDK driver with
`FOMTOOLS_CODEGEN_AGENT=1` (needs authentication), or pass an explicit driver
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
    "out/ista", next_state=next_state, config=config, name="ista",
    run_phase_b=True, profile=True,
)
print(result.timing.summary())
```

After Phase A verifies, the orchestrator records a baseline (Python reference
time, the initial generated-C time, and a profiling breakdown). Then the agent
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
  `make profile`, so this is behaviour- and performance-identical; it is just a
  clean, dependency-free source for embedding. Build it with `make release`
  (which links the shared library from `release/`), or drop it in for
  `src/fom_solver.c`.

## Targets

`CodegenConfig` selects what to emit:

- **Embedded** (`version="embedded"`, `backend="c"`): statically sized, no
  dynamic allocation, division-free (matrix inverses are folded into the
  coefficients offline). The default.
- **Large-scale** (`version="large_scale"`): multithreaded CPU (`backend="openmp"`)
  or GPU (`backend="cuda"`). These are generated and compile-checked; run
  verification of the CUDA backend requires a GPU.

## CLI

```
fomtools-codegen --example ista --out out/ista --driver agent
fomtools-codegen --example dr   --out out/dr   --scaffold-only
fomtools-codegen --example mpc  --out out/mpc  --driver agent   # OSQP MPC, larger
```

The built-in `mpc` example is a model-predictive-control QP solved with **OSQP's
ADMM** — a constant KKT solve (the inverse is folded into the coefficients at
codegen time) followed by a box projection (saturated-linear) and the ADMM
recombinations. It is the largest built-in example (state dimension in the
hundreds), so it is a good showcase for the Python-vs-C and the optimization-loop
speedups.

`--scaffold-only` emits everything except the solver body (useful to inspect the
spec and data before invoking the agent). A custom FOM can be supplied with
`--module mymodule.py`, where the module defines `build_fom()` returning a dict
with `verifier` and `next_state`.
