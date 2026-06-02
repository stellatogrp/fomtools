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
```

`--scaffold-only` emits everything except the solver body (useful to inspect the
spec and data before invoking the agent). A custom FOM can be supplied with
`--module mymodule.py`, where the module defines `build_fom()` returning a dict
with `verifier` and `next_state`.
