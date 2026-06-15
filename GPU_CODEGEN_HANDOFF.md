# FOMTools codegen — GPU implementation handoff

**Purpose.** This file is a self-contained brief for a Claude instance running on a
GPU cluster, to continue the *GPU / large-scale* part of the FOMTools code
generator. It captures the architecture, the invariants you must preserve, the
exact current state of the CUDA path, and a concrete prioritized work list.
Read the referenced source files for exact details — line numbers drift, but the
symbol/function names below are stable.

Everything described lives on git branch **`codegen`** (pushed to
`origin = github.com:stellatogrp/fomtools`). HEAD when this was written:
`522e672 codegen: harden factorization for non-quasidefinite systems`.

---

## 0. TL;DR of what's done and what's left

**Done (CPU side, all tested, 108 codegen tests passing):**
- IR extraction from a verified FOM → single-iteration intermediate representation.
- NumPy reference **oracle** (ground truth the generated code is verified against).
- **Embedded C** target: static workspace, no malloc, division-free.
- **Affine solves never form an inverse**: `factorize()` emits sparse **LDLᵀ**
  (SPD/quasidefinite), **LU** (general/nonsymmetric), or **CG** (indirect, SPD).
- **Sparse CSR matvecs** (`fom_ops.c`), **RCM fill-reducing ordering** for LDLᵀ.
- **OpenMP** target: `#pragma omp` on the data-parallel loops (matvecs + CG
  kernels), guarded by `#ifdef _OPENMP` (optional, off in serial build).
- **LLM agent loop**: Phase A (translate → verify) + Phase B (profile → optimize
  → re-verify with a perf non-regression gate).
- Examples: `ista`, `dr`, `boxqp`, `mpc` (OSQP-ADMM), `cgqp` (CG), `pdhg_lp`
  (PDHG-for-LP, no restarts), `indef_kkt` (symmetric-indefinite → LU).

**Done since (work item A, implemented on `aa/codegen_refactor`, not yet GPU-run):**
- **Deterministic CUDA kernels** are now emitted: `CudaTarget` overrides
  `_ops_source`/`_linsolve_source` to write `src/fom_ops.cu` (per-matrix
  CSR/dense matvec kernels, one thread per row, data in `__device__ const`
  arrays) and `src/fom_linsolve.cu` (device-resident Jacobi-PCG mirroring
  `factorize._pcg_solve`; direct LDLᵀ/LU stay host-side behind a device-pointer
  staging wrapper). Hand kernels, no cuSPARSE/cuBLAS (no handles/link deps;
  library backing is a possible later optimization). Fixed-grid block-reduction
  dots. **ABI unchanged but all `fom_apply_*`/`fom_lin_solve_*` pointers are
  DEVICE pointers on this target** (documented in the emitted headers and in
  `CudaTarget.agent_notes()`, which is appended to the agent prompts).
- `fom_data.h`/`fom_ops.h`/`fom_linsolve.h` now carry `extern "C"` guards
  (the old CUDA link would have failed on C++ name mangling).
- Goldens rewritten: `golden/ista/large_scale_cuda/fom_solver.cu` (device-resident
  state, pointer swap, residual reduction on device) + new
  `golden/cgqp/large_scale_cuda/fom_solver.cu` (device CG showcase).
- CUDA Makefile compiles the `.cu` helpers with nvcc and has a `bench` target.

**Left (your GPU job):**
- Real on-GPU verification + benchmarking (CI has no GPU, so it never ran).
  Compile-check of the new emission has NOT run anywhere yet (no nvcc on the
  dev machine) — run `pytest tests/codegen -q` on the GPU box first.
- AUTO now upgrades large SPD systems to **CG** on CUDA (work item C, below) —
  but the size threshold default (256) is a guess; re-tune with real GPU timings.
- No cuSPARSE/cuBLAS; no float32 path exercised on hardware.

---

## 1. Environment / how to run

```bash
# Python env (uv). From repo root:
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,codegen,gpu]"   # gpu extra = cupy-cuda12x
# Tests (CPU). Needs a C compiler for the *_pipeline / requires_compiler tests:
pytest tests/codegen -q
# Lint/format:
ruff check fomtools/codegen tests/codegen && ruff format --check fomtools/codegen tests/codegen
```

On the GPU box you additionally have `nvcc` and a GPU, so the CUDA `make verify`
(compile **and run**) and `make bench` will actually execute — they don't in CI.

To run the **real LLM agent** (translate/optimize `.cu` for you), either set
`FOMTOOLS_CODEGEN_AGENT=1` (uses `ClaudeAgentDriver`) or pass
`driver=ClaudeAgentDriver(...)` explicitly. It prefers the in-process
`claude_agent_sdk`, falling back to the `claude` CLI; runs headless with
`permission_mode="bypassPermissions"`. See `fomtools/codegen/agent/claude_sdk.py`.

---

## 2. Architecture / data flow

```
Verifier (records steps of a FOM)
   │  extract_ir(verifier, next_state, config)        fomtools/codegen/extract.py
   ▼
FomIR  ── single stationary iteration ───────────────  fomtools/codegen/ir.py
   │      state vec, params, body steps, state_update,
   │      matrices{name->MatrixData}, factorizations{name->Factorization}
   │
   ├── ReferenceOracle(ir).solve(params)  = GROUND TRUTH (NumPy)  oracle.py
   │
   ├── spec.save/load + render_markdown (SPEC.md for the agent)   spec.py
   │
   ▼
scaffold(ir, out_dir, config) → TargetDescriptor.scaffold(...)   targets/*.py
   │  emits EVERY deterministic file EXCEPT the solver body:
   │  headers (ABI), fom_data.c (numerics), fom_ops.c (matvecs),
   │  fom_linsolve.c (factorized/CG solves), verify_main.c, bench_main.c,
   │  Makefile, SPEC.md, spec.json/data.npz, python wrapper.
   ▼
generate_code(...)  drives the coding agent                      generate.py
   Phase A: agent writes src/fom_solver.c[.cu] until `make verify` == oracle.
   Phase B: profile→optimize→re-verify(correctness gate)→bench(perf gate).
   restore_hook re-scaffolds deterministic files before every verify so the
   agent can ONLY change the solver body.
```

**The agent owns exactly one file:** `src/fom_solver.c` (embedded/OpenMP) or
`src/fom_solver.cu` (CUDA). Everything else is generated by Python and restored
before each verify.

---

## 3. Hard invariants (must hold for GPU code too)

1. **Never form, store, or use a matrix inverse — EVER.** Affine solves go
   through factorization (LDLᵀ/LU) or CG (matvecs only). This is a project-wide
   rule from the PI. See `fomtools/codegen/factorize.py`.
2. **Generated code must match the oracle** to `config.tol_verify` (default
   1e-6) over random parameter samples. `make verify` builds the solver, runs it
   over `test/verify_data.h`, compares to oracle-computed expected solutions, and
   exits nonzero on any sample exceeding tol. The oracle is the arbiter.
   - **Subtle GPU trap:** for CG (and any iterative/reduction kernel), parallel
     floating-point reduction order differs from the serial NumPy oracle. CG
     converges to `cg_tol=1e-12`, so both reach the same solution to ~1e-10 ≪
     1e-6 → still verifies. But if you change the *algorithm* (not just the
     reduction order) the oracle must change too (it lives in NumPy in
     `factorize._pcg_solve` and `oracle.py`).
3. **Agent edits only the solver body.** Deterministic files are restored by
   `restore_hook` (`generate.py: restore_deterministic_files`). Don't rely on the
   agent fixing build files.
4. **Embedded target only**: no malloc, no `/` operator (division-free). GPU /
   OpenMP targets are exempt (`static_checks` returns `[]`), so `cudaMalloc` and
   divisions are allowed there.

---

## 4. The IR you generate code from (`fomtools/codegen/ir.py`)

- `FomIR`: `params: tuple[ParamSpec]`, `state: StateSpec` (dim, slots, init),
  `body: tuple[IterationStep]`, `state_update: LinearOp`, `termination`,
  `matrices: dict[str, MatrixData]`, `factorizations: dict[str, Factorization]`,
  `dtype`, `name`. Helpers: `state_dim`, `max_rhs_dim` (sizes scratch), `tmp_dims()`.
- `IterationStep`: `name` (e.g. "t0"), `out_dim`, `linear_op`, `activation`,
  `solve: str|None` (key into `factorizations` for an affine solve; else None).
- `LinearOp`: `out_dim`, `terms: tuple[Term]`, `const: str|None`. Each `Term` is
  `matrix @ source` where `source ∈ {"state", "param:<p>", "tmp:<t>"}` and
  `matrix` is a key into `ir.matrices`. Computed as
  `out = const + Σ matrix_i @ source_i`.
- `ActivationOp`: kind ∈ {IDENTITY, SOFT_THRESHOLD(lambd), RELU, SATURATED_LINEAR
  (lower/upper vector names or scalars)}. IDENTITY is used for affine-solve steps.
- `MatrixData`: dense `array` (+ optional `sparse_pattern`). The emitter decides
  CSR vs dense via `_is_csr` (size≥16 and 2·nnz<size).

So one iteration is: for each body step compute `tmp =
activation(linear_op(...))` — except an affine step does `tmp =
fom_lin_solve_<name>(linear_op(...))` — then `new_state =
linear_op(state_update)`; residual = `max|new_state - state|`.

---

## 5. Generated C ABI (OSQP-style; stable across targets)

Emitted in `include/fom_workspace.h` + `include/fom_solver.h`:
```c
typedef double fom_float;   /* float if config.dtype="float32" */
typedef int    fom_int;
typedef struct { fom_int max_iter; fom_float tol; } FomSettings;
typedef struct { fom_float p0[FOM_PARAM_P0_DIM]; ... } FomParams;
typedef struct { fom_float x[FOM_STATE_DIM]; fom_int iter; fom_float residual; fom_int status; } FomSolution;
typedef struct { FomSettings settings; FomParams params; FomSolution solution;
                 fom_float state[FOM_STATE_DIM], new_state[...], acc[FOM_MAX_VEC_DIM], t0[...], ...; } FomWorkspace;
void    fom_set_default_settings(FomSettings*);
void    fom_setup(FomWorkspace*, const FomSettings*);     /* reset state to FOM_STATE_INIT */
void    fom_update_params(FomWorkspace*, const FomParams*);/* also resets state */
fom_int fom_solve(FomWorkspace*);                          /* the iteration loop */
void    fom_cleanup(FomWorkspace*);
```
Deterministic helpers the solver body calls (generated in fom_ops.c / fom_linsolve.c):
- `void fom_apply_<matrix>(const fom_float *x, fom_float *out);` — `out += M@x`
  (CSR if structured, dense otherwise). **Accumulating**: zero `out` first.
- `void fom_lin_solve_<name>(const fom_float *b, fom_float *x);` — solves the
  affine system (LDLᵀ/LU triangular solves, or CG). Same signature regardless of
  method, so the solver body is method-agnostic.

The numerics live in `fom_data.c` (`FOM_MAT_*`, `FOM_STATE_INIT`, clip bounds,
`FOM_*_LAMBDA`) and, for sparse ops/factors, as CSR arrays in fom_ops.c /
fom_linsolve.c (`FOM_OP_*_ROWPTR/COLIDX/VALS`, `FOM_FAC_*_A_ROWPTR/...`, etc).

---

## 6. Solve methods (`config.SolveMethod`, `factorize.py`) — GPU-relevant

`SolveMethod ∈ {AUTO, LDLT, LU, CG}`. `factorize(M, method)` returns a
`Factorization` stored in `ir.factorizations`; its `.solve(b)` is the NumPy
reference that the emitted C mirrors **exactly**.

- **AUTO**: LDLᵀ if symmetric *and* it factorizes accurately (backward-error gate
  `‖LDLᵀ−M‖/‖M‖ ≤ 1e-9`), else LU. Singular → clear `ValueError`. RCM
  fill-reducing ordering applied for LDLᵀ.
- **LDLT/LU**: forced direct (with RCM for LDLᵀ).
- **CG** (`_cg_factorization`, `_pcg_solve`): Jacobi-preconditioned conjugate
  gradient. Stores the SPD matrix as CSC + diagonal preconditioner `minv` =
  reciprocals of diag (NOT an inverse — a diagonal scaling). Requires symmetric
  positive-definite (checked via Cholesky). Currently **gated to the large-scale
  version** (it uses runtime scalar division; embedded is division-free). Config
  rejects `CG + EMBEDDED`.

**Parallelism characteristics (critical for GPU):**
- **CSR SpMV** (`fom_apply_*`): row loop is embarrassingly parallel → ideal GPU
  kernel / cuSPARSE SpMV. Each row writes a distinct `out[i]`.
- **CG**: SpMV + dot products (reductions) + axpy updates — **every kernel is
  data-parallel**; only the outer iteration is sequential. This is THE method for
  GPU. The CPU CG kernel emission is in
  `embedded_c.py: EmbeddedCTarget._cg_routine` and is OpenMP-annotated; port that
  structure to CUDA (cuBLAS dot/axpy + cuSPARSE SpMV, or hand kernels + a
  reduction).
- **Direct LDLᵀ/LU triangular solves**: inherently **sequential** (loop-carried
  deps). Don't naively parallelize. On GPU either keep the factorized solve on
  host, use cuDSS/cuSPARSE triangular solve (level-scheduled), or — preferred —
  **use CG instead** for large SPD systems. Document the choice.

---

## 7. Targets (`fomtools/codegen/targets/`)

- `base.py: TargetDescriptor` — abstract: `scaffold`, `verify_command`,
  `solver_source_path`, `static_checks`.
- `embedded_c.py: EmbeddedCTarget` — the workhorse; ALL emission lives here
  (headers, fom_data.c, **fom_ops.c** `_ops_source`, **fom_linsolve.c**
  `_linsolve_source`/`_solve_routine`/`_cg_routine`, verify/bench mains,
  Makefile). Class attr `parallel=False`; method `_omp_for(clause, indent)`
  emits the `#ifdef _OPENMP` / `#pragma omp parallel for` guard lines.
- `large_scale_openmp.py: OpenMpTarget(EmbeddedCTarget)` — sets `parallel=True`
  (so the matvec/CG loops get OpenMP pragmas), relaxes `static_checks`, OpenMP
  Makefile (`OMPFLAGS ?=` empty by default; `make verify OMPFLAGS=-fopenmp`).
- `large_scale_cuda.py: CudaTarget(EmbeddedCTarget)` — **the file you'll extend.**
  - `solver_source_path() -> "src/fom_solver.cu"`.
  - `verify_command() -> ["make","compile"]` (compile-check only; **change to
    `make verify` once you have a GPU** so it actually runs the comparison).
  - `static_checks -> []` (malloc/division allowed).
  - `_makefile()`: nvcc compile of `fom_solver.cu` + cc compile of `fom_data.c`,
    `fom_linsolve.c`, `fom_ops.c`, link with nvcc. `compile` builds verify_bin but
    doesn't run; `verify` depends on compile then `./verify_bin`; `lib` builds a
    shared lib. **NOTE:** it currently compiles `fom_linsolve.c`/`fom_ops.c` as
    host C and links them — your CUDA solver can call those host helpers (slow,
    copies) OR you can emit device kernels (the work).

`targets/__init__.py: get_target(config)` maps version+backend →
{EmbeddedCTarget, OpenMpTarget, CudaTarget}.

**Current CUDA reality** (see `tests/codegen/golden/ista/large_scale_cuda/fom_solver.cu`
and `.../cgqp/large_scale_cuda/fom_solver.cu`): the goldens now orchestrate the
deterministic device helpers — device-resident state across the whole loop
(pointer swap), `fom_apply_*` kernel launches, an activation kernel, and a
max-|diff| residual reduction on device. Never compiled or run on a GPU yet
(no nvcc here): compile-check + run-verify them on the cluster first.

---

## 8. GPU work list (prioritized)

**(A) [DONE — see §0] Deterministically emit CUDA kernels for the matvecs and CG.** Mirror
`_ops_source` and `_cg_routine`. Two viable designs:
   1. *Library-backed*: emit CSR descriptors and call **cuSPARSE** SpMV +
      **cuBLAS** dot/axpy for CG. Cleanest, fastest to get right. The CSR arrays
      already exist (`FOM_OP_*`, `FOM_FAC_*_A_*`, `FOM_FAC_*_MINV`).
   2. *Hand kernels*: a CSR-SpMV kernel (one thread/row or warp/row) + a
      reduction kernel for dots + axpy kernels. More control, more code.
   Put device-side emission in `CudaTarget` overriding `_ops_source` /
   `_linsolve_source` (or new `_ops_source_cu` / `_cg_routine_cu`) so the solver
   body calls `fom_apply_*` / `fom_lin_solve_*` device wrappers — keeping the ABI
   identical so the agent's `.cu` orchestration stays simple and method-agnostic.

**(B) [DONE in the goldens + agent prompt; verify on hardware] Keep state on the
device across iterations.** The rewritten goldens allocate device buffers in
`fom_setup`, upload state/params once per solve, swap device pointers instead of
copying, reduce the residual on device, and copy the solution out once at the
end; `CudaTarget.agent_notes()` instructs the live agent to do the same. Remaining:
confirm on hardware, and consider reducing the residual only every
`FOM_CHECK_TERMINATION` iters for large problems (goldens reduce every iter).

**(C) [DONE] Make AUTO choose CG for large SPD systems on CUDA.** Implemented in
`extract.py: _resolve_solve_method(lhs_mat, config)`: when
`solve_method==AUTO and version==LARGE_SCALE and backend==CUDA` and the matrix is
SPD (`factorize.is_spd`, a non-throwing Cholesky certificate) and large
(`dim >= config.cg_auto_min_dim`, default 256), it upgrades to CG; otherwise it
returns `config.solve_method.value` unchanged. Explicit `LDLT`/`LU`/`CG` always
bypass the heuristic, and `cg_auto_min_dim` tunes/disables the size gate. A
quasidefinite (symmetric-indefinite) KKT fails the SPD check, so OSQP/MPC systems
stay on a direct factorization — they'd need MINRES or a normal-equations form
for an indirect GPU solve, not done here. Documented on `SolveMethod` /
`CodegenConfig`; tested in `test_factorize.py` (`is_spd`) and `test_large_scale.py`
(upgrade / size-gate / indefinite-stays-direct / embedded-never-CG /
explicit-bypass). Note: the *threshold default (256)* is an unvalidated guess —
re-tune once (D) gives real GPU timings for the CG-vs-direct crossover.

**(D) Real GPU verification + benchmarking.** Flip `CudaTarget.verify_command` to
`["make","verify"]` on the cluster; confirm `VERIFY OK` against the oracle for
ista/pdhg_lp/cgqp at GPU scale. Add a `make bench` path. Watch the FP-reduction
tolerance note in §3.2.

**(E) float32 path.** `CodegenConfig(dtype="float32")` sets `fom_float=float`.
GPUs prefer it. Re-check `tol_verify` (1e-6 may be tight for f32 CG; loosen for
f32 or tighten cg_tol). Large-scale GPU is the intended home for f32.

**(F) Scale the examples.** The built-in examples are tiny (verification-sized).
For GPU you want large instances (e.g. `mpc_osqp_example(nx, nu, N)` big, or a
large SPD `cg_qp_example(n=...)`, or `pdhg_lp` with big A). PDHG-for-LP and CG-QP
are the natural GPU showcases (pure matvec + projection / pure CG).

**(G) Optional: CuPy GPU oracle / wrapper.** `cupy-cuda12x` is in the `gpu`
extra. Could provide a CuPy reference for cross-checking, or a CuPy-based Python
wrapper alongside the ctypes one (`wrapper.py`).

---

## 9. Examples and their solve structure (in `fomtools/codegen/examples.py` + `tests/codegen/conftest.py` builders)

| name        | builder            | structure                                   | GPU relevance |
|-------------|--------------------|---------------------------------------------|---------------|
| `ista`      | build_ista         | soft-threshold (1 matvec + shrink)          | trivial SpMV demo |
| `boxqp`     | build_boxqp        | saturated-linear (box projection)           | matvec + clip |
| `dr`        | build_dr           | LU solve + partial ReLU                      | direct solve (host) |
| `mpc`       | mpc_osqp_example   | LDLᵀ KKT solve + box proj (OSQP-ADMM)        | indefinite KKT → direct |
| `cgqp`      | build_spd_cg       | **CG** solve (SPD)                           | **GPU CG showcase** |
| `pdhg_lp`   | build_pdhg_lp      | satlin + relu, **no linear solve**          | **pure-SpMV GPU showcase** |
| `indef_kkt` | build_indefinite_kkt| LU solve (symmetric-indefinite, not quasidef)| direct solve |

The CG (`cgqp`) and PDHG (`pdhg_lp`) examples are the cleanest GPU targets:
all-matvec/all-projection, no sequential triangular solve.

---

## 10. Verification & testing conventions

- **Oracle = truth.** `ReferenceOracle(ir).solve(param_values)` (NumPy). Generated
  code is compared to it. There's also `run_faithful` (runs the FOM from the
  Verifier's step objects, solving the affine system with `np.linalg.solve`) used
  to cross-check that IR extraction preserved the algorithm.
- **Golden fixtures + StubAgentDriver:** a known-good solver source dropped into
  the workspace so the whole pipeline runs in CI without the LLM. To add a CUDA
  golden for a new example, scaffold it, inspect the matrix/step names (the
  emitter names are deterministic: `t0_state`, `t0_p_p0`, `su_t_t0`, etc.), write
  the `.cu` to match, place at
  `tests/codegen/golden/<name>/large_scale_cuda/fom_solver.cu`.
- **CUDA tests are skipped without `nvcc`** (`@pytest.mark.skipif(not
  shutil.which("nvcc"))`) — they'll actually run on your cluster. See
  `tests/codegen/test_large_scale.py` (`test_cuda_scaffold`,
  `test_cuda_compile_check`, and the OpenMP threaded-build tests gated on a
  working `-fopenmp`).
- **Agent tests** are gated by `@pytest.mark.agent` + `FOMTOOLS_CODEGEN_AGENT=1`.

Quick generate-and-verify snippet (CUDA, stub driver) you can adapt:
```python
from fomtools.codegen.config import CodegenConfig, Backend, Version, TerminationSpec, SolveMethod
from fomtools.codegen.examples import cg_qp_example
from fomtools.codegen import StubAgentDriver, generate_code
ex = cg_qp_example(n=2000)
cfg = CodegenConfig(version=Version.LARGE_SCALE, backend=Backend.CUDA,
                    solve_method=SolveMethod.CG, dtype="float32",
                    termination=TerminationSpec(max_iters=500))
r = generate_code(ex["verifier"], ex["next_state"], "out_cuda", config=cfg,
                  driver=StubAgentDriver("path/to/golden.cu"), name="cgqp")
# Then on the GPU box: cd out_cuda && make verify   (after flipping verify_command)
```
Run the **live agent** instead by passing `driver=ClaudeAgentDriver(...)` and
`run_phase_b=True, profile=True` — it will profile, optimize the `.cu`, and report
speedups, keeping the change only if faster (perf gate in `generate.py`).

---

## 11. Current measured baselines (CPU, for reference)

Phase A (Python reference → generated C, golden/stub, `-O3`, per-solve):

| example | Python µs | C µs | codegen speedup | hottest region |
|---|--:|--:|--:|---|
| ista | 1152 | 1.85 | 622× | soft_threshold |
| boxqp | 1037 | 3.35 | 309× | box_project |
| dr | 6289 | 9.19 | 684× | affine (LU) |
| pdhg_lp | 1584 | 11.4 | 140× | primal proj |
| mpc | 11478 | 422 | 27× | kkt_solve 72% |
| cgqp | 876 | 12.0 | 73× | CG solve 85% |

Live agent run on MPC (Phase A + Phase B optimization, with perf gate):
Python 11439 µs → Phase A 388 µs (29.5×) → Phase B **71.7 µs (5.40× on top of A)**
→ **159.5× total**; the agent specialized/unrolled the LDLᵀ KKT solve with baked
constants (still division-free, no inverse, verified to 1.55e-15). GPU should
beat the *large-scale* versions of these, especially `cgqp`/`pdhg_lp` where the
work is all data-parallel.

---

## 12. Gotchas

- Harmless `RuntimeWarning: invalid value encountered in add` from
  `linearexpression.py` bound propagation when iterate bounds are degenerate
  (lb=ub=0) — cosmetic, not a codegen error.
- `EmbeddedCTarget.parallel` defaults False; `OpenMpTarget` sets it True;
  `CudaTarget` leaves it False (its parallelism is in kernels, not host loops).
- RCM reordering (in `factorize`, for LDLᵀ) already avoids many small-pivot
  instabilities; the backward-error gate is the safety net beyond that.
- The embedded ABI uses **host static** workspace arrays; the CUDA golden adds its
  own `static fom_float *d_...` device pointers and `cudaMalloc`s in `fom_setup`
  (allowed because CUDA `static_checks` is empty). Decide your device-memory
  ownership model deliberately (workspace struct vs. file-static device ptrs).
- Don't change the oracle's algorithm without changing both the NumPy reference
  (`factorize._pcg_solve`, `oracle.py`) AND the emitted code — they must agree.

---

## 13. Files to read first (in order)

1. `fomtools/codegen/ir.py` — the data model.
2. `fomtools/codegen/factorize.py` — solve methods, CG, never-invert, hardening.
3. `fomtools/codegen/targets/embedded_c.py` — `_ops_source`, `_linsolve_source`,
   `_cg_routine`, `_omp_for` (the patterns to port to CUDA).
4. `fomtools/codegen/targets/large_scale_cuda.py` — what you extend.
5. `tests/codegen/golden/ista/large_scale_cuda/fom_solver.cu` — current CUDA baseline.
6. `fomtools/codegen/oracle.py` + `fomtools/codegen/generate.py` — verification + agent loop.
7. `fomtools/codegen/agent/{prompts.py,claude_sdk.py}` — how the agent is told what to do.

Good luck — keep the ABI identical, never invert, and verify everything against
the oracle.
