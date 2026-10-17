"""Large-scale GPU (CUDA) target.

Reuses the embedded C target's ABI headers and numeric data emission but emits
the matrix-vector products and linear solves as *deterministic CUDA device
code* (``src/fom_ops.cu`` / ``src/fom_linsolve.cu``), so the agent-written
solver body (``src/fom_solver.cu``) only orchestrates kernel launches.

Design decisions (mirroring the CPU emission in ``embedded_c.py``):

* **Same ABI, device pointers.** ``fom_apply_<matrix>(x, out)`` and
  ``fom_lin_solve_<name>(b, x)`` keep their CPU signatures, but every pointer
  is a *device* pointer; each call launches kernels on the default stream.
  The solver body stays method-agnostic and free of matvec loops.
* **Hand-written kernels, no cuSPARSE/cuBLAS.** Matrix data is baked into
  ``__device__ const`` arrays (loaded with the CUDA module — no runtime
  upload, no library handles, no extra link dependencies). SpMV is one thread
  per row; dot products use a fixed-grid shared-memory block reduction, so the
  in-run floating-point reduction order is stable. Library-backed SpMV is a
  possible later optimization, not a correctness need.
* **CG runs entirely on the device** (SpMV + dot + axpy kernels), mirroring
  :func:`fomtools.codegen.factorize._pcg_solve` exactly: same iteration, same
  stopping test ``r.r <= tol^2 (b.b)``. Only the reduction *order* differs
  from the serial NumPy oracle; CG converges to ``cg_tol`` so both land well
  within the verify tolerance.
* **Direct (LDL^T / LU) solves stay on the host.** Sparse triangular solves
  have loop-carried dependencies and do not parallelize naively; the emitted
  ``fom_lin_solve_<name>`` stages ``b``/``x`` through host buffers and runs
  the CPU routine. Prefer ``solve_method=CG`` for large SPD systems on CUDA.
* No matrix inverse is ever formed or used (project-wide invariant).

The verify command auto-detects the hardware: on a GPU machine (``nvcc`` +
``nvidia-smi`` reporting a device, or ``FOMTOOLS_CUDA_RUN=1``) it is
``make verify`` — build *and* run the real comparison against the oracle —
while CI / dev boxes with no GPU fall back to a ``make compile`` compile-check.
See :meth:`CudaTarget.verify_command`.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets.embedded_c import (
    EmbeddedCTarget,
    _c_array,
    _c_int_array,
    _csr_from_dense,
    _fmt_float,
    _is_csr,
    _matvec_matrix_names,
)

#: Threads per block for every emitted kernel (matvecs, CG, reductions).
_THREADS = 256
#: Fixed grid size of the reduction kernels: keeps the partial-sum buffer
#: statically sized and the in-run floating-point reduction order stable.
_RED_BLOCKS = 64

_DEVICE_CONST = "__device__ static const"


class CudaTarget(EmbeddedCTarget):
    name = "large_scale_cuda"

    def solver_source_path(self) -> str:
        return "src/fom_solver.cu"

    def ops_source_path(self) -> str:
        return "src/fom_ops.cu"

    def optimization_editable_paths(self) -> tuple[str, ...]:
        """Only the solver: without a GPU, verify only compiles the CUDA sources,
        so a rewritten linear solve could not be checked."""
        return (self.solver_source_path(),)

    def linsolve_source_path(self) -> str:
        return "src/fom_linsolve.cu"

    def verify_command(self) -> list[str]:
        """``make verify`` (build *and run* on a GPU) when one is available, else
        ``make compile`` (compile-check only).

        CI and dev boxes have no GPU, so the default is a compile check. On a GPU
        machine the harness automatically runs the real comparison against the
        oracle. Detection: the ``FOMTOOLS_CUDA_RUN`` env var forces the choice
        (``1`` → run, ``0`` → compile-only); otherwise it auto-detects a usable
        GPU (``nvcc`` to build + ``nvidia-smi`` reporting a device to run).
        """
        return ["make", "verify"] if self._can_run_on_gpu() else ["make", "compile"]

    @staticmethod
    def _can_run_on_gpu() -> bool:
        override = os.environ.get("FOMTOOLS_CUDA_RUN")
        if override is not None:
            return override.strip().lower() not in ("", "0", "false", "no", "off")
        if not shutil.which("nvcc") or not shutil.which("nvidia-smi"):
            return False
        try:
            proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return False
        # `nvidia-smi -L` lists one "GPU <n>: ..." line per device, or
        # "No devices were found" when none are visible.
        return proc.returncode == 0 and b"GPU " in proc.stdout

    def static_checks(self, source: str) -> list[str]:
        return []

    def agent_notes(self) -> str:
        return "\n".join(
            [
                "Target-specific notes (large-scale CUDA):",
                "- Every pointer passed to `fom_apply_*` and `fom_lin_solve_*` is a",
                "  DEVICE pointer; each call launches CUDA kernels on the default",
                "  stream. Zero an accumulator with cudaMemset before the fom_apply_*",
                "  calls of a linear op (the helpers accumulate `out += M @ x`).",
                "- Keep the state resident on the device for the WHOLE iteration loop:",
                "  cudaMalloc buffers in `fom_setup`, upload state/params once at the",
                "  start of `fom_solve`, swap device pointers instead of copying, and",
                "  copy the solution back to the host workspace once at the end.",
                "  Avoid per-iteration host<->device copies of full vectors.",
                "- Write the activation kernels (soft-threshold / ReLU / clip) and the",
                "  residual reduction yourself in the .cu file; read constants from",
                "  `fom_data.h` (host) and pass them as kernel arguments.",
                "- The embedded no-malloc / division-free constraints do NOT apply to",
                "  this target.",
            ]
        )

    # -- matrix-vector products (device kernels) ----------------------------
    def _ops_header(self, ir: FomIR) -> str:
        lines = [
            "#ifndef FOM_OPS_H",
            "#define FOM_OPS_H",
            "",
            '#include "fom_workspace.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "/* out += M @ x for each linear-op matrix M (CSR if structurally",
            "   sparse, dense otherwise). CUDA target: `x` and `out` are DEVICE",
            "   pointers; each call launches a kernel (one thread per row) on the",
            "   default stream. Accumulating: zero `out` (cudaMemset) first. */",
            "",
        ]
        for name in _matvec_matrix_names(ir):
            md = ir.matrices[name]
            kind = "CSR" if _is_csr(md) else "dense"
            lines.append(
                f"void fom_apply_{name}(const fom_float *x, fom_float *out);  "
                f"/* {kind} {md.shape[0]}x{md.shape[1] if len(md.shape) > 1 else 1} */"
            )
        lines += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_OPS_H */", ""]
        return "\n".join(lines)

    def _ops_source(self, ir: FomIR) -> str:
        lines = [
            "/* Deterministic device-side matrix-vector products (out += M @ x).",
            "   One thread per row (rows are independent). Matrix data lives in",
            "   __device__ const arrays loaded with the module - no runtime upload.",
            "   All pointers are device pointers. */",
            '#include "fom_ops.h"',
            "",
            "#include <cuda_runtime.h>",
            "",
            f"#define FOM_OPS_THREADS {_THREADS}",
            "",
        ]
        for name in _matvec_matrix_names(ir):
            md = ir.matrices[name]
            shape = md.shape
            rows = shape[0]
            cols = shape[1] if len(shape) > 1 else 1
            up = name.upper()
            if _is_csr(md):
                rowptr, colidx, vals = _csr_from_dense(md.array)
                lines.append(_c_int_array(f"FOM_OP_{up}_ROWPTR", rowptr, storage=_DEVICE_CONST))
                lines.append(_c_int_array(f"FOM_OP_{up}_COLIDX", colidx, storage=_DEVICE_CONST))
                lines.append(_c_array(f"FOM_OP_{up}_VALS", vals, storage=_DEVICE_CONST))
                lines += [
                    "",
                    f"__global__ void fom_apply_{name}_kern(const fom_float *x, fom_float *out) {{",
                    "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                    f"    if (i >= {rows}) return;",
                    "    fom_float s = 0.0;",
                    f"    for (fom_int p = FOM_OP_{up}_ROWPTR[i]; p < FOM_OP_{up}_ROWPTR[i + 1]; "
                    "++p)",
                    f"        s += FOM_OP_{up}_VALS[p] * x[FOM_OP_{up}_COLIDX[p]];",
                    "    out[i] += s;",
                    "}",
                ]
            else:
                # Device copy of the dense FOM_MAT_<name> coefficients (fom_data.c
                # is host memory; kernels need the values in device global memory).
                lines.append(_c_array(f"FOM_OP_{up}_DENSE", md.array, storage=_DEVICE_CONST))
                lines += [
                    "",
                    f"__global__ void fom_apply_{name}_kern(const fom_float *x, fom_float *out) {{",
                    "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                    f"    if (i >= {rows}) return;",
                    "    fom_float s = 0.0;",
                    f"    for (fom_int j = 0; j < {cols}; ++j) "
                    f"s += FOM_OP_{up}_DENSE[i * {cols} + j] * x[j];",
                    "    out[i] += s;",
                    "}",
                ]
            blocks = (rows + _THREADS - 1) // _THREADS
            lines += [
                "",
                f'extern "C" void fom_apply_{name}(const fom_float *x, fom_float *out) {{',
                f"    fom_apply_{name}_kern<<<{blocks}, FOM_OPS_THREADS>>>(x, out);",
                "}",
                "",
            ]
        return "\n".join(lines)

    # -- linear solves -------------------------------------------------------
    def _linsolve_header(self, ir: FomIR) -> str:
        lines = [
            "#ifndef FOM_LINSOLVE_H",
            "#define FOM_LINSOLVE_H",
            "",
            '#include "fom_workspace.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "/* Solve M x = b for each affine operator; `b` and `x` are DEVICE",
            "   pointers. CG systems run entirely on the GPU (CSR SpMV + reduction",
            "   kernels + a diagonal Jacobi preconditioner). Direct factorizations",
            "   (LDL^T / LU) are sparse triangular solves with loop-carried",
            "   dependencies - inherently sequential - so they run on the host; the",
            "   wrapper stages b/x through host buffers. Prefer solve_method=CG for",
            "   large SPD systems on CUDA. No matrix inverse is ever formed or",
            "   used. */",
            "",
        ]
        for name, fac in ir.factorizations.items():
            lines.append(f"#define FOM_FAC_{name.upper()}_DIM {fac.dim}")
            where = "device CG" if fac.method == "cg" else f"host {fac.method.upper()}"
            lines.append(
                f"void fom_lin_solve_{name}(const fom_float *b, fom_float *x);  "
                f"/* {where}, dim {fac.dim}, nnz {fac.nnz} */"
            )
            lines.append("")
        lines += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_LINSOLVE_H */", ""]
        return "\n".join(lines)

    def _linsolve_source(self, ir: FomIR) -> str:
        lines = [
            "/* Deterministic linear solves for the CUDA target. CG solves run on",
            "   the device (kernels below); direct LDL^T/LU triangular solves are",
            "   sequential and run on the host behind a device-pointer wrapper.",
            "   No matrix inverse is ever formed or used. */",
            '#include "fom_linsolve.h"',
            "",
            "#include <cuda_runtime.h>",
            "",
        ]
        if any(fac.method == "cg" for fac in ir.factorizations.values()):
            lines.append(self._cg_shared_helpers())
            lines.append("")
        for name, fac in ir.factorizations.items():
            up = name.upper()
            if fac.method == "cg":
                lines.append(self._cg_routine_cu(name, fac))
                lines.append("")
                continue
            lines.append(
                f"/* ---- factorization '{name}' ({fac.method.upper()}, host: triangular"
                " solves are sequential) ---- */"
            )
            lines.append(_c_int_array(f"FOM_FAC_{up}_PERM", fac.perm))
            lines.append(_c_int_array(f"FOM_FAC_{up}_L_INDPTR", fac.L.indptr))
            lines.append(_c_int_array(f"FOM_FAC_{up}_L_INDICES", fac.L.indices))
            lines.append(_c_array(f"FOM_FAC_{up}_L_DATA", fac.L.data))
            if fac.method == "ldlt":
                lines.append(_c_array(f"FOM_FAC_{up}_DINV", fac.dinv))
            else:
                lines.append(_c_int_array(f"FOM_FAC_{up}_U_INDPTR", fac.U.indptr))
                lines.append(_c_int_array(f"FOM_FAC_{up}_U_INDICES", fac.U.indices))
                lines.append(_c_array(f"FOM_FAC_{up}_U_DATA", fac.U.data))
                lines.append(_c_array(f"FOM_FAC_{up}_UINV", fac.uinv))
            lines.append("")
            lines.append(
                self._solve_routine(
                    name, fac, func_name=f"fom_lin_solve_{name}_host", storage="static"
                )
            )
            lines += [
                "",
                f"static fom_float fom_lin_b_{name}[FOM_FAC_{up}_DIM];",
                f"static fom_float fom_lin_x_{name}[FOM_FAC_{up}_DIM];",
                "",
                f'extern "C" void fom_lin_solve_{name}(const fom_float *b, fom_float *x) {{',
                f"    cudaMemcpy(fom_lin_b_{name}, b, sizeof(fom_float) * FOM_FAC_{up}_DIM,",
                "               cudaMemcpyDeviceToHost);",
                f"    fom_lin_solve_{name}_host(fom_lin_b_{name}, fom_lin_x_{name});",
                f"    cudaMemcpy(x, fom_lin_x_{name}, sizeof(fom_float) * FOM_FAC_{up}_DIM,",
                "               cudaMemcpyHostToDevice);",
                "}",
                "",
            ]
        return "\n".join(lines)

    def _cg_shared_helpers(self) -> str:
        """Kernels shared by every device CG solve: dot product + vector updates."""
        return "\n".join(
            [
                f"#define FOM_RED_THREADS {_THREADS}",
                f"#define FOM_RED_BLOCKS {_RED_BLOCKS}",
                "",
                "/* Grid-stride dot product with a shared-memory block reduction.",
                "   The fixed grid keeps the partial buffer statically sized and the",
                "   in-run floating-point reduction order stable (it still differs",
                "   from the serial NumPy oracle; CG converges to cg_tol, so both",
                "   land well within the verify tolerance). */",
                "__global__ void fom_dot_kern(const fom_float *a, const fom_float *b, "
                "fom_int n, fom_float *partials) {",
                "    __shared__ fom_float sh[FOM_RED_THREADS];",
                "    fom_float s = 0.0;",
                "    for (fom_int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;",
                "         i += gridDim.x * blockDim.x)",
                "        s += a[i] * b[i];",
                "    sh[threadIdx.x] = s;",
                "    __syncthreads();",
                "    for (fom_int w = blockDim.x >> 1; w > 0; w >>= 1) {",
                "        if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];",
                "        __syncthreads();",
                "    }",
                "    if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];",
                "}",
                "",
                "static fom_float *fom_dot_partials_d = 0;",
                "static fom_float fom_dot_partials_h[FOM_RED_BLOCKS];",
                "",
                "static fom_float fom_device_dot(const fom_float *a, const fom_float *b, "
                "fom_int n) {",
                "    fom_int i;",
                "    fom_float s = 0.0;",
                "    if (!fom_dot_partials_d)",
                "        cudaMalloc((void **)&fom_dot_partials_d, "
                "sizeof(fom_float) * FOM_RED_BLOCKS);",
                "    fom_dot_kern<<<FOM_RED_BLOCKS, FOM_RED_THREADS>>>(a, b, n, "
                "fom_dot_partials_d);",
                "    cudaMemcpy(fom_dot_partials_h, fom_dot_partials_d,",
                "               sizeof(fom_float) * FOM_RED_BLOCKS, cudaMemcpyDeviceToHost);",
                "    for (i = 0; i < FOM_RED_BLOCKS; ++i) s += fom_dot_partials_h[i];",
                "    return s;",
                "}",
                "",
                "/* x += alpha p; r -= alpha Ap (fused) */",
                "__global__ void fom_cg_update_xr_kern(fom_float *x, fom_float *r, "
                "const fom_float *p,",
                "                                      const fom_float *Ap, fom_float alpha, "
                "fom_int n) {",
                "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    x[i] += alpha * p[i];",
                "    r[i] -= alpha * Ap[i];",
                "}",
                "",
                "/* p = z + beta p */",
                "__global__ void fom_cg_update_p_kern(fom_float *p, const fom_float *z, "
                "fom_float beta, fom_int n) {",
                "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    p[i] = z[i] + beta * p[i];",
                "}",
            ]
        )

    def _cg_routine_cu(self, name: str, fac) -> str:
        """Device-resident Jacobi-preconditioned CG for one SPD system.

        Mirrors :func:`fomtools.codegen.factorize._pcg_solve` exactly (same
        iteration, same stopping test ``r.r <= tol^2 (b.b)``); the host drives
        the outer loop, every vector kernel and the SpMV run on the device, and
        only the dot-product scalars cross back per iteration.
        """
        up = name.upper()
        n = f"FOM_FAC_{up}_DIM"
        tol2 = _fmt_float(fac.cg_tol * fac.cg_tol)
        maxit = int(fac.cg_max_iter)
        rowptr, colidx, vals = _csr_from_dense(fac.A.to_dense())
        out = [
            f"/* ---- solve '{name}' (CG, preconditioned conjugate gradient, on the"
            " device) ---- */",
            _c_int_array(f"FOM_FAC_{up}_A_ROWPTR", rowptr, storage=_DEVICE_CONST),
            _c_int_array(f"FOM_FAC_{up}_A_COLIDX", colidx, storage=_DEVICE_CONST),
            _c_array(f"FOM_FAC_{up}_A_VALS", vals, storage=_DEVICE_CONST),
            _c_array(f"FOM_FAC_{up}_MINV", fac.minv, storage=_DEVICE_CONST),
            "",
            "/* Ap = A @ p (sparse CSR matvec, one thread per row) */",
            f"__global__ void fom_cg_{name}_spmv_kern(const fom_float *p, fom_float *Ap) {{",
            "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
            f"    if (i >= {n}) return;",
            "    fom_float s = 0.0;",
            f"    for (fom_int k = FOM_FAC_{up}_A_ROWPTR[i]; k < FOM_FAC_{up}_A_ROWPTR[i + 1]; "
            "++k)",
            f"        s += FOM_FAC_{up}_A_VALS[k] * p[FOM_FAC_{up}_A_COLIDX[k]];",
            "    Ap[i] = s;",
            "}",
            "",
            "/* x = 0; r = b; z = Minv .* r; p = z */",
            f"__global__ void fom_cg_{name}_init_kern(const fom_float *b, fom_float *x, "
            "fom_float *r,",
            "                                        fom_float *z, fom_float *p) {",
            "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
            f"    if (i >= {n}) return;",
            "    x[i] = 0.0;",
            "    r[i] = b[i];",
            f"    z[i] = FOM_FAC_{up}_MINV[i] * b[i];",
            "    p[i] = z[i];",
            "}",
            "",
            "/* z = Minv .* r */",
            f"__global__ void fom_cg_{name}_precond_kern(const fom_float *r, fom_float *z) {{",
            "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
            f"    if (i >= {n}) return;",
            f"    z[i] = FOM_FAC_{up}_MINV[i] * r[i];",
            "}",
            "",
            f"static fom_float *fom_cg_{name}_r = 0, *fom_cg_{name}_z = 0;",
            f"static fom_float *fom_cg_{name}_p = 0, *fom_cg_{name}_Ap = 0;",
            "",
            f'extern "C" void fom_lin_solve_{name}(const fom_float *b, fom_float *x) {{',
            f"    const fom_int n = {n};",
            "    const fom_int blocks = (n + FOM_RED_THREADS - 1) / FOM_RED_THREADS;",
            "    fom_float bb, thresh, rz, rz_new, pAp, alpha, beta, rr;",
            "    fom_int it;",
            f"    if (!fom_cg_{name}_r) {{",
            f"        cudaMalloc((void **)&fom_cg_{name}_r, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_z, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_p, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_Ap, sizeof(fom_float) * n);",
            "    }",
            f"    fom_cg_{name}_init_kern<<<blocks, FOM_RED_THREADS>>>(b, x, fom_cg_{name}_r,",
            f"                                                        fom_cg_{name}_z, "
            f"fom_cg_{name}_p);",
            "    bb = fom_device_dot(b, b, n);",
            "    if (bb == 0.0) return;  /* x was zeroed by the init kernel */",
            f"    thresh = (fom_float){tol2} * bb;",
            f"    rz = fom_device_dot(fom_cg_{name}_r, fom_cg_{name}_z, n);",
            f"    for (it = 0; it < {maxit}; ++it) {{",
            f"        fom_cg_{name}_spmv_kern<<<blocks, FOM_RED_THREADS>>>(fom_cg_{name}_p, "
            f"fom_cg_{name}_Ap);",
            f"        pAp = fom_device_dot(fom_cg_{name}_p, fom_cg_{name}_Ap, n);",
            "        if (pAp == 0.0) break;",
            "        alpha = rz / pAp;",
            f"        fom_cg_update_xr_kern<<<blocks, FOM_RED_THREADS>>>(x, fom_cg_{name}_r,",
            f"                                                          fom_cg_{name}_p, "
            f"fom_cg_{name}_Ap, alpha, n);",
            f"        rr = fom_device_dot(fom_cg_{name}_r, fom_cg_{name}_r, n);",
            "        if (rr <= thresh) break;",
            f"        fom_cg_{name}_precond_kern<<<blocks, FOM_RED_THREADS>>>(fom_cg_{name}_r, "
            f"fom_cg_{name}_z);",
            f"        rz_new = fom_device_dot(fom_cg_{name}_r, fom_cg_{name}_z, n);",
            "        beta = rz_new / rz;",
            f"        fom_cg_update_p_kern<<<blocks, FOM_RED_THREADS>>>(fom_cg_{name}_p, "
            f"fom_cg_{name}_z, beta, n);",
            "        rz = rz_new;",
            "    }",
            "}",
        ]
        return "\n".join(out)

    # -- build ---------------------------------------------------------------
    def _makefile(self) -> str:
        return "\n".join(
            [
                "CC ?= cc",
                "NVCC ?= nvcc",
                "CFLAGS ?= -O3 -Iinclude -Itest",
                "NVCCFLAGS ?= -O3 -Iinclude -Itest",
                "",
                "CU_SRC = src/fom_solver.cu src/fom_ops.cu src/fom_linsolve.cu",
                "",
                ".PHONY: compile verify bench lib clean",
                "",
                "fom_solver.o: src/fom_solver.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "fom_ops.o: src/fom_ops.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "fom_linsolve.o: src/fom_linsolve.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "fom_data.o: src/fom_data.c",
                "\t$(CC) $(CFLAGS) -c $< -o $@",
                "",
                "OBJS = fom_solver.o fom_ops.o fom_linsolve.o fom_data.o",
                "",
                "# Compile-check: build BOTH the verify and bench binaries (no GPU run).",
                "# Building bench_bin here too means a CI compile-check catches bench",
                "# breakage even though benchmarking itself needs a GPU.",
                "compile: $(OBJS) test/verify_main.c test/bench_main.c",
                "\t$(CC) $(CFLAGS) -c test/verify_main.c -o verify_main.o",
                "\t$(NVCC) $(NVCCFLAGS) $(OBJS) verify_main.o -o verify_bin",
                "\t$(CC) $(CFLAGS) -c test/bench_main.c -o bench_main.o",
                "\t$(NVCC) $(NVCCFLAGS) $(OBJS) bench_main.o -o bench_bin",
                "",
                "# Run-verification (requires a GPU): check the solver vs the oracle.",
                "verify: compile",
                "\t./verify_bin",
                "",
                "# Benchmark full solves (requires a GPU); prints a BENCH line.",
                "bench: compile",
                "\t./bench_bin",
                "",
                "lib: $(CU_SRC) src/fom_data.c",
                "\t$(NVCC) $(NVCCFLAGS) --shared -Xcompiler -fPIC $(CU_SRC) src/fom_data.c "
                "-o libfomsolver.so",
                "",
                "clean:",
                "\trm -f verify_bin bench_bin *.o libfomsolver.so",
                "",
            ]
        )

    def _readme(self, ir) -> str:
        return "\n".join(
            [
                f"# Generated solver: `{ir.name}` (large-scale, CUDA GPU)",
                "",
                "GPU solver for a verified first-order method.",
                "",
                "## Build",
                "```",
                "make compile   # compile-check (no GPU needed to build)",
                "make verify    # build and run on a GPU (checks vs the reference oracle)",
                "make bench     # time full solves on a GPU (prints a BENCH line)",
                "make lib       # shared library for the Python wrapper",
                "```",
                "",
                "`src/fom_solver.cu` orchestrates the iteration; the deterministic",
                "device code lives in `src/fom_ops.cu` (matrix-vector kernels, CSR",
                "where structured) and `src/fom_linsolve.cu` (device CG; host LDL^T/LU",
                "behind a staging wrapper - triangular solves are sequential). All",
                "`fom_apply_*` / `fom_lin_solve_*` arguments are DEVICE pointers.",
                "Host-side numeric data lives in `src/fom_data.c`. See `SPEC.md`.",
                "",
            ]
        )
