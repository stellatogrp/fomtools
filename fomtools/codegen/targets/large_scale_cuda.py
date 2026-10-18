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
* **Nothing on the host during the iteration.** Every product, solve and
  reduction runs on the device; matrix data is baked into ``__device__ const``
  arrays (loaded with the CUDA module, no runtime upload). SpMV is one thread
  per row (hand-written, no cuSPARSE); dot products use a fixed-grid
  shared-memory block reduction, so the in-run floating-point reduction order
  is stable.
* **CG runs entirely on the device** (SpMV + dot + axpy kernels), mirroring
  :func:`fomtools.codegen.factorize._pcg_solve` exactly: same iteration, same
  stopping test ``r.r <= tol^2 (b.b)``. Only the reduction *order* differs
  from the serial NumPy oracle; CG converges to ``cg_tol`` so both land well
  within the verify tolerance.
* **Direct (LDL^T / LU) solves through cuDSS.** NVIDIA's sparse direct solver
  factorizes the fixed system matrix on the device at the first solve
  (``CUDSS_MTYPE_SYMMETRIC`` with symmetric pivoting for LDL^T, so indefinite
  KKT systems work; ``CUDSS_MTYPE_GENERAL`` with global row pivoting, as the
  CPU path's LU, for LU), and each call is one cuDSS
  solve on device vectors. cuDSS comes from the ``nvidia-cudss`` wheel or
  ``CUDSS_HOME`` and is linked only when a direct solve exists. Large SPD
  systems take device CG under ``solve_method=AUTO``.
* No matrix inverse is ever formed or used (project-wide invariant).

The verify command auto-detects the hardware: on a GPU machine (``nvcc`` +
``nvidia-smi`` reporting a device, or ``FOMTOOLS_CUDA_RUN=1``) it is
``make verify`` — build *and* run the real comparison against the oracle —
while CI / dev boxes with no GPU fall back to a ``make compile`` compile-check.
See :meth:`CudaTarget.verify_command`.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from fomtools.codegen.ir import FomIR, csr_arrays
from fomtools.codegen.objective import objective_c_params
from fomtools.codegen.targets.base import objective_violations
from fomtools.codegen.targets.embedded_c import (
    _KERNEL_CHECK_RECIPE,
    EmbeddedCTarget,
    _c_array,
    _c_int_array,
    _fmt_float,
    _is_csr,
    _matvec_matrix_names,
)


@functools.cache
def _gpu_visible() -> bool:
    """Whether ``nvcc`` can build here and ``nvidia-smi`` sees a device (probed
    once per process)."""
    if not shutil.which("nvcc") or not shutil.which("nvidia-smi"):
        return False
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    # `nvidia-smi -L` lists one "GPU <n>: ..." line per device, or
    # "No devices were found" when none are visible.
    return proc.returncode == 0 and b"GPU " in proc.stdout


def _cudss_home() -> str:
    """cuDSS's install prefix: ``CUDSS_HOME``, else the ``nvidia-cudss`` wheel
    (``nvidia/<cuda>/include/cudss.h``) in this Python, else empty."""
    if os.environ.get("CUDSS_HOME"):
        return os.environ["CUDSS_HOME"]
    spec = importlib.util.find_spec("nvidia")
    for root in spec.submodule_search_locations if spec else []:
        for header in sorted(Path(root).glob("*/include/cudss.h")):
            return str(header.parent.parent)
    return ""


def _direct(ir: FomIR) -> bool:
    """Whether any linear solve is direct (LDL^T / LU), so cuDSS is linked."""
    return any(fac.method != "cg" for fac in ir.factorizations.values())


#: Threads per block for every emitted kernel (matvecs, CG, reductions).
_THREADS = 256
#: Fixed grid size of the reduction kernels: keeps the partial-sum buffer
#: statically sized and the in-run floating-point reduction order stable.
_RED_BLOCKS = 64

_DEVICE_CONST = "__device__ static const"


class CudaTarget(EmbeddedCTarget):
    name = "large_scale_cuda"

    @property
    def csr_in_data(self) -> bool:
        """Sparse matvec data lives in device memory (``fom_ops_data.cuh``). A host
        copy in ``fom_data.c`` exists only where ``fom_ops.cu`` is editable: the
        direct check's reference reads it, independent of the device copy."""
        return self._ops_editable()

    def solver_source_path(self) -> str:
        return "src/fom_solver.cu"

    def ops_source_path(self) -> str:
        return "src/fom_ops.cu"

    def objective_source_path(self) -> str:
        """The device reductions, in their own fixed file (as on the CPU targets),
        so a rewritten ``fom_ops.cu`` can never change the metric."""
        return "src/fom_objective.cu"

    def _objective_file(self, ir: FomIR) -> str:
        lines = ['#include "fom_ops.h"', "", "#include <cuda_runtime.h>", ""]
        if ir.objective is None:
            lines += ["/* No verification objective: this solver does not stop on one. */", ""]
        lines += self._objective_source(ir)
        return "\n".join(lines)

    def optimization_editable_paths(self) -> tuple[str, ...]:
        """The same files as on the CPU targets when verify runs on a GPU, where
        a rewritten solve or product is checked directly; only the solver
        without one, since verify then only compiles the CUDA sources."""
        if self._can_run_on_gpu():
            return super().optimization_editable_paths()
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
        return _gpu_visible()

    def scaffold(self, ir: FomIR, out_dir: Path) -> None:
        super().scaffold(ir, out_dir)
        out = Path(out_dir)
        (out / "include" / "fom_ops_data.cuh").write_text(self._ops_data_header(ir))
        if self._checked_ops(ir) or self._checked_factorizations(ir):
            (out / "test" / "check_stage.h").write_text(self._check_stage_header(ir))
            (out / "test" / "check_stage.cu").write_text(self._check_stage_source(ir))
        else:
            for rel in ("test/check_stage.h", "test/check_stage.cu"):
                (out / rel).unlink(missing_ok=True)

    def ops_data_path(self) -> str:
        return "include/fom_ops_data.cuh"

    # -- direct checks: host probes staged through the device ------------------
    def _checked_call(self, kind: str, name: str) -> str:
        return f"fom_check_{kind}_{name}"

    def _verify_check_includes(self, ir: FomIR) -> list[str]:
        if self._checked_ops(ir) or self._checked_factorizations(ir):
            return ['#include "check_stage.h"']
        return []

    def _check_stage_header(self, ir: FomIR) -> str:
        lines = [
            "#ifndef FOM_CHECK_STAGE_H",
            "#define FOM_CHECK_STAGE_H",
            "",
            '#include "fom_workspace.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "/* Verify only: each checked routine on HOST pointers, staged through",
            "   device buffers, so the direct checks compare it to the host reference. */",
        ]
        for kind, name, _, _ in self._stage_list(ir):
            lines.append(f"void fom_check_{kind}_{name}(const fom_float *in, fom_float *out);")
        lines += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_CHECK_STAGE_H */", ""]
        return "\n".join(lines)

    def _stage_list(self, ir: FomIR) -> list[tuple[str, str, int, int]]:
        out = []
        for name in self._checked_ops(ir):
            shape = ir.matrices[name].shape
            out.append(("apply", name, shape[1] if len(shape) > 1 else 1, shape[0]))
        for name, fac in self._checked_factorizations(ir):
            out.append(("lin_solve", name, fac.dim, fac.dim))
        return out

    def _check_stage_source(self, ir: FomIR) -> str:
        lines = [
            '#include "check_stage.h"',
            '#include "fom_ops.h"',
            '#include "fom_linsolve.h"',
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <cuda_runtime.h>",
            "",
            "static void fom_check_cuda(const char *what) {",
            "    cudaError_t e = cudaDeviceSynchronize();",
            "    if (e == cudaSuccess) e = cudaGetLastError();",
            "    if (e != cudaSuccess) {",
            '        printf("VERIFY FAILED: %s: %s\\n", what, cudaGetErrorString(e));',
            "        exit(1);",
            "    }",
            "}",
            "",
        ]
        for kind, name, n_in, n_out in self._stage_list(ir):
            # A solve overwrites x; a product accumulates, so `out` goes in too.
            upload_out = (
                [
                    f"    cudaMemcpy(d_out, out, sizeof(fom_float) * {n_out}, cudaMemcpyHostToDevice);"
                ]
                if kind == "apply"
                else []
            )
            lines += [
                f'extern "C" void fom_check_{kind}_{name}(const fom_float *in, fom_float *out) {{',
                "    fom_float *d_in, *d_out;",
                f"    cudaMalloc((void **)&d_in, sizeof(fom_float) * {n_in});",
                f"    cudaMalloc((void **)&d_out, sizeof(fom_float) * {n_out});",
                f"    cudaMemcpy(d_in, in, sizeof(fom_float) * {n_in}, cudaMemcpyHostToDevice);",
                *upload_out,
                f"    fom_{kind}_{name}(d_in, d_out);",
                f'    fom_check_cuda("fom_{kind}_{name}");',
                f"    cudaMemcpy(out, d_out, sizeof(fom_float) * {n_out}, cudaMemcpyDeviceToHost);",
                "    cudaFree(d_in);",
                "    cudaFree(d_out);",
                "}",
                "",
            ]
        return "\n".join(lines)

    def static_checks(self, source: str, ir: FomIR | None = None) -> list[str]:
        return objective_violations(source, ir)

    def agent_notes(self) -> str:
        return "\n".join(
            [
                "Target-specific notes (large-scale CUDA):",
                "- Every pointer passed to `fom_apply_*` and `fom_lin_solve_*` is a",
                "  DEVICE pointer; each call launches CUDA kernels on the default",
                "  stream. Zero an accumulator with cudaMemset before the fom_apply_*",
                "  calls of a linear op (the helpers accumulate `out += M @ x`).",
                "- Run the WHOLE iteration on the device: cudaMalloc buffers in",
                "  `fom_setup`, upload state/params once at the start of `fom_solve`,",
                "  swap device pointers instead of copying, and copy the solution back",
                "  to the host workspace once at the end. Do reductions (residuals) on",
                "  the device too, down to one scalar: the only per-iteration host",
                "  traffic is that scalar, read back where the stopping test needs it.",
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
        lines += self._objective_declaration(
            ir,
            pointers="CUDA: every pointer is a DEVICE pointer; the value returns to the host.",
        )
        lines += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_OPS_H */", ""]
        return "\n".join(lines)

    def _ops_data_header(self, ir: FomIR) -> str:
        """``include/fom_ops_data.cuh``: the device copies of the matvec matrices,
        package-owned and restored before every check, so a rewritten
        ``fom_ops.cu`` reads the same data its direct check compares against."""
        lines = [
            "#ifndef FOM_OPS_DATA_CUH",
            "#define FOM_OPS_DATA_CUH",
            "",
            "/* Device copies of the matvec matrices (fom_data.c is host memory),",
            "   loaded with the module - no runtime upload. Included by fom_ops.cu",
            "   only. CSR if structurally sparse, dense otherwise. */",
            '#include "fom_workspace.h"',
            "",
        ]
        for name in _matvec_matrix_names(ir):
            md = ir.matrices[name]
            up = name.upper()
            if _is_csr(md):
                rowptr, colidx, vals = md.csr()
                lines.append(_c_int_array(f"FOM_OP_{up}_ROWPTR", rowptr, storage=_DEVICE_CONST))
                lines.append(_c_int_array(f"FOM_OP_{up}_COLIDX", colidx, storage=_DEVICE_CONST))
                lines.append(_c_array(f"FOM_OP_{up}_VALS", vals, storage=_DEVICE_CONST))
            else:
                lines.append(_c_array(f"FOM_OP_{up}_DENSE", md.dense(), storage=_DEVICE_CONST))
            lines.append("")
        lines += ["#endif /* FOM_OPS_DATA_CUH */", ""]
        return "\n".join(lines)

    def _ops_source(self, ir: FomIR) -> str:
        lines = [
            "/* Deterministic device-side matrix-vector products (out += M @ x).",
            "   One thread per row (rows are independent). Matrix data lives in",
            "   fom_ops_data.cuh as __device__ const arrays. All pointers are device",
            "   pointers. */",
            '#include "fom_ops.h"',
            '#include "fom_ops_data.cuh"',
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
            lines += [
                "static __global__ void "
                f"fom_apply_{name}_kern(const fom_float *x, fom_float *out) {{",
                "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                f"    if (i >= {rows}) return;",
                "    fom_float s = 0.0;",
            ]
            if _is_csr(md):
                lines += [
                    f"    for (fom_int p = FOM_OP_{up}_ROWPTR[i]; p < FOM_OP_{up}_ROWPTR[i + 1]; "
                    "++p)",
                    f"        s += FOM_OP_{up}_VALS[p] * x[FOM_OP_{up}_COLIDX[p]];",
                ]
            else:
                lines.append(
                    f"    for (fom_int j = 0; j < {cols}; ++j) "
                    f"s += FOM_OP_{up}_DENSE[i * {cols} + j] * x[j];"
                )
            blocks = (rows + _THREADS - 1) // _THREADS
            lines += [
                "    out[i] += s;",
                "}",
                "",
                f'extern "C" void fom_apply_{name}(const fom_float *x, fom_float *out) {{',
                f"    fom_apply_{name}_kern<<<{blocks}, FOM_OPS_THREADS>>>(x, out);",
                "}",
                "",
            ]
        return "\n".join(lines)

    def _objective_source(self, ir: FomIR) -> list[str]:
        """Device ``fom_objective``: one reduction kernel per objective block.

        Each kernel walks its rows grid-stride (fixed grid, so the partial buffer
        is statically sized and the in-run reduction order stable), computes a
        row exactly as the CPU target does, folds it into a shared-memory block
        reduction, and writes one partial per thread block; a one-block kernel
        folds the partials on the device, and only the final scalar comes back.
        Schedule values are read on the host and passed in as kernel arguments.
        """
        if ir.objective is None:
            return []
        obj = ir.objective
        args = objective_c_params(ir)
        names = [a.split("*")[-1] if "*" in a else a.split()[-1] for a in args]
        combine = (
            "sh[threadIdx.x] = sh[threadIdx.x] > sh[threadIdx.x + w] ? sh[threadIdx.x] : "
            "sh[threadIdx.x + w];"
            if obj.norm == "inf"
            else "sh[threadIdx.x] += sh[threadIdx.x + w];"
        )
        lines = [
            "/* ---- verification objective (stopping criterion), on the device ---- */",
            *(['#include "fom_data.h"  /* schedule accessors (host) */'] if ir.schedules else []),
            f"#define FOM_OBJ_THREADS {_THREADS}",
            f"#define FOM_OBJ_BLOCKS {_RED_BLOCKS}",
            *self._objective_arrays(ir, storage=_DEVICE_CONST),
            "",
        ]
        for block in obj.blocks:
            scales = self._objective_scales(ir, block)
            kern_args = args[1:] + [f"fom_float {s}" for s in scales.values()]
            lines += [
                f"__global__ void fom_objective_{block.name}_kern({', '.join(kern_args)},",
                "        fom_float *partials) {",
                "    __shared__ fom_float sh[FOM_OBJ_THREADS];",
                "    fom_float r = 0.0;",
                "    fom_int i, p;",
                f"    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < {block.out_dim};",
                "         i += gridDim.x * blockDim.x) {",
            ]
            # The CPU target's row code verbatim, folding into this thread's r.
            lines += self._objective_row(ir, block, scales, indent="        ")
            lines += [
                "    }",
                "    sh[threadIdx.x] = r;",
                "    __syncthreads();",
                "    for (fom_int w = blockDim.x >> 1; w > 0; w >>= 1) {",
                f"        if (threadIdx.x < w) {combine}",
                "        __syncthreads();",
                "    }",
                "    if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];",
                "    (void)p;",
                "}",
                "",
            ]
        pick = "a > b ? a : b" if obj.norm == "inf" else "a + b"
        lines += [
            "/* Fold the per-block partials on the device: one thread block, the same",
            "   combine as the row kernels; *r starts from the first objective block. */",
            "__device__ static fom_float fom_obj_combine(fom_float a, fom_float b) {",
            f"    return {pick};",
            "}",
            "__global__ void fom_objective_fold_kern(const fom_float *partials, fom_float *r, "
            "int first) {",
            "    __shared__ fom_float sh[FOM_OBJ_THREADS];",
            "    fom_float v = 0.0;",
            "    for (fom_int i = threadIdx.x; i < FOM_OBJ_BLOCKS; i += blockDim.x)",
            "        v = fom_obj_combine(v, partials[i]);",
            "    sh[threadIdx.x] = v;",
            "    __syncthreads();",
            "    for (fom_int w = blockDim.x >> 1; w > 0; w >>= 1) {",
            "        if (threadIdx.x < w)",
            "            sh[threadIdx.x] = fom_obj_combine(sh[threadIdx.x], sh[threadIdx.x + w]);",
            "        __syncthreads();",
            "    }",
            "    if (threadIdx.x == 0) *r = first ? sh[0] : fom_obj_combine(*r, sh[0]);",
            "}",
            "",
            "static fom_float *fom_obj_partials_d = 0, *fom_obj_r_d = 0;",
            "",
            "/* Computed on the device; only the final scalar comes back. */",
            f'extern "C" fom_float fom_objective({", ".join(args)}) {{',
            "    fom_float r;",
            "    if (!fom_obj_partials_d) {",
            "        cudaMalloc((void **)&fom_obj_partials_d, sizeof(fom_float) * FOM_OBJ_BLOCKS);",
            "        cudaMalloc((void **)&fom_obj_r_d, sizeof(fom_float));",
            "    }",
        ]
        if not any(self._objective_scales(ir, block) for block in obj.blocks):
            lines.append("    (void)it;")
        for k, block in enumerate(obj.blocks):
            scales = self._objective_scales(ir, block)
            values = [f"fom_sched_{block.linear_op.terms[j].scale}(it)" for j in scales]
            call = ", ".join(names[1:] + values + ["fom_obj_partials_d"])
            lines += [
                f"    fom_objective_{block.name}_kern<<<FOM_OBJ_BLOCKS, FOM_OBJ_THREADS>>>({call});",
                "    fom_objective_fold_kern<<<1, FOM_OBJ_THREADS>>>(fom_obj_partials_d, "
                f"fom_obj_r_d, {int(k == 0)});",
            ]
        lines += [
            "    cudaMemcpy(&r, fom_obj_r_d, sizeof(fom_float), cudaMemcpyDeviceToHost);",
            "    return r;",
            "}",
            "",
        ]
        return lines

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
            "   pointers and everything runs on the GPU. CG systems: CSR SpMV +",
            "   reduction kernels + a diagonal Jacobi preconditioner. Direct systems",
            "   (LDL^T / LU): cuDSS factorizes M once, at the first solve, and each",
            "   call is one cuDSS solve. No matrix inverse is ever formed or used. */",
            "",
        ]
        for name, fac in ir.factorizations.items():
            lines.append(f"#define FOM_FAC_{name.upper()}_DIM {fac.dim}")
            where = "device CG" if fac.method == "cg" else f"cuDSS {fac.method.upper()}"
            lines.append(
                f"void fom_lin_solve_{name}(const fom_float *b, fom_float *x);  "
                f"/* {where}, dim {fac.dim}, nnz {fac.nnz} */"
            )
            lines.append("")
        lines += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_LINSOLVE_H */", ""]
        return "\n".join(lines)

    def _linsolve_source(self, ir: FomIR) -> str:
        lines = [
            "/* Deterministic linear solves for the CUDA target, all on the device:",
            "   CG kernels below, and cuDSS for the direct (LDL^T / LU) systems.",
            "   No matrix inverse is ever formed or used. */",
            '#include "fom_linsolve.h"',
            "",
            "#include <cuda_runtime.h>",
            "",
        ]
        if any(fac.method == "cg" for fac in ir.factorizations.values()):
            lines.append(self._cg_shared_helpers())
            lines.append("")
        if _direct(ir):
            lines += self._cudss_shared(ir)
        for name, fac in ir.factorizations.items():
            if fac.method == "cg":
                lines.append(self._cg_routine_cu(name, fac))
            else:
                lines += self._cudss_routine(name, fac)
            lines.append("")
        return "\n".join(lines)

    def _cudss_shared(self, ir: FomIR) -> list[str]:
        value = "CUDSS_R_32F" if ir.dtype == "float32" else "CUDSS_R_64F"
        return [
            "#include <stdio.h>",
            "#include <cudss.h>",
            "",
            f"#define FOM_DSS_VALUE {value}",
            "/* cuDSS 0.8 types the CSR row offsets separately from the column indices. */",
            "#if CUDSS_VERSION >= 800",
            "#define FOM_DSS_CSR_TYPES CUDSS_R_32I, CUDSS_R_32I",
            "#else",
            "#define FOM_DSS_CSR_TYPES CUDSS_R_32I",
            "#endif",
            "",
            "static cudssHandle_t fom_dss_handle = 0;",
            "",
            "static void fom_dss_check(cudssStatus_t s, const char *what) {",
            "    if (s != CUDSS_STATUS_SUCCESS)",
            '        fprintf(stderr, "fom_linsolve: cuDSS %s failed (status %d)\\n", what, (int)s);',
            "}",
            "",
        ]

    @staticmethod
    def _cudss_lu_pivoting(config: str) -> list[str]:
        """Global row (partial) pivoting for LU, as the CPU path's LU does: cuDSS's
        default local pivoting perturbs small pivots instead, which loses accuracy
        on an ill-conditioned indefinite system. Needs a COLAMD ordering."""
        return [
            "#if CUDSS_VERSION < 800",
            '#error "pivoted LU needs cuDSS >= 0.8 (CUDSS_PIVOT_GLOBAL_ROW)"',
            "#endif",
            "        {",
            "            cudssReorderingAlg_t reorder = CUDSS_REORDERING_ALG_COLAMD;",
            "            cudssPivotType_t pivot = CUDSS_PIVOT_GLOBAL_ROW;",
            f"            fom_dss_check(cudssConfigSet({config}, CUDSS_CONFIG_REORDERING_ALG, &reorder,",
            '                                         sizeof(reorder)), "reordering");',
            f"            fom_dss_check(cudssConfigSet({config}, CUDSS_CONFIG_PIVOT_TYPE, &pivot,",
            '                                         sizeof(pivot)), "pivoting");',
            "        }",
        ]

    def _cudss_routine(self, name: str, fac) -> list[str]:
        """A direct system through cuDSS: analysis and factorization of the fixed
        ``M`` at the first call (``M`` is device data, so nothing leaves the
        GPU), then one cuDSS solve per call on the caller's device vectors."""
        up = name.upper()
        n = f"FOM_FAC_{up}_DIM"
        rowptr, colidx, vals = csr_arrays(fac.A.to_scipy())
        # The symmetric part for LDL^T (indefinite KKT systems pivot symmetrically).
        mtype = "CUDSS_MTYPE_SYMMETRIC" if fac.method == "ldlt" else "CUDSS_MTYPE_GENERAL"
        h = "fom_dss_handle"
        sys = f"fom_dss_{name}"
        args = f"{h}, {sys}_config, {sys}_data, {sys}_A, {sys}_x, {sys}_b"
        return [
            f"/* ---- solve '{name}' ({fac.method.upper()}, cuDSS on the device) ---- */",
            _c_int_array(f"FOM_FAC_{up}_A_ROWPTR", rowptr, storage=_DEVICE_CONST),
            _c_int_array(f"FOM_FAC_{up}_A_COLIDX", colidx, storage=_DEVICE_CONST),
            _c_array(f"FOM_FAC_{up}_A_VALS", vals, storage=_DEVICE_CONST),
            "",
            f"static cudssConfig_t {sys}_config;",
            f"static cudssData_t {sys}_data;",
            f"static cudssMatrix_t {sys}_A = 0, {sys}_b, {sys}_x;",
            "",
            f'extern "C" void fom_lin_solve_{name}(const fom_float *b, fom_float *x) {{',
            f"    if (!{sys}_A) {{",
            "        void *rowptr, *colidx, *vals;",
            f"        cudaGetSymbolAddress(&rowptr, FOM_FAC_{up}_A_ROWPTR);",
            f"        cudaGetSymbolAddress(&colidx, FOM_FAC_{up}_A_COLIDX);",
            f"        cudaGetSymbolAddress(&vals, FOM_FAC_{up}_A_VALS);",
            f'        if (!{h}) fom_dss_check(cudssCreate(&{h}), "create");',
            f'        fom_dss_check(cudssConfigCreate(&{sys}_config), "config");',
            *(self._cudss_lu_pivoting(f"{sys}_config") if fac.method == "lu" else []),
            f'        fom_dss_check(cudssDataCreate({h}, &{sys}_data), "data");',
            f"        fom_dss_check(cudssMatrixCreateCsr(&{sys}_A, {n}, {n}, {fac.A.nnz}, rowptr, "
            "NULL, colidx, vals,",
            f"                                           FOM_DSS_CSR_TYPES, FOM_DSS_VALUE, {mtype},",
            "                                           CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),",
            '                      "matrix");',
            f"        fom_dss_check(cudssMatrixCreateDn(&{sys}_b, {n}, 1, {n}, b, FOM_DSS_VALUE,",
            '                                          CUDSS_LAYOUT_COL_MAJOR), "rhs");',
            f"        fom_dss_check(cudssMatrixCreateDn(&{sys}_x, {n}, 1, {n}, x, FOM_DSS_VALUE,",
            '                                          CUDSS_LAYOUT_COL_MAJOR), "solution");',
            f'        fom_dss_check(cudssExecute({h}, CUDSS_PHASE_ANALYSIS, {args[len(h) + 2 :]}), "analysis");',
            f"        fom_dss_check(cudssExecute({h}, CUDSS_PHASE_FACTORIZATION, "
            f'{args[len(h) + 2 :]}), "factorization");',
            "    } else {",
            f"        cudssMatrixSetValues({sys}_b, b);",
            f"        cudssMatrixSetValues({sys}_x, x);",
            "    }",
            f'    fom_dss_check(cudssExecute({h}, CUDSS_PHASE_SOLVE, {args[len(h) + 2 :]}), "solve");',
            "}",
        ]

    def _cg_shared_helpers(self) -> str:
        """Kernels shared by every device CG solve: a two-stage dot product into a
        device scalar, the scalar steps of the iteration, and the vector updates.
        The scalars stay on the device; the host reads back only the stop flag."""
        return "\n".join(
            [
                f"#define FOM_RED_THREADS {_THREADS}",
                f"#define FOM_RED_BLOCKS {_RED_BLOCKS}",
                "",
                "/* CG scalars, kept in device memory (one slot array per system). */",
                "enum { FOM_CG_BB, FOM_CG_THRESH, FOM_CG_RZ, FOM_CG_PAP, FOM_CG_ALPHA, FOM_CG_RR,",
                "       FOM_CG_RZ_NEW, FOM_CG_BETA, FOM_CG_STOP, FOM_CG_SLOTS };",
                "",
                "/* Grid-stride dot product with a shared-memory block reduction, then",
                "   one block folds the per-block partials into *out. The fixed grid keeps",
                "   the in-run floating-point reduction order stable (it still differs",
                "   from the serial NumPy oracle; CG converges to cg_tol, so both land",
                "   well within the verify tolerance). */",
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
                "__global__ void fom_sum_partials_kern(const fom_float *partials, fom_float *out) {",
                "    __shared__ fom_float sh[FOM_RED_THREADS];",
                "    fom_float s = 0.0;",
                "    for (fom_int i = threadIdx.x; i < FOM_RED_BLOCKS; i += blockDim.x) "
                "s += partials[i];",
                "    sh[threadIdx.x] = s;",
                "    __syncthreads();",
                "    for (fom_int w = blockDim.x >> 1; w > 0; w >>= 1) {",
                "        if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];",
                "        __syncthreads();",
                "    }",
                "    if (threadIdx.x == 0) *out = sh[0];",
                "}",
                "",
                "static fom_float *fom_dot_partials_d = 0;",
                "",
                "/* *out = a . b, computed and stored on the device. */",
                "static void fom_device_dot(const fom_float *a, const fom_float *b, fom_int n, "
                "fom_float *out) {",
                "    if (!fom_dot_partials_d)",
                "        cudaMalloc((void **)&fom_dot_partials_d, "
                "sizeof(fom_float) * FOM_RED_BLOCKS);",
                "    fom_dot_kern<<<FOM_RED_BLOCKS, FOM_RED_THREADS>>>(a, b, n, "
                "fom_dot_partials_d);",
                "    fom_sum_partials_kern<<<1, FOM_RED_THREADS>>>(fom_dot_partials_d, out);",
                "}",
                "",
                "/* The scalar steps of one CG iteration, on one device thread. */",
                "__global__ void fom_cg_start_kern(fom_float *s, fom_float tol2) {",
                "    s[FOM_CG_THRESH] = tol2 * s[FOM_CG_BB];",
                "    s[FOM_CG_STOP] = (s[FOM_CG_BB] == 0.0);",
                "}",
                "/* alpha = rz / pAp, or stop (alpha = 0, a no-op update) on breakdown. */",
                "__global__ void fom_cg_alpha_kern(fom_float *s) {",
                "    s[FOM_CG_STOP] = (s[FOM_CG_PAP] == 0.0);",
                "    s[FOM_CG_ALPHA] = s[FOM_CG_STOP] ? 0.0 : s[FOM_CG_RZ] / s[FOM_CG_PAP];",
                "}",
                "__global__ void fom_cg_converged_kern(fom_float *s) {",
                "    if (s[FOM_CG_RR] <= s[FOM_CG_THRESH]) s[FOM_CG_STOP] = 1.0;",
                "}",
                "__global__ void fom_cg_beta_kern(fom_float *s) {",
                "    s[FOM_CG_BETA] = s[FOM_CG_RZ_NEW] / s[FOM_CG_RZ];",
                "    s[FOM_CG_RZ] = s[FOM_CG_RZ_NEW];",
                "}",
                "",
                "/* x += alpha p; r -= alpha Ap (fused) */",
                "__global__ void fom_cg_update_xr_kern(fom_float *x, fom_float *r, "
                "const fom_float *p,",
                "                                      const fom_float *Ap, const fom_float *s, "
                "fom_int n) {",
                "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    x[i] += s[FOM_CG_ALPHA] * p[i];",
                "    r[i] -= s[FOM_CG_ALPHA] * Ap[i];",
                "}",
                "",
                "/* p = z + beta p */",
                "__global__ void fom_cg_update_p_kern(fom_float *p, const fom_float *z, "
                "const fom_float *s, fom_int n) {",
                "    fom_int i = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    p[i] = z[i] + s[FOM_CG_BETA] * p[i];",
                "}",
            ]
        )

    def _cg_routine_cu(self, name: str, fac) -> str:
        """Device-resident Jacobi-preconditioned CG for one SPD system.

        Mirrors :func:`fomtools.codegen.factorize._pcg_solve` exactly (same
        iteration, same stopping test ``r.r <= tol^2 (b.b)``, same breakdown
        exit); every vector, dot product and scalar step runs on the device, and
        the host, which drives the loop, reads back only the stop flag.
        """
        up = name.upper()
        n = f"FOM_FAC_{up}_DIM"
        tol2 = _fmt_float(fac.cg_tol * fac.cg_tol)
        maxit = int(fac.cg_max_iter)
        rowptr, colidx, vals = csr_arrays(fac.A.to_scipy())
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
            f"static fom_float *fom_cg_{name}_p = 0, *fom_cg_{name}_Ap = 0, *fom_cg_{name}_s = 0;",
            "",
            "/* Every vector and scalar stays on the device; the host reads back only",
            "   the stop flag (one scalar per iteration) to end the loop. */",
            f'extern "C" void fom_lin_solve_{name}(const fom_float *b, fom_float *x) {{',
            f"    const fom_int n = {n};",
            "    const fom_int blocks = (n + FOM_RED_THREADS - 1) / FOM_RED_THREADS;",
            "    fom_float *s, stop;",
            "    fom_int it;",
            f"    if (!fom_cg_{name}_r) {{",
            f"        cudaMalloc((void **)&fom_cg_{name}_r, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_z, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_p, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_Ap, sizeof(fom_float) * n);",
            f"        cudaMalloc((void **)&fom_cg_{name}_s, sizeof(fom_float) * FOM_CG_SLOTS);",
            "    }",
            f"    s = fom_cg_{name}_s;",
            f"    fom_cg_{name}_init_kern<<<blocks, FOM_RED_THREADS>>>(b, x, fom_cg_{name}_r,",
            f"                                                        fom_cg_{name}_z, "
            f"fom_cg_{name}_p);",
            "    fom_device_dot(b, b, n, s + FOM_CG_BB);",
            f"    fom_cg_start_kern<<<1, 1>>>(s, (fom_float){tol2});",
            "    cudaMemcpy(&stop, s + FOM_CG_STOP, sizeof(fom_float), cudaMemcpyDeviceToHost);",
            "    if (stop != 0.0) return;  /* b == 0: x was zeroed by the init kernel */",
            f"    fom_device_dot(fom_cg_{name}_r, fom_cg_{name}_z, n, s + FOM_CG_RZ);",
            f"    for (it = 0; it < {maxit}; ++it) {{",
            f"        fom_cg_{name}_spmv_kern<<<blocks, FOM_RED_THREADS>>>(fom_cg_{name}_p, "
            f"fom_cg_{name}_Ap);",
            f"        fom_device_dot(fom_cg_{name}_p, fom_cg_{name}_Ap, n, s + FOM_CG_PAP);",
            "        fom_cg_alpha_kern<<<1, 1>>>(s);",
            f"        fom_cg_update_xr_kern<<<blocks, FOM_RED_THREADS>>>(x, fom_cg_{name}_r,",
            f"                                                          fom_cg_{name}_p, "
            f"fom_cg_{name}_Ap, s, n);",
            f"        fom_device_dot(fom_cg_{name}_r, fom_cg_{name}_r, n, s + FOM_CG_RR);",
            "        fom_cg_converged_kern<<<1, 1>>>(s);",
            "        cudaMemcpy(&stop, s + FOM_CG_STOP, sizeof(fom_float), "
            "cudaMemcpyDeviceToHost);",
            "        if (stop != 0.0) break;",
            f"        fom_cg_{name}_precond_kern<<<blocks, FOM_RED_THREADS>>>(fom_cg_{name}_r, "
            f"fom_cg_{name}_z);",
            f"        fom_device_dot(fom_cg_{name}_r, fom_cg_{name}_z, n, s + FOM_CG_RZ_NEW);",
            "        fom_cg_beta_kern<<<1, 1>>>(s);",
            f"        fom_cg_update_p_kern<<<blocks, FOM_RED_THREADS>>>(fom_cg_{name}_p, "
            f"fom_cg_{name}_z, s, n);",
            "    }",
            "}",
        ]
        return "\n".join(out)

    # -- build ---------------------------------------------------------------
    def _makefile(self, ir: FomIR | None = None) -> str:
        guard, guard_rules = self._objective_guard(ir)
        if ir is not None and _direct(ir):
            cudss = [
                "# cuDSS, for the direct linear solves on the device: CUDSS_HOME, by default",
                "# the nvidia-cudss wheel found when this directory was generated.",
                f"CUDSS_HOME ?= {_cudss_home()}",
                "CUDSS_LIB = $(firstword $(wildcard $(CUDSS_HOME)/lib/libcudss.so "
                "$(CUDSS_HOME)/lib/libcudss.so.0 $(CUDSS_HOME)/lib64/libcudss.so))",
                "LINSOLVE_FLAGS = -I$(CUDSS_HOME)/include",
                "# nvcc rejects a library file named *.so.0 (the wheel's only name): hand",
                "# it to the host linker directly.",
                "LDLIBS = -Xlinker $(CUDSS_LIB) -Xlinker -rpath=$(dir $(CUDSS_LIB))",
            ]
        else:
            cudss = ["LINSOLVE_FLAGS =", "LDLIBS ="]
        return "\n".join(
            [
                "CC ?= cc",
                "NVCC ?= nvcc",
                "CFLAGS ?= -O3 -Iinclude -Itest",
                "NVCCFLAGS ?= -O3 -Iinclude -Itest",
                *cudss,
                "",
                "CU_SRC = src/fom_solver.cu src/fom_ops.cu src/fom_linsolve.cu "
                "src/fom_objective.cu",
                "",
                ".PHONY: compile verify bench lib clean objective_guard",
                "",
                "fom_solver.o: src/fom_solver.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "fom_ops.o: src/fom_ops.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "fom_linsolve.o: src/fom_linsolve.cu",
                *(
                    [
                        '\t@test -n "$(CUDSS_HOME)" || (echo "cuDSS not found: pip install '
                        'nvidia-cudss-cu13, or set CUDSS_HOME"; exit 1)'
                    ]
                    if ir is not None and _direct(ir)
                    else []
                ),
                "\t$(NVCC) $(NVCCFLAGS) $(LINSOLVE_FLAGS) -c $< -o $@",
                "fom_objective.o: src/fom_objective.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "fom_data.o: src/fom_data.c",
                "\t$(CC) $(CFLAGS) -c $< -o $@",
                "",
                "OBJS = fom_solver.o fom_ops.o fom_linsolve.o fom_objective.o fom_data.o",
                "# Verify only, where fom_ops.cu / fom_linsolve.cu are editable: the host",
                "# reference matvecs and the device staging of the direct checks.",
                "CHECK_OBJS = $(if $(wildcard test/ops_ref.c),ops_ref.o) "
                "$(if $(wildcard test/check_stage.cu),check_stage.o)",
                "ops_ref.o: test/ops_ref.c",
                "\t$(CC) $(CFLAGS) -c $< -o $@",
                "check_stage.o: test/check_stage.cu",
                "\t$(NVCC) $(NVCCFLAGS) -c $< -o $@",
                "",
                "# Compile-check: build BOTH the verify and bench binaries (no GPU run).",
                "# Building bench_bin here too means a CI compile-check catches bench",
                "# breakage even though benchmarking itself needs a GPU.",
                f"compile:{guard} $(OBJS) $(CHECK_OBJS) test/verify_main.c test/bench_main.c",
                _KERNEL_CHECK_RECIPE,
                "\t$(CC) $(CFLAGS) -c test/verify_main.c -o verify_main.o",
                "\t$(NVCC) $(NVCCFLAGS) $(OBJS) $(CHECK_OBJS) verify_main.o -o verify_bin $(LDLIBS)",
                "\t$(CC) $(CFLAGS) -c test/bench_main.c -o bench_main.o",
                "\t$(NVCC) $(NVCCFLAGS) $(OBJS) bench_main.o -o bench_bin $(LDLIBS)",
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
                "\t$(NVCC) $(NVCCFLAGS) $(LINSOLVE_FLAGS) --shared -Xcompiler -fPIC $(CU_SRC) "
                "src/fom_data.c -o libfomsolver.so $(LDLIBS)",
                "",
                "clean:",
                "\trm -f verify_bin bench_bin *.o libfomsolver.so",
                "",
                *guard_rules,
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
                "where structured), `src/fom_linsolve.cu` (device CG, or cuDSS for",
                "LDL^T/LU; set CUDSS_HOME if cuDSS was not found at generation) and",
                "`src/fom_objective.cu` (the verification objective). All",
                "`fom_apply_*` / `fom_lin_solve_*` arguments are DEVICE pointers.",
                "Host-side numeric data lives in `src/fom_data.c`. See `SPEC.md`.",
                "",
            ]
        )
