"""Embedded C target: OSQP-style, statically sized, allocation- and division-free.

Emits (deterministically, *not* the solver body):

* ``include/fom_workspace.h`` — dims as ``#define``s + the ABI structs.
* ``include/fom_solver.h``    — the public ``fom_setup/update_params/solve/cleanup`` ABI.
* ``include/fom_data.h``      — extern declarations + per-matrix dim macros.
* ``src/fom_data.c``          — all numeric coefficients as ``static const`` arrays.
* ``test/verify_data.h``      — random parameter samples + oracle-expected solutions.
* ``test/verify_main.c``      — runs the solver over the samples and checks them.
* ``Makefile``                — ``make verify`` / ``make lib`` targets.
* ``README.md``, ``SPEC.md``, ``spec.json``, ``data.npz`` (the IR, at the
  output root), and the Python wrapper (via :mod:`fomtools.codegen.wrapper`).

The coding agent (or a golden fixture) supplies ``src/fom_solver.c``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as spa

from fomtools.codegen import spec
from fomtools.codegen.ir import ActivationKind, FomIR, csr_arrays
from fomtools.codegen.objective import objective_c_params, objective_c_source
from fomtools.codegen.oracle import ReferenceOracle
from fomtools.codegen.targets.base import (
    TargetDescriptor,
    objective_violations,
    strip_c_comments_and_strings,
)


def _fmt_float(x: float) -> str:
    return f"{float(x):.17g}"


def _c_array(
    name: str, values: np.ndarray, ctype: str = "fom_float", storage: str = "static const"
) -> str:
    flat = np.asarray(values, dtype=float).reshape(-1)
    body = ", ".join(_fmt_float(v) for v in flat)
    return f"{storage} {ctype} {name}[{flat.size}] = {{{body}}};"


def _c_extern(name: str, ctype: str = "fom_float") -> str:
    return f"extern const {ctype} {name}[];"


def _c_int_array(name: str, values, storage: str = "static const") -> str:
    flat = np.asarray(values).reshape(-1).astype(int)
    body = ", ".join(str(int(v)) for v in flat)
    return f"{storage} fom_int {name}[{flat.size}] = {{{body}}};"


def _strict_triangle(csc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(indptr, indices, data)`` of a triangular CSC factor without its diagonal.

    The emitted solves keep the diagonal elsewhere (unit for ``L``, the reciprocals
    ``UINV`` for ``U``), so their inner loops touch off-diagonal entries only and
    need no per-entry branch.
    """
    m = spa.csc_matrix((csc.data, csc.indices, csc.indptr), shape=(csc.n_rows, csc.n_cols))
    m = (m - spa.diags(m.diagonal())).tocsc()
    m.eliminate_zeros()
    m.sort_indices()
    return m.indptr, m.indices, m.data


def _factor_arrays(name: str, fac) -> list[str]:
    """The constant arrays of a direct factorization: permutation, the strictly
    lower ``L``, and the reciprocal pivots (LDL^T) or the strictly upper ``U`` and
    reciprocal diagonal (LU)."""
    up = name.upper()
    lines = [_c_int_array(f"FOM_FAC_{up}_PERM", fac.perm)]
    indptr, indices, data = _strict_triangle(fac.L)
    lines.append(_c_int_array(f"FOM_FAC_{up}_L_INDPTR", indptr))
    lines.append(_c_int_array(f"FOM_FAC_{up}_L_INDICES", indices))
    lines.append(_c_array(f"FOM_FAC_{up}_L_DATA", data))
    if fac.method == "ldlt":
        lines.append(_c_array(f"FOM_FAC_{up}_DINV", fac.dinv))
    else:
        indptr, indices, data = _strict_triangle(fac.U)
        lines.append(_c_int_array(f"FOM_FAC_{up}_U_INDPTR", indptr))
        lines.append(_c_int_array(f"FOM_FAC_{up}_U_INDICES", indices))
        lines.append(_c_array(f"FOM_FAC_{up}_U_DATA", data))
        lines.append(_c_array(f"FOM_FAC_{up}_UINV", fac.uinv))
    return lines


#: Extra ``check_termination`` intervals ``make verify`` runs an objective-terminated
#: solver at, besides its default: every iteration, and one that divides nothing
#: in sight, so a solver that ignores the runtime setting cannot pass.
_VERIFY_CHECK_INTERVALS = (1, 7)


def _live_terms(ir: FomIR, lin) -> list[tuple[int, object]]:
    """``(index, term)`` of a linear op's terms whose block has a nonzero entry.

    A scheduled block can be identically zero in its base (it varies only
    through its slope); its row loop would add nothing and its CSR arrays would
    be empty, which C does not allow, so the emitters skip it.
    """
    return [(k, t) for k, t in enumerate(lin.terms) if ir.matrices[t.matrix].nnz() > 0]


def _objective_matrix_names(ir: FomIR) -> list[str]:
    """Names of the coefficient blocks and constants the objective reads."""
    names: list[str] = []
    if ir.objective is None:
        return names
    for lin in ir.objective.linops():
        terms = [t.matrix for _, t in _live_terms(ir, lin)]
        for name in terms + ([lin.const] if lin.const else []):
            if name not in names:
                names.append(name)
    return names


def _objective_only_names(ir: FomIR) -> set[str]:
    """Objective blocks nothing else reads. They are emitted next to
    ``fom_objective`` in the ops source, so the data source leaves them out."""
    used = set()
    for lin in [step.linear_op for step in ir.body] + [ir.state_update]:
        used.update(t.matrix for t in lin.terms)
        if lin.const:
            used.add(lin.const)
    for step in ir.body:
        used.update(b for b in (step.activation.lower, step.activation.upper) if b)
    return set(_objective_matrix_names(ir)) - used


def _matvec_matrix_names(ir: FomIR) -> list[str]:
    """Names of matrices used as ``matrix @ source`` in any linear op."""
    names: list[str] = []
    seen = set()
    ops = [s.linear_op for s in ir.body] + [ir.state_update]
    for op in ops:
        for term in op.terms:
            if term.matrix not in seen:
                seen.add(term.matrix)
                names.append(term.matrix)
    return names


def _is_csr(md) -> bool:
    """Emit a matrix as sparse CSR when it is large and structurally sparse."""
    shape = md.shape
    rows = shape[0]
    cols = shape[1] if len(shape) > 1 else 1
    size = rows * cols
    nnz = md.nnz()
    return size >= 16 and 2 * nnz < size


def _c_array_2d(name: str, mat: np.ndarray, decl_dims: str, ctype: str = "fom_float") -> str:
    """Emit a 2-D ``static const`` array with proper nested braces per row."""
    mat = np.atleast_2d(np.asarray(mat, dtype=float))
    rows = ", ".join("{" + ", ".join(_fmt_float(v) for v in row) + "}" for row in mat)
    return f"static const {ctype} {name}{decl_dims} = {{{rows}}};"


def _c_int_array_2d(name: str, rows, decl_dims: str) -> str:
    """Emit a 2-D ``static const fom_int`` array with nested braces per row."""
    body = ", ".join("{" + ", ".join(str(int(v)) for v in row) + "}" for row in rows)
    return f"static const fom_int {name}{decl_dims} = {{{body}}};"


class EmbeddedCTarget(TargetDescriptor):
    name = "embedded_c"

    #: Emit the CSR arrays of sparse matvec matrices into the package-owned
    #: ``fom_data.c`` (read by ``fom_ops.c``), so a rewritten ``fom_ops.c`` cannot
    #: change the data it is checked against. CUDA keeps them device-side in
    #: ``fom_ops.cu`` instead, to avoid a second (host) copy.
    csr_in_data = True

    #: Annotate data-parallel loops (matrix-vector products and the CG kernels)
    #: with optional OpenMP pragmas. Off for the embedded target (kept simple and
    #: serial); large-scale CPU targets switch it on. The triangular solves of the
    #: direct factorizations are inherently sequential and are never annotated.
    parallel = False

    def _omp_for(self, clause: str = "", indent: str = "    ") -> list[str]:
        """Lines for an optional ``#pragma omp parallel for``.

        Returns nothing unless the target opts into parallelism. The pragma is
        guarded by ``#ifdef _OPENMP`` so the default (serial) build compiles it
        away without an unknown-pragma warning, and it only becomes active when
        the user builds with ``OMPFLAGS=-fopenmp`` — i.e. threading is optional.
        """
        if not self.parallel:
            return []
        pragma = "#pragma omp parallel for" + (f" {clause}" if clause else "")
        return [f"{indent}#ifdef _OPENMP", f"{indent}{pragma}", f"{indent}#endif"]

    def solver_source_path(self) -> str:
        return "src/fom_solver.c"

    def verify_command(self) -> list[str]:
        return ["make", "verify"]

    # -- scaffolding -------------------------------------------------------
    def scaffold(self, ir: FomIR, out_dir: Path) -> None:
        out = Path(out_dir)
        (out / "include").mkdir(parents=True, exist_ok=True)
        (out / "src").mkdir(parents=True, exist_ok=True)
        (out / "test").mkdir(parents=True, exist_ok=True)

        # IR spec (for the agent + harness). Lives at the output root so it is
        # the first thing a developer or the agent sees; nothing in the build
        # consumes it, so it is intentionally not buried under test/.
        spec.save(ir, out)
        (out / "SPEC.md").write_text(spec.render_markdown(ir))

        (out / "include" / "fom_workspace.h").write_text(self._workspace_header(ir))
        (out / "include" / "fom_solver.h").write_text(self._solver_header())
        (out / "include" / "fom_data.h").write_text(self._data_header(ir))
        (out / "include" / "fom_profile.h").write_text(self._profile_header())
        (out / "include" / "fom_linsolve.h").write_text(self._linsolve_header(ir))
        (out / self.linsolve_source_path()).write_text(self._linsolve_source(ir))
        (out / "include" / "fom_ops.h").write_text(self._ops_header(ir))
        (out / self.ops_source_path()).write_text(self._ops_source(ir))
        (out / "src" / "fom_data.c").write_text(self._data_source(ir))
        (out / "src" / "fom_profile.c").write_text(self._profile_source())
        (out / "test" / "verify_data.h").write_text(self._verify_data(ir))
        (out / "test" / "verify_main.c").write_text(self._verify_main(ir))
        (out / "test" / "bench_main.c").write_text(self._bench_main(ir))
        (out / "Makefile").write_text(self._makefile(ir))
        (out / "README.md").write_text(self._readme(ir))

    # -- headers -----------------------------------------------------------
    def _workspace_header(self, ir: FomIR) -> str:
        c_float = self.config.c_float
        term = ir.termination
        tol = term.tol if term.tol is not None else 0.0
        # The macro is clamped to >= 1 so `% FOM_CHECK_TERMINATION` is safe
        # even when checking is disabled (check_termination == 0). Not for the
        # objective: its loop guards `check > 0` and still tests the last
        # iteration, so 0 has to reach the runtime setting as 0.
        use_tol = 1 if term.uses_tolerance else 0
        check = term.check_termination if term.uses_objective else max(term.check_termination, 1)

        lines = [
            "#ifndef FOM_WORKSPACE_H",
            "#define FOM_WORKSPACE_H",
            "",
            f"typedef {c_float} fom_float;",
            "typedef int fom_int;",
            "",
            f"#define FOM_STATE_DIM {ir.state_dim}",
            f"#define FOM_MAX_ITER {term.max_iters}",
            f"#define FOM_TOL ((fom_float){_fmt_float(tol)})",
            f"#define FOM_USE_TOL {use_tol}",
            # 1 when the loop stops on the verification objective: call
            # fom_objective (fom_ops.h) on check iterations and the last one.
            f"#define FOM_USE_OBJECTIVE {1 if term.uses_objective else 0}",
            f"#define FOM_CHECK_TERMINATION {check}",
            f"#define FOM_NUM_PARAMS {len(ir.params)}",
            f"#define FOM_MAX_VEC_DIM {ir.max_rhs_dim}",
            # 1 when the body is the whole recorded run rather than one
            # repeatable iteration, so it must be executed exactly once and
            # settings.max_iter must not gate it.
            f"#define FOM_UNROLLED {1 if ir.unrolled else 0}",
            "",
            "#define FOM_SOLVED 0",
            "#define FOM_MAX_ITER_REACHED 1",
            "",
        ]
        for p in ir.params:
            lines.append(f"#define FOM_PARAM_{p.name.upper()}_DIM {p.dim}")
        for step in ir.body:
            lines.append(f"#define FOM_{step.name.upper()}_DIM {step.out_dim}")
        lines.append("")

        # FomSettings. ``check_termination`` is runtime-tunable (initialized to
        # the generated FOM_CHECK_TERMINATION default in fom_set_default_settings):
        # callers may change how often the inf-norm change criterion is evaluated
        # without regenerating, mirroring OSQP's setting of the same name.
        lines += [
            "typedef struct {",
            "    fom_int max_iter;",
            "    fom_float tol;",
            "    fom_int check_termination;",
            "} FomSettings;",
            "",
        ]
        # FomParams
        lines.append("typedef struct {")
        if ir.params:
            for p in ir.params:
                lines.append(f"    fom_float {p.name}[FOM_PARAM_{p.name.upper()}_DIM];")
        else:
            lines.append("    fom_int _unused;")
        lines += ["} FomParams;", ""]
        # FomSolution
        lines += [
            "typedef struct {",
            "    fom_float x[FOM_STATE_DIM];",
            "    fom_int iter;",
            "    fom_float residual;",
            "    fom_int status;",
            "} FomSolution;",
            "",
        ]
        # FomWorkspace
        lines.append("typedef struct {")
        lines.append("    FomSettings settings;")
        lines.append("    FomParams params;")
        lines.append("    FomSolution solution;")
        lines.append("    fom_float state[FOM_STATE_DIM];")
        lines.append("    fom_float new_state[FOM_STATE_DIM];")
        lines.append("    fom_float acc[FOM_MAX_VEC_DIM];")
        for step in ir.body:
            lines.append(f"    fom_float {step.name}[FOM_{step.name.upper()}_DIM];")
        lines += ["} FomWorkspace;", "", "#endif /* FOM_WORKSPACE_H */", ""]
        return "\n".join(lines)

    def _solver_header(self) -> str:
        return "\n".join(
            [
                "#ifndef FOM_SOLVER_H",
                "#define FOM_SOLVER_H",
                "",
                '#include "fom_workspace.h"',
                "",
                "#ifdef __cplusplus",
                'extern "C" {',
                "#endif",
                "",
                "/* Fill settings with the generated defaults (max_iter, tol). */",
                "void fom_set_default_settings(FomSettings *settings);",
                "",
                "/* Initialize the workspace: copy settings and the initial state. */",
                "void fom_setup(FomWorkspace *work, const FomSettings *settings);",
                "",
                "/* Copy a new parameter vector set into the workspace. */",
                "void fom_update_params(FomWorkspace *work, const FomParams *params);",
                "",
                "/* Run the first-order method. Returns the status code. */",
                "fom_int fom_solve(FomWorkspace *work);",
                "",
                "/* Release resources (no-op for the embedded target). */",
                "void fom_cleanup(FomWorkspace *work);",
                "",
                "#ifdef __cplusplus",
                "}",
                "#endif",
                "",
                "#endif /* FOM_SOLVER_H */",
                "",
            ]
        )

    def _data_header(self, ir: FomIR) -> str:
        lines = [
            "#ifndef FOM_DATA_H",
            "#define FOM_DATA_H",
            "",
            '#include "fom_workspace.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "/* Initial state. */",
            "extern const fom_float FOM_STATE_INIT[FOM_STATE_DIM];",
            "",
        ]
        # Iteration-varying coefficients, tabulated at generation time and read at
        # the 0-based loop counter. Tabulating keeps the iteration division-free
        # and free of any libm call.
        for sched in sorted(ir.schedules):
            upper = sched.upper()
            lines.append(f"/* Schedule {sched}: one value per iteration, indexed by the")
            lines.append("   0-based loop counter. Read it through the accessor below, which")
            lines.append("   cannot run off the end: the loop bound is the runtime")
            lines.append("   settings.max_iter, which a caller may raise above FOM_MAX_ITER.")
            lines.append("   Past the table the last value is held. */")
            lines.append(f"#define FOM_SCHED_{upper}_LEN {len(ir.schedules[sched])}")
            lines.append(_c_extern(f"FOM_SCHED_{sched}"))
            lines.append(f"static inline fom_float fom_sched_{sched}(fom_int it) {{")
            lines.append(
                f"    return FOM_SCHED_{sched}[(it < FOM_SCHED_{upper}_LEN) "
                f"? it : (FOM_SCHED_{upper}_LEN - 1)];"
            )
            lines.append("}")
            lines.append("")
        csr = {n for n in _matvec_matrix_names(ir) if _is_csr(ir.matrices[n])}
        csr |= _objective_only_names(ir)  # emitted with fom_objective in fom_ops.c
        for name, md in ir.matrices.items():
            if name in csr:
                if self.csr_in_data:
                    rows, cols = md.shape[0], md.shape[1] if len(md.shape) > 1 else 1
                    lines.append(f"#define FOM_MAT_{name}_ROWS {rows}")
                    lines.append(f"#define FOM_MAT_{name}_COLS {cols}")
                    lines.append(f"#define FOM_MAT_{name}_NNZ {md.nnz()}")
                    lines.append(_c_extern(f"FOM_MAT_{name}_ROWPTR", "fom_int"))
                    lines.append(_c_extern(f"FOM_MAT_{name}_COLIDX", "fom_int"))
                    lines.append(_c_extern(f"FOM_MAT_{name}_VALS"))
                    lines.append("")
                continue  # sparse matvec matrices are stored as CSR only
            shape = md.shape
            rows = shape[0]
            cols = shape[1] if len(shape) > 1 else 1
            lines.append(f"#define FOM_MAT_{name}_ROWS {rows}")
            lines.append(f"#define FOM_MAT_{name}_COLS {cols}")
            lines.append(_c_extern(f"FOM_MAT_{name}"))
            lines.append("")

        # Activation scalars (soft-threshold lambda; scalar saturated-linear
        # bounds). Vector clip bounds are emitted above as FOM_MAT_* arrays.
        for step in ir.body:
            act = step.activation
            if act.kind == ActivationKind.SOFT_THRESHOLD:
                lines.append(
                    f"#define FOM_{step.name.upper()}_LAMBDA ((fom_float){_fmt_float(act.lambd)})"
                )
            elif act.kind == ActivationKind.SATURATED_LINEAR and act.lower_scalar is not None:
                lines.append(
                    f"#define FOM_{step.name.upper()}_LOWER "
                    f"((fom_float){_fmt_float(act.lower_scalar)})"
                )
                lines.append(
                    f"#define FOM_{step.name.upper()}_UPPER "
                    f"((fom_float){_fmt_float(act.upper_scalar)})"
                )
        lines.append("")
        lines += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_DATA_H */", ""]
        return "\n".join(lines)

    def _data_source(self, ir: FomIR) -> str:
        csr = {n for n in _matvec_matrix_names(ir) if _is_csr(ir.matrices[n])}
        csr |= _objective_only_names(ir)  # emitted with fom_objective in fom_ops.c
        lines = ['#include "fom_data.h"', ""]
        lines.append(_c_array("FOM_STATE_INIT", ir.state.init, storage="const"))
        lines.append("")
        for sched in sorted(ir.schedules):
            lines.append(_c_array(f"FOM_SCHED_{sched}", ir.schedules[sched], storage="const"))
            lines.append("")
        for name, md in ir.matrices.items():
            if name in csr:
                if self.csr_in_data:  # CSR only: never densified
                    rowptr, colidx, vals = md.csr()
                    lines.append(_c_int_array(f"FOM_MAT_{name}_ROWPTR", rowptr, storage="const"))
                    lines.append(_c_int_array(f"FOM_MAT_{name}_COLIDX", colidx, storage="const"))
                    lines.append(_c_array(f"FOM_MAT_{name}_VALS", vals, storage="const"))
                    lines.append("")
                continue
            lines.append(_c_array(f"FOM_MAT_{name}", md.dense(), storage="const"))
            lines.append("")
        return "\n".join(lines)

    # -- matrix-vector products (sparse CSR where structured) --------------
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
            "   sparse, dense otherwise). Accumulating: zero `out` first. */",
            "",
        ]
        for name in _matvec_matrix_names(ir):
            md = ir.matrices[name]
            kind = "CSR" if _is_csr(md) else "dense"
            lines.append(
                f"void fom_apply_{name}(const fom_float *x, fom_float *out);  "
                f"/* {kind} {md.shape[0]}x{md.shape[1] if len(md.shape) > 1 else 1} */"
            )
        lines += self._objective_declaration(ir)
        lines += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_OPS_H */", ""]
        return "\n".join(lines)

    def _objective_declaration(self, ir: FomIR, pointers: str = "") -> list[str]:
        """The ``fom_objective`` prototype and its contract, for ``fom_ops.h``."""
        if ir.objective is None:
            return []
        obj = ir.objective
        what = {
            "inf": "max_i |r_i|",
            "1": "sum_i |r_i|",
            "2": "sum_i r_i^2 (the squared two-norm, as the verifier defines it)",
        }[obj.norm]
        lines = [
            "",
            f"#define FOM_OBJ_DIM {obj.out_dim}",
            "",
            "/* The verification objective r^k, the metric the solver stops on:",
            f"   returns {what}",
            "   over the FOM_OBJ_DIM rows of r^k. Call it at the end of iteration `it`",
            "   (0-based), after new_state = w^k is computed and BEFORE it replaces",
            "   state = w^{k-1}; temporaries are that iteration's values. Stop when the",
            "   returned value is <= settings.tol. Never compute this metric yourself.",
        ]
        if pointers:
            lines.append(f"   {pointers}")
        lines[-1] += " */"
        lines.append(f"fom_float fom_objective({', '.join(objective_c_params(ir))});")
        return lines

    def _ops_source(self, ir: FomIR) -> str:
        lines = ['#include "fom_ops.h"', '#include "fom_data.h"', ""]
        for name in _matvec_matrix_names(ir):
            md = ir.matrices[name]
            shape = md.shape
            rows = shape[0]
            cols = shape[1] if len(shape) > 1 else 1
            if _is_csr(md):
                mat = f"FOM_MAT_{name}"
                lines += [
                    f"void fom_apply_{name}(const fom_float *x, fom_float *out) {{",
                    "    fom_int i, p;",
                ]
                # Rows are independent (each writes a distinct out[i]) -> parallel.
                lines += self._omp_for("private(p)")
                lines += [
                    f"    for (i = 0; i < {rows}; ++i) {{",
                    "        fom_float s = 0.0;",
                    f"        for (p = {mat}_ROWPTR[i]; p < {mat}_ROWPTR[i + 1]; ++p) "
                    f"s += {mat}_VALS[p] * x[{mat}_COLIDX[p]];",
                    "        out[i] += s;",
                    "    }",
                    "}",
                    "",
                ]
            else:
                lines += [
                    f"void fom_apply_{name}(const fom_float *x, fom_float *out) {{",
                    "    fom_int i, j;",
                ]
                lines += self._omp_for("private(j)")
                lines += [
                    f"    for (i = 0; i < {rows}; ++i) {{",
                    "        fom_float s = 0.0;",
                    f"        for (j = 0; j < {cols}; ++j) s += FOM_MAT_{name}[i * {cols} + j] * x[j];",
                    "        out[i] += s;",
                    "    }",
                    "}",
                    "",
                ]
        lines += self._objective_source(ir)
        return "\n".join(lines)

    # -- verification objective ----------------------------------------------
    def _objective_arrays(self, ir: FomIR, storage: str = "static const") -> list[str]:
        """CSR arrays of every objective block, and its constant vectors."""
        lines = []
        for name in _objective_matrix_names(ir):
            md = ir.matrices[name]
            up = name.upper()
            if md.array.ndim < 2:
                lines.append(_c_array(f"FOM_OBJ_{up}", md.array, storage=storage))
                continue
            rowptr, colidx, vals = md.csr()
            lines.append(_c_int_array(f"FOM_OBJ_{up}_ROWPTR", rowptr, storage=storage))
            lines.append(_c_int_array(f"FOM_OBJ_{up}_COLIDX", colidx, storage=storage))
            lines.append(_c_array(f"FOM_OBJ_{up}_VALS", vals, storage=storage))
        return lines

    def _objective_row(self, ir: FomIR, block, scales: dict, indent: str) -> list[str]:
        """C statements computing row ``i`` of ``block`` and folding it into ``r``.

        Same order of operations as the oracle: the constant, then each term in
        turn (scaled ones multiplied by their schedule value), then the
        activation, then the norm.
        """
        lin = block.linear_op
        norm = ir.objective.norm
        first = f"FOM_OBJ_{lin.const.upper()}[i]" if lin.const else "0.0"
        out = [f"{indent}fom_float v = {first};", f"{indent}fom_float s;"]
        for k, term in _live_terms(ir, lin):
            up = term.matrix.upper()
            src = objective_c_source(term.source)
            out += [
                f"{indent}s = 0.0;",
                f"{indent}for (p = FOM_OBJ_{up}_ROWPTR[i]; p < FOM_OBJ_{up}_ROWPTR[i + 1]; ++p)",
                f"{indent}    s += FOM_OBJ_{up}_VALS[p] * {src}[FOM_OBJ_{up}_COLIDX[p]];",
                f"{indent}v += {scales[k]} * s;" if k in scales else f"{indent}v += s;",
            ]
        if block.activation.kind == ActivationKind.RELU:
            out.append(f"{indent}if (v < 0.0) v = 0.0;")
        if norm == "2":
            out.append(f"{indent}r += v * v;")
        else:
            out.append(f"{indent}if (v < 0.0) v = -v;")
            out.append(f"{indent}if (v > r) r = v;" if norm == "inf" else f"{indent}r += v;")
        return out

    @staticmethod
    def _objective_scales(ir: FomIR, block) -> dict[int, str]:
        """Local names of the schedule values a block's scaled terms read."""
        return {
            k: f"sc_{block.name}_{k}"
            for k, term in _live_terms(ir, block.linear_op)
            if term.scale is not None
        }

    def _objective_source(self, ir: FomIR) -> list[str]:
        """``fom_objective``: the verification objective, fused row by row.

        Each row of each block is computed from CSR coefficients and folded into
        the norm at once, so nothing is stored: no scratch, no allocation, and no
        division. Rows are independent, so the loops parallelize as reductions.
        """
        if ir.objective is None:
            return []
        obj = ir.objective
        used = {objective_c_source(t.source) for lin in obj.linops() for t in lin.terms}
        lines = [
            "/* ---- verification objective (stopping criterion) ---- */",
            *self._objective_arrays(ir),
            "",
            f"fom_float fom_objective({', '.join(objective_c_params(ir))}) {{",
            "    fom_float r = 0.0;",
            "    fom_int i, p;",
        ]
        for arg in ("state", "new_state"):
            if arg not in used:
                lines.append(f"    (void){arg};")
        if not any(self._objective_scales(ir, block) for block in obj.blocks):
            lines.append("    (void)it;")
        reduction = "reduction(max : r)" if obj.norm == "inf" else "reduction(+ : r)"
        # Schedule values are fixed within the call: read each once, up front.
        for block in obj.blocks:
            for k, local in self._objective_scales(ir, block).items():
                sched = block.linear_op.terms[k].scale
                lines.append(f"    const fom_float {local} = fom_sched_{sched}(it);")
        for block in obj.blocks:
            lines.append(
                f"    /* {block.name}: {block.out_dim} row(s), "
                f"{block.activation.kind.value} activation */"
            )
            lines += self._omp_for(f"private(p) {reduction}")
            lines.append(f"    for (i = 0; i < {block.out_dim}; ++i) {{")
            lines += self._objective_row(
                ir, block, self._objective_scales(ir, block), indent="        "
            )
            lines.append("    }")
        lines += ["    return r;", "}", ""]
        return lines

    # -- linear solves (factorizations, never inverses) --------------------
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
            "/* Solve M x = b for each affine operator. Direct methods (LDL^T / LU)",
            "   apply a precomputed sparse factorization and are division-free",
            "   (diagonals stored as reciprocals); the CG method runs preconditioned",
            "   conjugate gradient (sparse matvecs + a diagonal preconditioner). No",
            "   matrix inverse is ever formed or used. */",
            "",
        ]
        for name, fac in ir.factorizations.items():
            lines.append(f"#define FOM_FAC_{name.upper()}_DIM {fac.dim}")
            lines.append(
                f"void fom_lin_solve_{name}(const fom_float *b, fom_float *x);  "
                f"/* {fac.method.upper()}, dim {fac.dim}, nnz {fac.nnz} */"
            )
            lines.append("")
        lines += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* FOM_LINSOLVE_H */", ""]
        return "\n".join(lines)

    def _linsolve_source(self, ir: FomIR) -> str:
        lines = ['#include "fom_linsolve.h"', ""]
        for name, fac in ir.factorizations.items():
            up = name.upper()
            if fac.method == "cg":
                lines.append(
                    f"/* ---- solve '{name}' (CG, preconditioned conjugate gradient) ---- */"
                )
                rowptr, colidx, vals = csr_arrays(fac.A.to_scipy())
                lines.append(_c_int_array(f"FOM_FAC_{up}_A_ROWPTR", rowptr))
                lines.append(_c_int_array(f"FOM_FAC_{up}_A_COLIDX", colidx))
                lines.append(_c_array(f"FOM_FAC_{up}_A_VALS", vals))
                lines.append(_c_array(f"FOM_FAC_{up}_MINV", fac.minv))
                lines.append("")
                lines.append(self._cg_routine(name, fac))
                lines.append("")
                continue
            lines.append(f"/* ---- factorization '{name}' ({fac.method.upper()}) ---- */")
            lines += _factor_arrays(name, fac)
            lines.append("")
            lines.append(self._solve_routine(name, fac))
            lines.append("")
        return "\n".join(lines)

    def _solve_routine(
        self, name: str, fac, func_name: str | None = None, storage: str = ""
    ) -> str:
        up = name.upper()
        n = f"FOM_FAC_{up}_DIM"
        fname = func_name or f"fom_lin_solve_{name}"
        prefix = f"{storage} " if storage else ""
        out = [
            f"{prefix}void {fname}(const fom_float *b, fom_float *x) {{",
            f"    fom_float y[{n}];",
            "    fom_int i, j, p;",
            f"    for (i = 0; i < {n}; ++i) y[i] = b[FOM_FAC_{up}_PERM[i]];",
            "    /* forward solve: unit lower-triangular L y = P^T b (L stored without",
            "       its unit diagonal) */",
            f"    for (j = 0; j < {n}; ++j) {{",
            "        const fom_float yj = y[j];",
            f"        for (p = FOM_FAC_{up}_L_INDPTR[j]; p < FOM_FAC_{up}_L_INDPTR[j + 1]; ++p)",
            f"            y[FOM_FAC_{up}_L_INDICES[p]] -= FOM_FAC_{up}_L_DATA[p] * yj;",
            "    }",
        ]
        if fac.method == "ldlt":
            out += [
                "    /* diagonal solve (multiply by reciprocals) */",
                f"    for (i = 0; i < {n}; ++i) y[i] *= FOM_FAC_{up}_DINV[i];",
                "    /* backward solve in place: unit upper-triangular L^T y = y */",
                f"    for (j = {n} - 1; j >= 0; --j) {{",
                "        fom_float acc = y[j];",
                f"        for (p = FOM_FAC_{up}_L_INDPTR[j]; p < FOM_FAC_{up}_L_INDPTR[j + 1]; ++p)",
                f"            acc -= FOM_FAC_{up}_L_DATA[p] * y[FOM_FAC_{up}_L_INDICES[p]];",
                "        y[j] = acc;",
                "    }",
                "    /* symmetric scatter: x[perm[i]] = y[i] */",
                f"    for (i = 0; i < {n}; ++i) x[FOM_FAC_{up}_PERM[i]] = y[i];",
                "}",
            ]
        else:
            out += [
                "    /* backward solve: upper-triangular U x = y (U stored without its",
                "       diagonal, which is applied through the reciprocals UINV) */",
                f"    for (j = {n} - 1; j >= 0; --j) {{",
                f"        const fom_float xj = y[j] * FOM_FAC_{up}_UINV[j];",
                "        x[j] = xj;",
                f"        for (p = FOM_FAC_{up}_U_INDPTR[j]; p < FOM_FAC_{up}_U_INDPTR[j + 1]; ++p)",
                f"            y[FOM_FAC_{up}_U_INDICES[p]] -= FOM_FAC_{up}_U_DATA[p] * xj;",
                "    }",
                "}",
            ]
        return "\n".join(out)

    def _cg_routine(self, name: str, fac) -> str:
        """Emit a self-contained Jacobi-preconditioned CG solve.

        Mirrors :func:`fomtools.codegen.factorize._pcg_solve` exactly (same
        iteration, same stopping test ``r·r <= tol^2 (b·b)``). Uses an internal
        sparse (CSR) matrix-vector product against the stored SPD matrix; never
        forms a factorization or inverse. Division is used for the CG scalars,
        which is why CG is reserved for the large-scale (non-embedded) targets.
        """
        up = name.upper()
        n = f"FOM_FAC_{up}_DIM"
        tol2 = _fmt_float(fac.cg_tol * fac.cg_tol)
        maxit = int(fac.cg_max_iter)
        out = [
            f"void fom_lin_solve_{name}(const fom_float *b, fom_float *x) {{",
            f"    fom_float r[{n}], z[{n}], p[{n}], Ap[{n}];",
            "    fom_float bb = 0.0, thresh, rz = 0.0, rz_new, pAp, alpha, beta, s, rr;",
            "    fom_int i, k, it;",
            "    /* x <- 0; bb = b.b; the initial residual r = b - A x = b. Every CG",
            "       kernel below (matvec, dot product, vector update) is data-parallel;",
            "       only the outer iteration is sequential. */",
        ]
        out += self._omp_for("reduction(+ : bb)")
        out += [
            f"    for (i = 0; i < {n}; ++i) {{ x[i] = 0.0; bb += b[i] * b[i]; }}",
            "    if (bb == 0.0) return;",
            f"    thresh = (fom_float){tol2} * bb;",
            "    /* r = b; z = Minv .* r; p = z; rz = r.z */",
        ]
        out += self._omp_for("reduction(+ : rz)")
        out += [
            f"    for (i = 0; i < {n}; ++i) {{",
            "        r[i] = b[i];",
            f"        z[i] = FOM_FAC_{up}_MINV[i] * r[i];",
            "        p[i] = z[i];",
            "        rz += r[i] * z[i];",
            "    }",
            f"    for (it = 0; it < {maxit}; ++it) {{",
            "        /* Ap = A @ p  (sparse CSR matvec) */",
        ]
        out += self._omp_for("private(k, s)", indent="        ")
        out += [
            f"        for (i = 0; i < {n}; ++i) {{",
            "            s = 0.0;",
            f"            for (k = FOM_FAC_{up}_A_ROWPTR[i]; k < FOM_FAC_{up}_A_ROWPTR[i + 1]; ++k)",
            f"                s += FOM_FAC_{up}_A_VALS[k] * p[FOM_FAC_{up}_A_COLIDX[k]];",
            "            Ap[i] = s;",
            "        }",
            "        pAp = 0.0;",
        ]
        out += self._omp_for("reduction(+ : pAp)", indent="        ")
        out += [
            f"        for (i = 0; i < {n}; ++i) pAp += p[i] * Ap[i];",
            "        if (pAp == 0.0) break;",
            "        alpha = rz / pAp;",
            "        rr = 0.0;",
        ]
        out += self._omp_for("reduction(+ : rr)", indent="        ")
        out += [
            f"        for (i = 0; i < {n}; ++i) {{",
            "            x[i] += alpha * p[i];",
            "            r[i] -= alpha * Ap[i];",
            "            rr += r[i] * r[i];",
            "        }",
            "        if (rr <= thresh) break;",
            "        rz_new = 0.0;",
        ]
        out += self._omp_for("reduction(+ : rz_new)", indent="        ")
        out += [
            f"        for (i = 0; i < {n}; ++i) {{",
            f"            z[i] = FOM_FAC_{up}_MINV[i] * r[i];",
            "            rz_new += r[i] * z[i];",
            "        }",
            "        beta = rz_new / rz;",
        ]
        out += self._omp_for(indent="        ")
        out += [
            f"        for (i = 0; i < {n}; ++i) p[i] = z[i] + beta * p[i];",
            "        rz = rz_new;",
            "    }",
            "}",
        ]
        return "\n".join(out)

    # -- verify ------------------------------------------------------------
    def _sample_params(self, ir: FomIR) -> tuple[dict[str, np.ndarray], list[dict]]:
        rng = np.random.default_rng(self.config.seed)
        oracle = ReferenceOracle(ir)
        samples = []
        for _ in range(self.config.num_verify_samples):
            pvals = {}
            for p in ir.params:
                lo = np.where(np.isfinite(p.lb), p.lb, -1.0)
                hi = np.where(np.isfinite(p.ub), p.ub, 1.0)
                pvals[p.name] = rng.uniform(lo, hi)
            res = oracle.solve(param_values=pvals)
            samples.append(
                {
                    "params": pvals,
                    "solution": res.solution,
                    "iters": res.iters,
                    "residual": res.residual,
                    "status": 0 if res.solved else 1,  # FOM_SOLVED / FOM_MAX_ITER_REACHED
                }
            )
        return samples

    #: Right-hand sides per factorization in the direct linear-solve check.
    _LINSOLVE_CHECK_RHS = 3

    def _checked_factorizations(self, ir: FomIR) -> list:
        """Direct factorizations whose C solve ``make verify`` checks on its own.

        Only when the code optimization phase may rewrite the linear solve
        (:meth:`optimization_editable_paths`): the end-to-end check sees a solve
        only through the whole iteration, so a rewrite is also held directly to
        the factorization's reference solve. CG is iterative, so its C and
        reference results differ legitimately; it is checked end to end only.
        """
        if self.linsolve_source_path() not in self.optimization_editable_paths():
            return []
        return [(name, fac) for name, fac in ir.factorizations.items() if fac.method != "cg"]

    def _linsolve_check_data(self, ir: FomIR) -> list[str]:
        lines = []
        rng = np.random.default_rng(self.config.seed + 1)
        k = self._LINSOLVE_CHECK_RHS
        for name, fac in self._checked_factorizations(ir):
            up = name.upper()
            rhs = rng.standard_normal((k, fac.dim))
            sol = np.vstack([fac.solve(b) for b in rhs])
            lines.append(_c_array_2d(f"FOM_VERIFY_LS_{up}_B", rhs, f"[{k}][{fac.dim}]"))
            lines.append(_c_array_2d(f"FOM_VERIFY_LS_{up}_X", sol, f"[{k}][{fac.dim}]"))
            lines.append("")
        return lines

    def _linsolve_check_main(self, ir: FomIR) -> list[str]:
        """Check each ``fom_lin_solve_<name>`` against its reference solve."""
        lines = []
        k = self._LINSOLVE_CHECK_RHS
        for name, fac in self._checked_factorizations(ir):
            up = name.upper()
            lines += [
                "    {",
                f"        /* fom_lin_solve_{name}: direct check against the reference solve. */",
                f"        static fom_float x[{fac.dim}];",
                "        int r;",
                f"        for (r = 0; r < {k}; ++r) {{",
                "            fom_float err = 0.0, scale = 1.0;",
                f"            fom_lin_solve_{name}(FOM_VERIFY_LS_{up}_B[r], x);",
                f"            for (j = 0; j < {fac.dim}; ++j) {{",
                f"                fom_float d = x[j] - FOM_VERIFY_LS_{up}_X[r][j];",
                f"                fom_float a = FOM_VERIFY_LS_{up}_X[r][j];",
                "                if (d < 0) d = -d;",
                "                if (a < 0) a = -a;",
                "                if (d > err) err = d;",
                "                if (1.0 + a > scale) scale = 1.0 + a;",
                "            }",
                "            if (err > tol * scale) {",
                f'                printf("VERIFY FAILED: fom_lin_solve_{name} differs from its "',
                '                       "reference solve by %.3e on right-hand side %d\\n", (double)err, r);',
                "                return 1;",
                "            }",
                "        }",
                "    }",
            ]
        return lines

    def _objective_cases(self, ir: FomIR, samples: list[dict]) -> list[tuple[int, int, list]]:
        """Runtime settings ``make verify`` re-runs an objective-terminated solver at.

        Each case is ``(check_termination, max_iter, oracle results per sample)``:
        the intervals in :data:`_VERIFY_CHECK_INTERVALS`, and a ``max_iter`` that
        stops every sample short of its own exit, so the solver must honor both
        settings at run time and test the objective at the last iteration.
        """
        term = ir.termination
        if not term.uses_objective or ir.unrolled or not samples:
            return []
        cases = [
            (c, term.max_iters) for c in _VERIFY_CHECK_INTERVALS if c != term.check_termination
        ]
        shortest = min(s["iters"] for s in samples)
        if shortest > 1:
            cases.append((term.check_termination, shortest - 1))
        out = []
        for check, max_iter in cases:
            update = {"check_termination": check, "max_iters": max_iter}
            oracle = ReferenceOracle(replace(ir, termination=term.model_copy(update=update)))
            out.append((check, max_iter, [oracle.solve(param_values=s["params"]) for s in samples]))
        return out

    def _verify_data(self, ir: FomIR) -> str:
        samples = self._sample_params(ir)
        n = len(samples)
        lines = [
            "#ifndef FOM_VERIFY_DATA_H",
            "#define FOM_VERIFY_DATA_H",
            "",
            '#include "fom_workspace.h"',
            "",
            f"#define FOM_NUM_SAMPLES {n}",
            "",
        ]
        for p in ir.params:
            rows = [s["params"][p.name] for s in samples]
            mat = np.vstack(rows) if rows else np.zeros((0, p.dim))
            lines.append(
                _c_array_2d(
                    f"FOM_VERIFY_{p.name.upper()}", mat, f"[{n}][FOM_PARAM_{p.name.upper()}_DIM]"
                )
            )
            lines.append("")
        expected = (
            np.vstack([s["solution"] for s in samples]) if samples else np.zeros((0, ir.state_dim))
        )
        lines.append(_c_array_2d("FOM_VERIFY_EXPECTED", expected, f"[{n}][FOM_STATE_DIM]"))
        lines.append("")
        iters = np.array([s["iters"] for s in samples], dtype=int)
        body = ", ".join(str(int(v)) for v in iters)
        lines.append(f"static const fom_int FOM_VERIFY_ITERS[{n}] = {{{body}}};")
        lines += self._verify_objective_data(ir, samples)
        lines.append("")
        lines += self._linsolve_check_data(ir)
        lines += ["#endif /* FOM_VERIFY_DATA_H */", ""]
        return "\n".join(lines)

    def _verify_objective_data(self, ir: FomIR, samples: list[dict]) -> list[str]:
        """Where the oracle stops, and on what objective value, per sample and case."""
        if not ir.termination.uses_objective:
            return ["#define FOM_VERIFY_OBJECTIVE 0", "#define FOM_VERIFY_NUM_CASES 0"]
        n = len(samples)
        lines = [
            "",
            "/* Objective termination: the oracle's stopping iteration, objective value",
            "   at the stop (solution.residual) and status, per sample. */",
            "#define FOM_VERIFY_OBJECTIVE 1",
            _c_array("FOM_VERIFY_RESIDUAL", [s["residual"] for s in samples], ctype="double"),
            _c_int_array("FOM_VERIFY_STATUS", [s["status"] for s in samples]),
        ]
        cases = self._objective_cases(ir, samples)
        lines.append("")
        lines.append("/* The same, re-run at other runtime (check_termination, max_iter). */")
        lines.append(f"#define FOM_VERIFY_NUM_CASES {len(cases)}")
        if cases:
            c = len(cases)
            lines.append(_c_int_array("FOM_VERIFY_CASE_CHECK", [case[0] for case in cases]))
            lines.append(_c_int_array("FOM_VERIFY_CASE_MAXITER", [case[1] for case in cases]))
            table = [[r.iters for r in case[2]] for case in cases]
            lines.append(_c_int_array_2d("FOM_VERIFY_CASE_ITERS", table, f"[{c}][{n}]"))
            table = [[r.residual for r in case[2]] for case in cases]
            lines.append(_c_array_2d("FOM_VERIFY_CASE_RESIDUAL", table, f"[{c}][{n}]", "double"))
            table = [[0 if r.solved else 1 for r in case[2]] for case in cases]
            lines.append(_c_int_array_2d("FOM_VERIFY_CASE_STATUS", table, f"[{c}][{n}]"))
        return lines

    def _verify_update_block(self, ir: FomIR, sample: str, indent: str) -> str:
        """Copy one verify sample's parameter values into ``params``."""
        return "\n".join(
            f"{indent}for (j = 0; j < FOM_PARAM_{p.name.upper()}_DIM; ++j)\n"
            f"{indent}    params.{p.name}[j] = FOM_VERIFY_{p.name.upper()}[{sample}][j];"
            for p in ir.params
        )

    def _verify_single_pass(self, ir: FomIR) -> list[str]:
        """Check that an unrolled body ignores ``settings.max_iter``.

        The body is the whole recorded run, so raising the iteration budget must
        not change the answer. A body that wrapped itself in the usual
        ``for (it = 0; it < work->settings.max_iter; ++it)`` would pass every
        other check -- the default budget is 1 -- and then silently run the
        algorithm several times over for any caller who raised it.
        """
        if not ir.unrolled:
            return []
        return [
            "",
            "    /* FOM_UNROLLED: the body is the whole run, executed once. */",
            "    settings.max_iter = FOM_MAX_ITER + 3;",
            "    fom_setup(&work, &settings);",
            self._verify_update_block(ir, "0", "    "),
            "    fom_update_params(&work, &params);",
            "    fom_solve(&work);",
            "    {",
            "        fom_float err = 0.0;",
            "        for (j = 0; j < FOM_STATE_DIM; ++j) {",
            "            fom_float d = work.solution.x[j] - FOM_VERIFY_EXPECTED[0][j];",
            "            if (d < 0) d = -d;",
            "            if (d > err) err = d;",
            "        }",
            "        if (err > tol) {",
            '            printf("VERIFY FAILED: raising max_iter changed the answer by "',
            '                   "%.3e; an unrolled body must execute exactly once\\n",',
            "                   (double)err);",
            "            return 1;",
            "        }",
            "    }",
        ]

    def _verify_stop_helper(self) -> list[str]:
        """``check_stop``: compare where and how a solve stopped with the oracle.

        The objective value is compared relative to ``max(1, |expected|)`` at the
        verify tolerance; the iteration and the status must match exactly.
        """
        return [
            "static int check_stop(const char *what, int s, fom_int iter, fom_float res,",
            "                      fom_int status, fom_int e_iter, double e_res, fom_int e_status,",
            "                      double tol) {",
            "    double d = (double)res - e_res;",
            "    double scale = e_res < 0.0 ? -e_res : e_res;",
            "    if (d < 0.0) d = -d;",
            "    if (scale < 1.0) scale = 1.0;",
            "    if (iter == e_iter && status == e_status && d <= tol * scale) return 0;",
            '    printf("%s sample %d: stopped at iteration %d with objective %.6e (status %d); "',
            '           "the oracle stops at %d with %.6e (status %d)\\n",',
            "           what, s, (int)iter, (double)res, (int)status, (int)e_iter, e_res,",
            "           (int)e_status);",
            "    return 1;",
            "}",
            "",
        ]

    def _verify_stop_checks(self, ir: FomIR, tol: str) -> tuple[list[str], list[str]]:
        """Objective-termination checks: per sample, then per runtime case.

        Returns the statements for inside the main sample loop and those that run
        after it. Empty unless the loop stops on the verification objective.
        """
        if not ir.termination.uses_objective:
            return [], []
        per_sample = [
            '        stop_failures += check_stop("default settings:", s, work.solution.iter,',
            "                                    work.solution.residual, stop_status,",
            "                                    FOM_VERIFY_ITERS[s], FOM_VERIFY_RESIDUAL[s],",
            f"                                    FOM_VERIFY_STATUS[s], {tol});",
        ]
        after = [
            "",
            "#if FOM_VERIFY_NUM_CASES > 0",
            "    /* The runtime settings must drive the stop: re-run every sample at",
            "       other (check_termination, max_iter) and compare with the oracle. */",
            "    {",
            "        int c;",
            "        char what[64];",
            "        for (c = 0; c < FOM_VERIFY_NUM_CASES; ++c) {",
            "            fom_set_default_settings(&settings);",
            "            settings.check_termination = FOM_VERIFY_CASE_CHECK[c];",
            "            settings.max_iter = FOM_VERIFY_CASE_MAXITER[c];",
            "            fom_setup(&work, &settings);",
            '            snprintf(what, sizeof what, "check_termination=%d max_iter=%d:",',
            "                     (int)FOM_VERIFY_CASE_CHECK[c], (int)FOM_VERIFY_CASE_MAXITER[c]);",
            "            for (s = 0; s < FOM_NUM_SAMPLES; ++s) {",
            self._verify_update_block(ir, "s", "                "),
            "                fom_update_params(&work, &params);",
            "                stop_status = fom_solve(&work);",
            "                stop_failures += check_stop(what, s, work.solution.iter,",
            "                                            work.solution.residual, stop_status,",
            "                                            FOM_VERIFY_CASE_ITERS[c][s],",
            "                                            FOM_VERIFY_CASE_RESIDUAL[c][s],",
            f"                                            FOM_VERIFY_CASE_STATUS[c][s], {tol});",
            "            }",
            "        }",
            "        fom_set_default_settings(&settings);",
            "        fom_setup(&work, &settings);",
            "    }",
            "#endif",
            "    if (stop_failures) {",
            '        printf("VERIFY FAILED: %d solve(s) stopped at a different iteration, "',
            '               "objective value or status than the oracle\\n", stop_failures);',
            "        return 1;",
            "    }",
        ]
        return per_sample, after

    def _verify_main(self, ir: FomIR) -> str:
        tol = _fmt_float(self.config.tol_verify)
        update_block = self._verify_update_block(ir, "s", "        ")
        objective = ir.termination.uses_objective
        per_sample, after = self._verify_stop_checks(ir, f"(double){tol}")
        return "\n".join(
            [
                "#include <stdio.h>",
                "#include <math.h>",
                '#include "fom_solver.h"',
                *(['#include "fom_linsolve.h"'] if self._checked_factorizations(ir) else []),
                '#include "verify_data.h"',
                "",
                *(self._verify_stop_helper() if objective else []),
                "int main(void) {",
                "    FomWorkspace work;",
                "    FomSettings settings;",
                "    FomParams params;",
                "    int s, j;",
                f"    fom_float tol = (fom_float){tol};",
                "    fom_float worst = 0.0;",
                "    int failures = 0;",
                *(["    int stop_failures = 0;", "    fom_int stop_status;"] if objective else []),
                "",
                *self._linsolve_check_main(ir),
                "",
                "    fom_set_default_settings(&settings);",
                "    fom_setup(&work, &settings);",
                "",
                "    for (s = 0; s < FOM_NUM_SAMPLES; ++s) {",
                update_block,
                "        fom_update_params(&work, &params);",
                "        stop_status = fom_solve(&work);"
                if objective
                else "        fom_solve(&work);",
                "        fom_float err = 0.0;",
                "        for (j = 0; j < FOM_STATE_DIM; ++j) {",
                "            fom_float d = work.solution.x[j] - FOM_VERIFY_EXPECTED[s][j];",
                "            if (d < 0) d = -d;",
                "            if (d > err) err = d;",
                "        }",
                "        if (err > worst) worst = err;",
                "        if (err > tol) {",
                '            printf("sample %d: max abs error %.3e > tol %.3e\\n", s, (double)err, (double)tol);',
                "            failures++;",
                "        }",
                *per_sample,
                "    }",
                '    printf("worst max-abs error over %d samples: %.3e (tol %.3e)\\n",',
                "           FOM_NUM_SAMPLES, (double)worst, (double)tol);",
                "    if (failures) {",
                '        printf("VERIFY FAILED: %d/%d samples exceed tol\\n", failures, FOM_NUM_SAMPLES);',
                "        return 1;",
                "    }",
                *after,
                *self._verify_single_pass(ir),
                "",
                '    printf("VERIFY OK\\n");',
                "    return 0;",
                "}",
                "",
            ]
        )

    # -- profiling + benchmark --------------------------------------------
    def _profile_header(self) -> str:
        return "\n".join(
            [
                "#ifndef FOM_PROFILE_H",
                "#define FOM_PROFILE_H",
                "",
                "/* Portable, allocation-free region profiler. Active only when the",
                "   translation unit is compiled with -DFOM_PROFILE; otherwise every",
                "   macro is a no-op, so it never affects the shipped solver. */",
                "",
                "#include <time.h>",
                "",
                "static inline double fom_now_ns(void) {",
                "    struct timespec ts;",
                "    clock_gettime(CLOCK_MONOTONIC, &ts);",
                "    return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;",
                "}",
                "",
                "#ifdef FOM_PROFILE",
                "void fom_prof_add(const char *name, double ns);",
                "void fom_prof_report(void);",
                "void fom_prof_reset(void);",
                "#define FOM_PROF_BEGIN(name) double fom__t_##name = fom_now_ns()",
                "#define FOM_PROF_END(name) fom_prof_add(#name, fom_now_ns() - fom__t_##name)",
                "#else",
                "#define FOM_PROF_BEGIN(name) ((void)0)",
                "#define FOM_PROF_END(name) ((void)0)",
                "static inline void fom_prof_report(void) {}",
                "static inline void fom_prof_reset(void) {}",
                "#endif",
                "",
                "#endif /* FOM_PROFILE_H */",
                "",
            ]
        )

    def _profile_source(self) -> str:
        # Uses division for percentages/units; compiled only for `make profile`,
        # never into the shipped solver, so the division-free invariant holds.
        return "\n".join(
            [
                '#include "fom_profile.h"',
                "",
                "#ifdef FOM_PROFILE",
                "#include <stdio.h>",
                "#include <string.h>",
                "",
                "#define FOM_PROF_MAX 64",
                "static char fom_prof_names[FOM_PROF_MAX][48];",
                "static double fom_prof_totals[FOM_PROF_MAX];",
                "static long fom_prof_counts[FOM_PROF_MAX];",
                "static int fom_prof_n = 0;",
                "",
                "void fom_prof_reset(void) { fom_prof_n = 0; }",
                "",
                "void fom_prof_add(const char *name, double ns) {",
                "    int i;",
                "    for (i = 0; i < fom_prof_n; ++i) {",
                "        if (strcmp(fom_prof_names[i], name) == 0) {",
                "            fom_prof_totals[i] += ns;",
                "            fom_prof_counts[i] += 1;",
                "            return;",
                "        }",
                "    }",
                "    if (fom_prof_n < FOM_PROF_MAX) {",
                "        strncpy(fom_prof_names[fom_prof_n], name, 47);",
                "        fom_prof_names[fom_prof_n][47] = 0;",
                "        fom_prof_totals[fom_prof_n] = ns;",
                "        fom_prof_counts[fom_prof_n] = 1;",
                "        fom_prof_n += 1;",
                "    }",
                "}",
                "",
                "void fom_prof_report(void) {",
                "    int i, j;",
                "    double grand = 0.0;",
                "    for (i = 0; i < fom_prof_n; ++i) grand += fom_prof_totals[i];",
                "    for (i = 0; i < fom_prof_n; ++i) {",
                "        for (j = i + 1; j < fom_prof_n; ++j) {",
                "            if (fom_prof_totals[j] > fom_prof_totals[i]) {",
                "                double td = fom_prof_totals[i];",
                "                long cd = fom_prof_counts[i];",
                "                char nd[48];",
                "                fom_prof_totals[i] = fom_prof_totals[j]; fom_prof_totals[j] = td;",
                "                fom_prof_counts[i] = fom_prof_counts[j]; fom_prof_counts[j] = cd;",
                "                memcpy(nd, fom_prof_names[i], 48);",
                "                memcpy(fom_prof_names[i], fom_prof_names[j], 48);",
                "                memcpy(fom_prof_names[j], nd, 48);",
                "            }",
                "        }",
                "    }",
                "    for (i = 0; i < fom_prof_n; ++i) {",
                "        double pct = (grand > 0.0) ? (100.0 * fom_prof_totals[i] / grand) : 0.0;",
                '        printf("PROF region=%s total_us=%.4f calls=%ld pct=%.2f\\n",',
                "               fom_prof_names[i], fom_prof_totals[i] / 1000.0,",
                "               fom_prof_counts[i], pct);",
                "    }",
                "}",
                "#endif /* FOM_PROFILE */",
                "",
            ]
        )

    def _bench_main(self, ir: FomIR) -> str:
        set_params = []
        for p in ir.params:
            set_params.append(
                f"        for (j = 0; j < FOM_PARAM_{p.name.upper()}_DIM; ++j)\n"
                f"            params.{p.name}[j] = FOM_VERIFY_{p.name.upper()}[s % FOM_NUM_SAMPLES][j];"
            )
        set_block = "\n".join(set_params)
        return "\n".join(
            [
                "#include <stdio.h>",
                "#include <stdlib.h>",
                '#include "fom_solver.h"',
                '#include "fom_profile.h"',
                '#include "verify_data.h"',
                "",
                "/* Benchmark: time full solves (each = the complete iteration loop).",
                "   Usage: bench_bin [num_solves]. Without num_solves it times up to",
                "   FOM_BENCH_MAX_SOLVES solves but stops after FOM_BENCH_BUDGET_NS of timed",
                "   solving (at least FOM_BENCH_MIN_SOLVES), so one bench of a slow solver",
                "   cannot use up a caller's time budget. Prints a parseable BENCH line",
                "   and, when built with -DFOM_PROFILE, a per-region PROF breakdown. */",
                "#define FOM_BENCH_MAX_SOLVES 5000",
                "#define FOM_BENCH_MIN_SOLVES 5",
                "#define FOM_BENCH_BUDGET_NS 2.0e9",
                "",
                "int main(int argc, char **argv) {",
                "    int reps = (argc > 1) ? atoi(argv[1]) : FOM_BENCH_MAX_SOLVES;",
                "    double budget_ns = (argc > 1) ? 0.0 : FOM_BENCH_BUDGET_NS;",
                "    int done = 0;",
                "    FomWorkspace work;",
                "    FomSettings settings;",
                "    FomParams params;",
                "    int s, j;",
                "    double total = 0.0, best = 1.0e30;",
                "",
                "    fom_set_default_settings(&settings);",
                "    fom_setup(&work, &settings);",
                "",
                "    /* warmup */",
                "    for (s = 0; s < FOM_NUM_SAMPLES; ++s) {",
                set_block,
                "        fom_update_params(&work, &params);",
                "        fom_solve(&work);",
                "    }",
                "    fom_prof_reset();",
                "",
                "    for (s = 0; s < reps; ++s) {",
                set_block,
                "        fom_update_params(&work, &params);",
                "        double t0 = fom_now_ns();",
                "        fom_solve(&work);",
                "        double dt = fom_now_ns() - t0;",
                "        total += dt;",
                "        if (dt < best) best = dt;",
                "        ++done;",
                "        if (budget_ns > 0.0 && total >= budget_ns && done >= FOM_BENCH_MIN_SOLVES)",
                "            break;",
                "    }",
                "",
                '    printf("BENCH solves=%d total_us=%.4f mean_us=%.6f min_us=%.6f\\n",',
                "           done, total / 1000.0, (total / 1000.0) / (double)done, best / 1000.0);",
                "    fom_prof_report();",
                "    return 0;",
                "}",
                "",
            ]
        )

    def _objective_guard(self, ir: FomIR | None) -> tuple[str, list[str]]:
        """``(prerequisite, rules)`` of the Makefile check run before verifying an
        objective-terminated solver: it must call ``fom_objective`` and read the
        runtime ``check_termination``. Empty for any other criterion."""
        if ir is None or not ir.termination.uses_objective:
            return "", []
        src = self.solver_source_path()
        return " objective_guard", [
            "# Objective termination: the solver stops on the generated fom_objective,",
            "# at the runtime check_termination interval, and never on its own metric.",
            "objective_guard:",
            f"\t@grep -q 'fom_objective *(' {src} || (echo 'VERIFY FAILED: {src} never "
            "calls fom_objective, the generated verification objective it must stop on'; "
            "exit 1)",
            f"\t@grep -q 'check_termination' {src} || (echo 'VERIFY FAILED: {src} does not "
            "read settings.check_termination, the runtime interval of the objective "
            "check'; exit 1)",
            "",
        ]

    def _makefile(self, ir: FomIR | None = None) -> str:
        guard, guard_rules = self._objective_guard(ir)
        return "\n".join(
            [
                "CC ?= cc",
                "CFLAGS ?= -O2 -Wall -Iinclude",
                "BENCHFLAGS ?= -O3 -Iinclude",
                "SRC = src/fom_solver.c src/fom_data.c src/fom_linsolve.c src/fom_ops.c",
                "",
                "UNAME_S := $(shell uname -s)",
                "ifeq ($(UNAME_S),Darwin)",
                "LIBEXT = dylib",
                "else",
                "LIBEXT = so",
                "endif",
                "",
                ".PHONY: verify bench profile lib release clean objective_guard",
                "",
                f"verify:{guard} $(SRC) test/verify_main.c",
                "\t$(CC) $(CFLAGS) -Itest $(SRC) test/verify_main.c -o verify_bin -lm",
                "\t./verify_bin",
                "",
                "# Build the shared library from the marker-free production source.",
                "release: release/fom_solver.c src/fom_data.c release/fom_linsolve.c src/fom_ops.c",
                "\t$(CC) $(CFLAGS) -shared -fPIC release/fom_solver.c src/fom_data.c "
                "release/fom_linsolve.c src/fom_ops.c -o libfomsolver.$(LIBEXT) -lm",
                "",
                "bench: $(SRC) test/bench_main.c",
                "\t$(CC) $(BENCHFLAGS) -Itest $(SRC) test/bench_main.c -o bench_bin -lm",
                "\t./bench_bin",
                "",
                "profile: $(SRC) src/fom_profile.c test/bench_main.c",
                "\t$(CC) $(BENCHFLAGS) -DFOM_PROFILE -Itest $(SRC) src/fom_profile.c "
                "test/bench_main.c -o profile_bin -lm",
                "\t./profile_bin",
                "",
                "lib: $(SRC)",
                "\t$(CC) $(CFLAGS) -shared -fPIC $(SRC) -o libfomsolver.$(LIBEXT) -lm",
                "",
                "clean:",
                "\trm -f verify_bin bench_bin profile_bin libfomsolver.so libfomsolver.dylib",
                "",
                *guard_rules,
            ]
        )

    def _readme(self, ir: FomIR) -> str:
        return "\n".join(
            [
                f"# Generated solver: `{ir.name}` (embedded C)",
                "",
                "OSQP-codegen-style embedded solver for a verified first-order method.",
                "No dynamic memory allocation; division-free.",
                "",
                "## Build, verify, benchmark, profile",
                "```",
                "make verify    # build and check against the reference oracle",
                "make bench     # time full solves (prints a BENCH line)",
                "make profile   # per-region timing breakdown (prints PROF lines)",
                "make lib       # shared library for the Python wrapper",
                "```",
                "Wrap hot regions of `fom_solve` with `FOM_PROF_BEGIN(name)` /",
                "`FOM_PROF_END(name)` (from `fom_profile.h`) to get a profiling",
                "breakdown; the macros are no-ops in the normal (verify/lib) build.",
                "",
                "## ABI",
                "See `include/fom_solver.h`. Typical usage:",
                "```c",
                "FomWorkspace work; FomSettings s;",
                "fom_set_default_settings(&s);",
                "fom_setup(&work, &s);",
                "fom_update_params(&work, &params);",
                "fom_solve(&work);",
                "/* work.solution.x holds the solution */",
                "```",
                "",
                (
                    "`src/fom_solver.c` implements the recorded run as a single pass"
                    if ir.unrolled
                    else "`src/fom_solver.c` implements the iteration"
                )
                + "; all numeric data lives in",
                "`src/fom_data.c`. The algorithm is specified in `SPEC.md`.",
                "",
            ]
            + (
                [
                    "Each iteration of this method differs, so the body is unrolled:",
                    f"`fom_solve` runs it once ({len(ir.body)} steps) and",
                    "`settings.max_iter` does not apply.",
                    "",
                ]
                if ir.unrolled
                else []
            )
        )

    # -- static checks -----------------------------------------------------
    def static_checks(self, source: str, ir: FomIR | None = None) -> list[str]:
        violations = objective_violations(source, ir)
        code = strip_c_comments_and_strings(source)
        for fn in ("malloc", "calloc", "realloc", "free", "alloca"):
            if re.search(rf"\b{fn}\s*\(", code):
                violations.append(f"dynamic memory call '{fn}' is forbidden in embedded C")
        # division-free: the folded-inverse lowering removes all division
        if re.search(r"/(?!/)", code):  # any '/' not starting a // (already stripped)
            if "/" in code:
                violations.append(
                    "division operator '/' found; embedded target must be division-free"
                )
        # Schedule tables must be read through the generated accessor, which
        # clamps the index. The loop bound is the runtime settings.max_iter, so
        # indexing the table directly can read past its end.
        for match in re.finditer(r"\bFOM_SCHED_(\w+?)\s*\[", code):
            violations.append(
                f"schedule table 'FOM_SCHED_{match.group(1)}' is indexed directly; "
                f"use the generated accessor fom_sched_{match.group(1).lower()}(it), "
                "which clamps the index"
            )
        # crude VLA guard: array decl with a non-numeric, non-macro size token
        return violations
