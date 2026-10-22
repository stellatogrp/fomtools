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
  output root), and the Python wrapper (via `fomtools.codegen.wrapper`).

The coding agent (or a golden fixture) supplies ``src/fom_solver.c``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as spa

from fomtools.codegen import kernel_rules, spec
from fomtools.codegen.ir import (
    ActivationKind,
    FomIR,
    csr_arrays,
    selection_description,
    selection_indices,
)
from fomtools.codegen.kernel_rules import strip_c_comments_and_strings
from fomtools.codegen.objective import objective_c_params, objective_c_source
from fomtools.codegen.oracle import ReferenceOracle
from fomtools.codegen.targets.base import TargetDescriptor, objective_violations


def write_if_changed(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` unless it already holds exactly that.

    ``scaffold`` runs before every verify of an agent's rewrite, restoring the
    generated files; leaving an unchanged file alone keeps its modification time,
    so ``make`` does not rebuild objects that depend on it (a large problem's
    ``fom_data.c`` takes most of a minute to compile).
    """
    path = Path(path)
    if path.exists() and path.stat().st_size == len(text.encode()) and path.read_text() == text:
        return
    path.write_text(text)


def _fmt_float(x: float) -> str:
    return f"{float(x):.17g}"


#: Written while an agent works (translation or code optimization, removed
#: after), so that ``make verify`` also applies the kernel rules to its code.
_KERNEL_CHECK_CONFIG = "test/kernel_check.json"


#: ``make verify`` step that applies the kernel rules while a config is present
#: (only while an agent works, and only if python3 is available).
_KERNEL_CHECK_RECIPE = (
    f"\t@if [ -f {_KERNEL_CHECK_CONFIG} ] && command -v python3 >/dev/null; then "
    f"python3 test/kernel_check.py {_KERNEL_CHECK_CONFIG}; fi"
)


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


@dataclass(frozen=True)
class FusedBlock:
    """One output block's products, applied by a single ``fom_fused_<name>``.

    The terms of one linear op whose source changes every iteration (the state
    and the temporaries; parameter terms stay separate so the solver can hoist
    them out of the loop), as ``out += sum_k scale_k * (M_k @ src_k)``. A rewrite
    of the fused routine may exploit relations between its members (one a
    rearrangement of another, a shared block) that no single ``fom_apply_*``
    can, and ``make verify`` recovers every member from it exactly.
    """

    name: str  # the body step's name, or "su" for the state update
    terms: tuple  # the member Terms, in linear-op order
    rows: int

    @property
    def sources(self) -> tuple[str, ...]:
        """Distinct IR sources (``state``, ``tmp:<name>``), in term order."""
        return tuple(dict.fromkeys(t.source for t in self.terms))

    @property
    def scales(self) -> tuple[str, ...]:
        """Distinct schedules scaling a member, in term order."""
        return tuple(dict.fromkeys(t.scale for t in self.terms if t.scale is not None))

    @property
    def members(self) -> list[str]:
        return [t.matrix for t in self.terms]


def _fused_src(source: str) -> str:
    """The C parameter name of a fused routine's input vector."""
    return "state" if source == "state" else "t_" + source.split(":", 1)[1]


def _fused_scale(sched: str) -> str:
    """The C parameter name of a fused routine's schedule value."""
    return f"sc_{sched}"


def _fused_signature(ir: FomIR, block: FusedBlock, fname: str) -> str:
    args = [f"const fom_float *{_fused_src(s)}" for s in block.sources]
    args += [f"fom_float {_fused_scale(c)}" for c in block.scales]
    return f"void {fname}({', '.join(args + ['fom_float *out'])})"


def _fused_formula(block: FusedBlock) -> str:
    parts = []
    for t in block.terms:
        prod = f"{t.matrix} @ {_fused_src(t.source)}"
        parts.append(f"{_fused_scale(t.scale)} * ({prod})" if t.scale else prod)
    return "out += " + " + ".join(parts)


def _source_dim(ir: FomIR, block: FusedBlock, source: str) -> int:
    md = ir.matrices[next(t.matrix for t in block.terms if t.source == source)]
    return md.shape[1] if len(md.shape) > 1 else 1


def _fused_blocks(ir: FomIR) -> list[FusedBlock]:
    """The output blocks with two or more products by per-iteration sources."""
    ops = [(s.name, s.linear_op) for s in ir.body] + [("su", ir.state_update)]
    blocks = []
    for name, lin in ops:
        terms = tuple(t for _, t in _live_terms(ir, lin) if not t.source.startswith("param:"))
        if len(terms) >= 2:
            blocks.append(FusedBlock(name=name, terms=terms, rows=lin.out_dim))
    return blocks


def _selections(ir: FomIR) -> dict[str, np.ndarray]:
    """Matvec matrices that are 0/1 selections, with each row's source index."""
    out = {}
    for name in _matvec_matrix_names(ir):
        src = selection_indices(ir.matrices[name])
        if src is not None:
            out[name] = src
    return out


@dataclass(frozen=True)
class StepBlock(FusedBlock):
    """One output block computed whole by a ``fom_step_<name>``.

    ``out = act(bias + sum_k scale_k * (M_k @ src_k))``, overwriting ``out``: the
    per-iteration products of a body step (or of the state update, ``su``), with
    the caller's ``bias`` (the constant and the parameter terms, which the solver
    can still hoist) and the step's activation folded in. The activation's
    parameters come in through a ``FomAct_<name>`` the caller passes (the package
    default is ``FOM_ACT_<name>``), so ``make verify`` can switch the activation
    off and recover every member exactly, then check it on and at random bounds.
    """

    act: object = None  # the body step's ActivationOp; None for identity and su

    @property
    def activated(self) -> bool:
        return self.act is not None


def _step_blocks(ir: FomIR) -> list[StepBlock]:
    """A `StepBlock` per body step without a linear solve whose linear op has a
    product by a per-iteration source, and one for the state update. None for an
    unrolled IR, whose body is the whole run."""
    if getattr(ir, "unrolled", False):
        return []
    ops = [(s.name, s.linear_op, s.activation, s.solve) for s in ir.body]
    ops.append(("su", ir.state_update, None, None))
    blocks = []
    for name, lin, act, solve in ops:
        if solve is not None:
            continue
        terms = tuple(t for _, t in _live_terms(ir, lin) if not t.source.startswith("param:"))
        if not terms:
            continue
        if act is not None and act.kind == ActivationKind.IDENTITY:
            act = None
        blocks.append(StepBlock(name=name, terms=terms, rows=lin.out_dim, act=act))
    return blocks


def _iterate_blocks(ir: FomIR, steps: list[StepBlock]) -> list[StepBlock]:
    """The step blocks ``fom_iterate`` chains (every body step, then ``su``), or
    ``[]`` when some body step has no step routine (a linear solve, say)."""
    names = [s.name for s in ir.body] + ["su"]
    by_name = {b.name: b for b in steps}
    return [by_name[n] for n in names] if all(n in by_name for n in names) else []


def _act_type(block: StepBlock) -> str:
    return f"FomAct_{block.name}"


def _act_typedef(block: StepBlock) -> list[str]:
    """The ``FomAct_<block>`` struct holding the activation's parameters."""
    kind = block.act.kind
    if kind == ActivationKind.SOFT_THRESHOLD:
        fields, what = ["fom_float lambd;"], "soft threshold: v -> sign(v) max(|v| - lambd, 0)"
    elif kind == ActivationKind.RELU:
        fields, what = ["const fom_float *lo;"], "relu: v -> max(v, lo[i]) (lo = 0 by default)"
    elif kind == ActivationKind.SATURATED_LINEAR:
        fields = ["const fom_float *lo;", "const fom_float *hi;"]
        what = "saturated linear: v -> min(max(v, lo[i]), hi[i])"
    else:
        raise ValueError(f"no step activation for {kind}")
    return [f"/* {what} */", f"typedef struct {{ {' '.join(fields)} }} {_act_type(block)};"]


def _act_apply(block: StepBlock, v: str, i: str, act: str, indent: str) -> list[str]:
    """C statements applying ``block``'s activation, with parameters ``act->``, to
    the variable ``v`` of row ``i`` (the oracle's order of operations)."""
    if block.act is None:
        return []
    kind = block.act.kind
    if kind == ActivationKind.SOFT_THRESHOLD:
        return [
            f"{indent}{v} = ({v} > {act}->lambd) ? ({v} - {act}->lambd)"
            f" : (({v} < -{act}->lambd) ? ({v} + {act}->lambd) : 0.0);"
        ]
    lines = [f"{indent}if ({v} < {act}->lo[{i}]) {v} = {act}->lo[{i}];"]
    if kind == ActivationKind.SATURATED_LINEAR:
        lines.append(f"{indent}if ({v} > {act}->hi[{i}]) {v} = {act}->hi[{i}];")
    return lines


def _selection_rows(ir: FomIR, block: FusedBlock) -> list[tuple[int, int, int]]:
    """``(row, source index, column)`` of the rows of ``block`` that copy one
    source entry: exactly one unscaled member has exactly one entry there, equal
    to exactly 1.0, and every other member's row is empty."""
    rows = block.rows
    count = np.zeros(rows, dtype=int)
    single: dict[int, tuple[int, int, float, object]] = {}
    for t in block.terms:
        rowptr, colidx, vals = ir.matrices[t.matrix].csr()
        lengths = np.diff(np.asarray(rowptr))
        count += lengths
        si = block.sources.index(t.source)
        for i in np.flatnonzero(lengths == 1):
            single[int(i)] = (si, int(colidx[rowptr[i]]), float(vals[rowptr[i]]), t.scale)
    out = []
    for i in np.flatnonzero(count == 1):
        si, col, val, scale = single[int(i)]
        if val == 1.0 and scale is None:
            out.append((int(i), si, col))
    return out


def _selection_rows_description(block: FusedBlock, rows: list[tuple[int, int, int]]) -> str:
    """Runs of copied rows, e.g. ``rows 0..48 copy t_t0[0..48]``."""
    runs: list[list[int]] = []
    for i, si, col in rows:
        if runs and runs[-1][1] == i - 1 and runs[-1][2] == si and runs[-1][4] == col - 1:
            runs[-1][1], runs[-1][4] = i, col
        else:
            runs.append([i, i, si, col, col])
    parts = []
    for a, b, si, c, d in runs:
        src = _fused_src(block.sources[si])
        parts.append(
            f"row {a} copies {src}[{c}]" if a == b else f"rows {a}..{b} copy {src}[{c}..{d}]"
        )
    return "; ".join(parts)


def _step_signature(block: StepBlock, fname: str) -> str:
    args = [f"const fom_float *{_fused_src(s)}" for s in block.sources]
    args += [f"fom_float {_fused_scale(c)}" for c in block.scales]
    args.append("const fom_float *bias")
    if block.activated:
        args.append(f"const {_act_type(block)} *act")
    return f"void {fname}({', '.join(args + ['fom_float *out'])})"


def _step_formula(block: StepBlock) -> str:
    inner = _fused_formula(block)[len("out += ") :]
    act = block.act.kind.value if block.activated else ""
    return f"out = {act}(bias + {inner})" if act else f"out = bias + {inner}"


def _iterate_scales(blocks: list[StepBlock]) -> list[str]:
    return list(dict.fromkeys(c for b in blocks for c in b.scales))


def _iterate_signature(blocks: list[StepBlock], fname: str) -> str:
    args = ["const fom_float *state"]
    args += [f"fom_float {_fused_scale(c)}" for c in _iterate_scales(blocks)]
    args += [f"const fom_float *bias_{b.name}" for b in blocks]
    args += [f"const {_act_type(b)} *act_{b.name}" for b in blocks if b.activated]
    args += [f"fom_float *t_{b.name}" for b in blocks if b.name != "su"]
    return f"void {fname}({', '.join(args + ['fom_float *new_state'])})"


def _step_call(block: StepBlock, fname: str, out: str) -> str:
    """A call of ``fname`` (a step routine of ``block``) inside ``fom_iterate``."""
    args = [_fused_src(s) for s in block.sources]
    args += [_fused_scale(c) for c in block.scales]
    args.append(f"bias_{block.name}")
    if block.activated:
        args.append(f"act_{block.name}")
    return f"    {fname}({', '.join(args + [out])});"


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

    #: Emit a ``fom_fused_<block>`` per output block (`FusedBlock`) and the
    #: selection index data (``FOM_SEL_<name>_SRC``) for the code optimization
    #: phase. Off on CUDA, whose products live on the device in its own sources.
    fused_kernels = True

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

    def objective_source_path(self) -> str:
        """``fom_objective`` gets its own source: ``fom_ops.c`` is editable, the
        metric the solver stops on is not."""
        return "src/fom_objective.c"

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
        write_if_changed(out / "SPEC.md", spec.render_markdown(ir))

        write_if_changed(out / "include" / "fom_workspace.h", self._workspace_header(ir))
        write_if_changed(out / "include" / "fom_solver.h", self._solver_header())
        write_if_changed(out / "include" / "fom_data.h", self._data_header(ir))
        write_if_changed(out / "include" / "fom_profile.h", self._profile_header())
        write_if_changed(out / "include" / "fom_linsolve.h", self._linsolve_header(ir))
        (out / self.linsolve_source_path()).write_text(self._linsolve_source(ir))
        write_if_changed(out / "include" / "fom_ops.h", self._ops_header(ir))
        (out / self.ops_source_path()).write_text(self._ops_source(ir))
        if self.objective_source_path() != self.ops_source_path():
            (out / self.objective_source_path()).write_text(self._objective_file(ir))
        write_if_changed(out / "src" / "fom_data.c", self._data_source(ir))
        write_if_changed(out / "src" / "fom_profile.c", self._profile_source())
        write_if_changed(out / "test" / "kernel_check.py", Path(kernel_rules.__file__).read_text())
        if self._checked_ops(ir):
            write_if_changed(out / "test" / "ops_ref.h", self._ops_ref_header(ir))
            write_if_changed(out / "test" / "ops_ref.c", self._ops_ref_source(ir))
        else:
            for rel in ("test/ops_ref.h", "test/ops_ref.c"):
                (out / rel).unlink(missing_ok=True)
        write_if_changed(out / "test" / "verify_data.h", self._verify_data(ir))
        write_if_changed(out / "test" / "verify_main.c", self._verify_main(ir))
        write_if_changed(out / "test" / "bench_main.c", self._bench_main(ir))
        write_if_changed(out / "Makefile", self._makefile(ir))
        write_if_changed(out / "README.md", self._readme(ir))

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
        objective_only = _objective_only_names(ir)  # emitted with fom_objective
        for name, md in ir.matrices.items():
            if name in objective_only:
                continue
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

        selections = self._selection_data(ir)
        if selections:
            lines.append("/* 0/1 selection matrices: (M @ x)[i] = x[FOM_SEL_<name>_SRC[i]], or 0")
            lines.append(
                "   where the index is -1. A solver may apply these itself (see SPEC.md). */"
            )
        for name, src in selections.items():
            lines.append(f"#define FOM_SEL_{name}_ROWS {src.size}")
            lines.append(_c_extern(f"FOM_SEL_{name}_SRC", "fom_int"))
            lines.append("")

        lines += self._step_data_header(ir)

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
        objective_only = _objective_only_names(ir)  # emitted with fom_objective
        lines = ['#include "fom_data.h"', ""]
        lines.append(_c_array("FOM_STATE_INIT", ir.state.init, storage="const"))
        lines.append("")
        for sched in sorted(ir.schedules):
            lines.append(_c_array(f"FOM_SCHED_{sched}", ir.schedules[sched], storage="const"))
            lines.append("")
        for name, md in ir.matrices.items():
            if name in objective_only:
                continue
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
        for name, src in self._selection_data(ir).items():
            lines.append(_c_int_array(f"FOM_SEL_{name}_SRC", src, storage="const"))
            lines.append("")
        steps = self._step_data_source(ir)
        if any(b.activated for b in self._steps(ir)):
            lines.insert(1, '#include "fom_ops.h"  /* the FomAct_<block> types */')
        lines += steps
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
        blocks = self._fused(ir)
        if blocks:
            lines += [
                "",
                "/* Fused products: every per-iteration product of one output block in a",
                "   single routine (parameter terms are left out, so they can be hoisted).",
                "   Same accumulate convention; each is checked against its members. */",
            ]
        for block in blocks:
            lines.append(f"/* {_fused_formula(block)} */")
            lines.append(_fused_signature(ir, block, f"fom_fused_{block.name}") + ";")
        lines += self._step_ops_header(ir)
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
            lines += self._apply_routine(name, ir.matrices[name], f"fom_apply_{name}")
        for block in self._fused(ir):
            lines += self._fused_routine(ir, block)
        lines += self._step_ops_source(ir)
        return "\n".join(lines)

    def _fused(self, ir: FomIR) -> list[FusedBlock]:
        """The fused routines this target emits and checks: only where the
        per-matrix routines are checked (`_checked_ops`), whose references the
        fused check composes."""
        return _fused_blocks(ir) if self.fused_kernels and self._checked_ops(ir) else []

    def _selection_data(self, ir: FomIR) -> dict[str, np.ndarray]:
        """Selections whose index data ``fom_data.c`` carries (`_selections`)."""
        return _selections(ir) if self.fused_kernels else {}

    def _steps(self, ir: FomIR) -> list[StepBlock]:
        """The step routines this target emits and checks (as `_fused`)."""
        return _step_blocks(ir) if self.fused_kernels and self._checked_ops(ir) else []

    def _iterate(self, ir: FomIR) -> list[StepBlock]:
        """The step blocks ``fom_iterate`` chains, or ``[]`` if it is not emitted."""
        return _iterate_blocks(ir, self._steps(ir))

    def _selrow_blocks(self, ir: FomIR) -> dict[str, tuple[FusedBlock, list]]:
        """Fused and step blocks with selection rows (`_selection_rows`), by name."""
        out = {}
        for block in [*self._fused(ir), *self._steps(ir)]:
            if block.name not in out:
                rows = _selection_rows(ir, block)
                if rows:
                    out[block.name] = (block, rows)
        return out

    def _rests(self, ir: FomIR) -> list[tuple[str, FusedBlock, set[int]]]:
        """``(routine, block, selection rows)`` of each ``fom_fused_<block>__rest`` /
        ``fom_step_<block>__rest``: the block on every row but its selection rows,
        emitted where it has some of each."""
        sel = self._selrow_blocks(ir)
        out = []
        for kind, blocks in (("fused", self._fused(ir)), ("step", self._steps(ir))):
            for b in blocks:
                if b.name in sel and len(sel[b.name][1]) < b.rows:
                    out.append((f"fom_{kind}_{b.name}__rest", b, {r for r, _, _ in sel[b.name][1]}))
        return out

    def _member_row_sums(
        self, ir: FomIR, block: FusedBlock, acc: str
    ) -> tuple[list[str], list[str]]:
        """Row ``i`` of each member's product, added to ``acc`` in linear-op order
        (scaled ones times their schedule value), and the loop indices it uses."""
        body, need = [], set()
        for t in block.terms:
            md = ir.matrices[t.matrix]
            src = _fused_src(t.source)
            mat = f"FOM_MAT_{t.matrix}"
            body.append("        s = 0.0;")
            if _is_csr(md):
                need.add("p")
                body.append(
                    f"        for (p = {mat}_ROWPTR[i]; p < {mat}_ROWPTR[i + 1]; ++p) "
                    f"s += {mat}_VALS[p] * {src}[{mat}_COLIDX[p]];"
                )
            else:
                need.add("j")
                cols = md.shape[1] if len(md.shape) > 1 else 1
                body.append(
                    f"        for (j = 0; j < {cols}; ++j) s += {mat}[i * {cols} + j] * {src}[j];"
                )
            body.append(
                f"        {acc} += {_fused_scale(t.scale)} * s;"
                if t.scale
                else f"        {acc} += s;"
            )
        return body, [v for v in ("p", "j") if v in need]

    def _block_routine(
        self, ir: FomIR, block: FusedBlock, signature: str, step: bool, skip: set[int] = frozenset()
    ) -> list[str]:
        """A generated fused (``out +=``) or step (``out = act(bias + ...)``)
        routine, on every row of ``block`` but those in ``skip`` (left untouched)."""
        body, private = self._member_row_sums(ir, block, "v" if step else "out[i]")
        lines = [signature + " {", f"    fom_int {', '.join(['i', 'r'] + private)};"]
        lines += self._omp_for(f"private({', '.join(['i'] + private)})")
        count = f"FOM_REST_{block.name}_N" if skip else str(block.rows)
        lines += [
            f"    for (r = 0; r < {count}; ++r) {{",
            "        fom_float s" + (", v;" if step else ";"),
            f"        i = FOM_REST_{block.name}_ROWS[r];" if skip else "        i = r;",
        ]
        if step:
            lines.append("        v = bias[i];")
        lines += body
        if step:
            lines += _act_apply(block, "v", "i", "act", "        ")
            lines.append("        out[i] = v;")
        lines += ["    }", "}", ""]
        return lines

    def _step_ops_header(self, ir: FomIR) -> list[str]:
        """Prototypes of the step, rest and iterate routines, for ``fom_ops.h``."""
        steps, lines = self._steps(ir), []
        if steps:
            lines += [
                "",
                "/* Step routines: a whole output block, out = act(bias + sum_k s_k M_k src_k),",
                "   OVERWRITING out. `bias` is the constant plus the parameter terms (the",
                "   caller computes it, so it can be hoisted); `act` carries the activation's",
                "   parameters (pass &FOM_ACT_<block>). Checked against the members and a",
                "   reference activation, with the activation off, on, and at random bounds. */",
            ]
        for b in steps:
            if b.activated:
                lines += _act_typedef(b)
                lines.append(f"extern const {_act_type(b)} FOM_ACT_{b.name};")
            lines.append(f"/* {_step_formula(b)} */")
            lines.append(_step_signature(b, f"fom_step_{b.name}") + ";")
        rests = self._rests(ir)
        if rests:
            lines += [
                "",
                "/* The same, on every row but the block's selection rows (FOM_SELROWS_<block>",
                "   in fom_data.h), which it leaves untouched: the solver writes those copies. */",
            ]
        for fn, b, _ in rests:
            sig = (
                _step_signature(b, fn)
                if fn.startswith("fom_step_")
                else _fused_signature(ir, b, fn)
            )
            lines.append(sig + ";")
        blocks = self._iterate(ir)
        if blocks:
            lines += [
                "",
                "/* One whole iteration: every body step in order (temporaries written to",
                "   t_<name>), then the state update into new_state. Checked against the",
                "   chained reference steps, temporaries included. */",
                _iterate_signature(blocks, "fom_iterate") + ";",
            ]
        return lines

    def _step_ops_source(self, ir: FomIR) -> list[str]:
        """The generated step, rest and iterate routines, for ``fom_ops.c``."""
        lines = []
        for b in self._steps(ir):
            lines += self._block_routine(ir, b, _step_signature(b, f"fom_step_{b.name}"), step=True)
        for fn, b, skip in self._rests(ir):
            step = fn.startswith("fom_step_")
            sig = _step_signature(b, fn) if step else _fused_signature(ir, b, fn)
            lines += self._block_routine(ir, b, sig, step=step, skip=skip)
        blocks = self._iterate(ir)
        if blocks:
            lines.append(_iterate_signature(blocks, "fom_iterate") + " {")
            for b in blocks:
                out = "new_state" if b.name == "su" else f"t_{b.name}"
                lines.append(_step_call(b, f"fom_step_{b.name}", out))
            lines += ["}", ""]
        return lines

    def _step_ops_ref(self, ir: FomIR) -> tuple[list[str], list[str]]:
        """Reference step and iterate routines (prototypes, definitions) for
        ``test/ops_ref``: the members' references, then the activation."""
        protos, lines = [], []
        for b in self._steps(ir):
            sig = _step_signature(b, f"fom_ref_step_{b.name}")
            protos.append(sig + ";")
            lines += [
                sig + " {",
                f"    static fom_float pre[{b.rows}], tmp[{b.rows}];",
                "    fom_int i;",
                f"    for (i = 0; i < {b.rows}; ++i) pre[i] = bias[i];",
            ]
            for t in b.terms:
                scale = f"{_fused_scale(t.scale)} * " if t.scale else ""
                lines += [
                    f"    for (i = 0; i < {b.rows}; ++i) tmp[i] = 0.0;",
                    f"    fom_ref_apply_{t.matrix}({_fused_src(t.source)}, tmp);",
                    f"    for (i = 0; i < {b.rows}; ++i) pre[i] += {scale}tmp[i];",
                ]
            lines += [f"    for (i = 0; i < {b.rows}; ++i) {{", "        fom_float v = pre[i];"]
            lines += _act_apply(b, "v", "i", "act", "        ")
            lines += ["        out[i] = v;", "    }", "}", ""]
        blocks = self._iterate(ir)
        if blocks:
            sig = _iterate_signature(blocks, "fom_ref_iterate")
            protos.append(sig + ";")
            lines.append(sig + " {")
            for b in blocks:
                out = "new_state" if b.name == "su" else f"t_{b.name}"
                lines.append(_step_call(b, f"fom_ref_step_{b.name}", out))
            lines += ["}", ""]
        return protos, lines

    def _step_data_header(self, ir: FomIR) -> list[str]:
        """``FOM_SELROWS_<block>_*`` declarations for ``fom_data.h``."""
        lines = []
        sel = self._selrow_blocks(ir)
        if sel:
            lines += [
                "/* Selection rows: row FOM_SELROWS_<b>_ROW[k] of block <b> is exactly",
                "   source FOM_SELROWS_<b>_SRC[k] (0-based, in the order of the block's",
                "   fom_fused/fom_step arguments) at index FOM_SELROWS_<b>_COL[k]. A solver",
                "   may write these rows itself (see SPEC.md and the __rest routines). */",
            ]
        for name, (block, rows) in sel.items():
            lines.append(f"/* {name}: {_selection_rows_description(block, rows)} */")
            lines.append(f"#define FOM_SELROWS_{name}_N {len(rows)}")
            for part in ("ROW", "SRC", "COL"):
                lines.append(_c_extern(f"FOM_SELROWS_{name}_{part}", "fom_int"))
            lines.append("")
        for name, block, skip in self._rest_blocks(ir):
            lines.append(
                f"/* {name}: the rows a __rest routine computes (all but the selection rows). */"
            )
            lines.append(f"#define FOM_REST_{name}_N {block.rows - len(skip)}")
            lines.append(_c_extern(f"FOM_REST_{name}_ROWS", "fom_int"))
            lines.append("")
        return lines

    def _rest_blocks(self, ir: FomIR) -> list[tuple[str, FusedBlock, set[int]]]:
        """The blocks with a ``__rest`` routine, once each."""
        seen = {}
        for _, block, skip in self._rests(ir):
            seen.setdefault(block.name, (block.name, block, skip))
        return list(seen.values())

    def _step_data_source(self, ir: FomIR) -> list[str]:
        """Selection-row arrays and the default ``FOM_ACT_<block>`` instances."""
        lines = []
        for name, block, skip in self._rest_blocks(ir):
            rows = [i for i in range(block.rows) if i not in skip]
            lines.append(_c_int_array(f"FOM_REST_{name}_ROWS", rows, "const"))
            lines.append("")
        for name, (_, rows) in self._selrow_blocks(ir).items():
            for k, part in enumerate(("ROW", "SRC", "COL")):
                lines.append(
                    _c_int_array(f"FOM_SELROWS_{name}_{part}", [r[k] for r in rows], "const")
                )
            lines.append("")
        for b in self._steps(ir):
            if not b.activated:
                continue
            act, n = b.act, b.rows
            if act.kind == ActivationKind.SOFT_THRESHOLD:
                init = f"(fom_float){_fmt_float(act.lambd)}"
            else:
                bounds = []
                for side, vec, scalar in (
                    ("LO", act.lower, act.lower_scalar if act.kind != ActivationKind.RELU else 0.0),
                    ("HI", act.upper, act.upper_scalar),
                ):
                    if side == "HI" and act.kind == ActivationKind.RELU:
                        continue
                    if vec is not None and act.kind == ActivationKind.SATURATED_LINEAR:
                        bounds.append(f"FOM_MAT_{vec}")
                    else:
                        lines.append(
                            _c_array(f"FOM_ACT_{b.name}_{side}", np.full(n, float(scalar)))
                        )
                        bounds.append(f"FOM_ACT_{b.name}_{side}")
                init = ", ".join(bounds)
            lines.append(f"const {_act_type(b)} FOM_ACT_{b.name} = {{{init}}};")
            lines.append("")
        return lines

    def _fused_routine(self, ir: FomIR, block: FusedBlock) -> list[str]:
        """The generated ``fom_fused_<block>``: each member's row product, in
        linear-op order, added in one pass over the output rows (in the same
        order as the reference, so the two agree to the bit)."""
        body, need_p, need_j = [], False, False
        for t in block.terms:
            md = ir.matrices[t.matrix]
            src = _fused_src(t.source)
            mat = f"FOM_MAT_{t.matrix}"
            body.append("        s = 0.0;")
            if _is_csr(md):
                need_p = True
                body.append(
                    f"        for (p = {mat}_ROWPTR[i]; p < {mat}_ROWPTR[i + 1]; ++p) "
                    f"s += {mat}_VALS[p] * {src}[{mat}_COLIDX[p]];"
                )
            else:
                need_j = True
                cols = md.shape[1] if len(md.shape) > 1 else 1
                body.append(
                    f"        for (j = 0; j < {cols}; ++j) s += {mat}[i * {cols} + j] * {src}[j];"
                )
            body.append(
                f"        out[i] += {_fused_scale(t.scale)} * s;"
                if t.scale
                else "        out[i] += s;"
            )
        private = [v for v, need in (("p", need_p), ("j", need_j)) if need]
        lines = [
            _fused_signature(ir, block, f"fom_fused_{block.name}") + " {",
            f"    fom_int {', '.join(['i'] + private)};",
        ]
        lines += self._omp_for(f"private({', '.join(private)})")
        lines += [
            f"    for (i = 0; i < {block.rows}; ++i) {{",
            "        fom_float s;",
            *body,
            "    }",
            "}",
            "",
        ]
        return lines

    def _apply_routine(self, name: str, md, fname: str) -> list[str]:
        """``void <fname>(x, out)``: ``out += M @ x``, reading ``M`` from ``fom_data.c``
        (CSR if structurally sparse, dense otherwise)."""
        shape = md.shape
        rows = shape[0]
        cols = shape[1] if len(shape) > 1 else 1
        mat = f"FOM_MAT_{name}"
        if _is_csr(md):
            lines = [
                f"void {fname}(const fom_float *x, fom_float *out) {{",
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
            lines = [
                f"void {fname}(const fom_float *x, fom_float *out) {{",
                "    fom_int i, j;",
            ]
            lines += self._omp_for("private(j)")
            lines += [
                f"    for (i = 0; i < {rows}; ++i) {{",
                "        fom_float s = 0.0;",
                f"        for (j = 0; j < {cols}; ++j) s += {mat}[i * {cols} + j] * x[j];",
                "        out[i] += s;",
                "    }",
                "}",
                "",
            ]
        return lines

    # -- reference matvecs for the direct check of a rewritten fom_ops.c ---
    def _ops_editable(self) -> bool:
        """Whether the code optimization phase may rewrite ``fom_ops.c``. The one
        condition for both the direct check (`_checked_ops`) and the
        ``fom_ops.c`` kernel rules (`_kernel_rule_kinds`): neither is
        shipped without the other."""
        return self.ops_source_path() in self.optimization_editable_paths()

    def _checked_call(self, kind: str, name: str) -> str:
        """The function the direct checks call for ``fom_<kind>_<name>`` (``kind``
        ``apply`` or ``lin_solve``) on host pointers: the routine itself here; a
        device-staging wrapper on CUDA."""
        return f"fom_{kind}_{name}"

    def _verify_check_includes(self, ir: FomIR) -> list[str]:
        """Extra includes the direct checks need in ``verify_main.c``."""
        return []

    def ops_data_path(self) -> str:
        """Where the matrix data ``fom_ops.c`` reads lives (package-owned)."""
        return "src/fom_data.c"

    def _checked_ops(self, ir: FomIR) -> list[str]:
        """Matrices whose ``fom_apply_<name>`` ``make verify`` checks on its own.

        Only when the code optimization phase may rewrite ``fom_ops.c``
        (`_ops_editable`): each routine is held to a package-written
        reference over the same ``fom_data.c`` arrays, so a rewrite cannot change
        the matrix it applies.
        """
        return _matvec_matrix_names(ir) if self._ops_editable() else []

    def _ops_ref_header(self, ir: FomIR) -> str:
        lines = [
            "#ifndef FOM_OPS_REF_H",
            "#define FOM_OPS_REF_H",
            "",
            '#include "fom_workspace.h"',
            "",
            "/* Package-written reference matvecs (out += M @ x), the ones generated",
            "   into fom_ops.c before any rewrite. Built into the verify binary only. */",
        ]
        lines += [
            f"void fom_ref_apply_{name}(const fom_float *x, fom_float *out);"
            for name in self._checked_ops(ir)
        ]
        lines += [
            _fused_signature(ir, block, f"fom_ref_fused_{block.name}") + ";"
            for block in self._fused(ir)
        ]
        if any(b.activated for b in self._steps(ir)):
            lines.insert(4, '#include "fom_ops.h"')
        lines += self._step_ops_ref(ir)[0]
        lines += ["", "#endif /* FOM_OPS_REF_H */", ""]
        return "\n".join(lines)

    def _ops_ref_source(self, ir: FomIR) -> str:
        lines = ['#include "ops_ref.h"', '#include "fom_data.h"', ""]
        for name in self._checked_ops(ir):
            lines += self._apply_routine(name, ir.matrices[name], f"fom_ref_apply_{name}")
        for block in self._fused(ir):
            # Composed from the members' references, one product at a time.
            lines += [
                _fused_signature(ir, block, f"fom_ref_fused_{block.name}") + " {",
                f"    static fom_float tmp[{block.rows}];",
                "    fom_int i;",
            ]
            for t in block.terms:
                scale = f"{_fused_scale(t.scale)} * " if t.scale else ""
                lines += [
                    f"    for (i = 0; i < {block.rows}; ++i) tmp[i] = 0.0;",
                    f"    fom_ref_apply_{t.matrix}({_fused_src(t.source)}, tmp);",
                    f"    for (i = 0; i < {block.rows}; ++i) out[i] += {scale}tmp[i];",
                ]
            lines += ["}", ""]
        lines += self._step_ops_ref(ir)[1]
        return "\n".join(lines)

    def _ops_basis_probes(self, md) -> bool:
        """Probe every unit vector (an exact check) when it fits the work budget."""
        rows = md.shape[0]
        cols = md.shape[1] if len(md.shape) > 1 else 1
        nnz = md.nnz() if _is_csr(md) else rows * cols
        return cols * (nnz + rows) <= self._OPS_PROBE_BUDGET

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

    def _objective_file(self, ir: FomIR) -> str:
        """``src/fom_objective.c``: ``fom_objective`` on its own, so a rewritten
        ``fom_ops.c`` can never change the metric the solver stops on. Written
        for every IR (the build lists it unconditionally); without an objective
        it holds no code."""
        lines = ['#include "fom_ops.h"', '#include "fom_data.h"', ""]
        if ir.objective is None:
            lines += ["/* No verification objective: this solver does not stop on one. */", ""]
        lines += self._objective_source(ir)
        return "\n".join(lines)

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

        Mirrors `fomtools.codegen.factorize._pcg_solve` exactly (same
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

    #: Random probes per matrix in the direct matvec check. They run on top of the
    #: unit-vector probes (which alone miss a routine that branches on its input)
    #: and are the whole check for a matrix over the probe budget.
    _OPS_CHECK_RANDOM = 2

    #: Work budget (about ``cols * (nnz + rows)``) up to which the matvec check
    #: probes every unit vector, recovering the routine's matrix column by column.
    _OPS_PROBE_BUDGET = 1e8

    def _checked_factorizations(self, ir: FomIR) -> list:
        """Direct factorizations whose C solve ``make verify`` checks on its own.

        Only when the code optimization phase may rewrite the linear solve
        (`optimization_editable_paths`): the end-to-end check sees a solve
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
                f"            {self._checked_call('lin_solve', name)}(FOM_VERIFY_LS_{up}_B[r], x);",
                f"            for (j = 0; j < {fac.dim}; ++j) {{",
                f"                fom_float d = x[j] - FOM_VERIFY_LS_{up}_X[r][j];",
                f"                fom_float a = FOM_VERIFY_LS_{up}_X[r][j];",
                "                if (d < 0) d = -d;",
                "                if (a < 0) a = -a;",
                "                if (!(d <= err)) err = d;  /* NaN-safe: a NaN difference sticks */",
                "                if (1.0 + a > scale) scale = 1.0 + a;",
                "            }",
                "            if (!(err <= tol * scale)) {",
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
        the intervals in `_VERIFY_CHECK_INTERVALS`, and a ``max_iter`` that
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

    def _ops_check_data(self, ir: FomIR) -> list[str]:
        """Random probe vectors only: O(cols) per matrix, never a reference output."""
        lines = []
        rng = np.random.default_rng(self.config.seed + 2)
        k = self._OPS_CHECK_RANDOM
        for name in self._checked_ops(ir):
            md = ir.matrices[name]
            cols = md.shape[1] if len(md.shape) > 1 else 1
            probes = rng.standard_normal((k, cols))
            lines.append(_c_array_2d(f"FOM_VERIFY_OP_{name.upper()}_X", probes, f"[{k}][{cols}]"))
            lines.append("")
        return lines

    def _ops_check_main(self, ir: FomIR) -> list[str]:
        """Check each ``fom_apply_<name>`` against the package's reference matvec.

        Both start from the same nonzero ``out``, so the accumulate semantics are
        checked too; the tolerance scales with ``M @ x``, not with that starting
        value, so it stays relative to the entries being checked. Unit-vector
        probes (when within budget) compare the routine's matrix with ``M`` entry
        by entry; the random probes follow.
        """
        lines = []
        k = self._OPS_CHECK_RANDOM
        for name in self._checked_ops(ir):
            md = ir.matrices[name]
            rows = md.shape[0]
            cols = md.shape[1] if len(md.shape) > 1 else 1
            nb = cols if self._ops_basis_probes(md) else 0
            mode = "basis" if nb else "random"
            up = name.upper()
            lines += [
                "    {",
                f"        /* fom_apply_{name}: direct check against its matrix. */",
                f"        static fom_float x[{cols}], got[{rows}], ref[{rows}];",
                "        fom_float worst_op = 0.0;",
                "        int r, i, c;",
                f"        for (r = 0; r < {nb + k}; ++r) {{",
                "            fom_float err = 0.0, scale = 1.0;",
                "            int bad = 0;",
                f"            for (c = 0; c < {cols}; ++c)",
                f"                x[c] = (r < {nb}) ? (fom_float)(c == r) "
                f": FOM_VERIFY_OP_{up}_X[r - {nb}][c];",
                f"            for (i = 0; i < {rows}; ++i) got[i] = ref[i] = (fom_float)(i % 7 - 3);",
                f"            {self._checked_call('apply', name)}(x, got);",
                f"            fom_ref_apply_{name}(x, ref);",
                f"            for (i = 0; i < {rows}; ++i) {{",
                "                fom_float d = got[i] - ref[i];",
                "                fom_float a = ref[i] - (fom_float)(i % 7 - 3);  /* (M x)[i] */",
                "                if (d < 0) d = -d;",
                "                if (a < 0) a = -a;",
                "                if (!(d <= err)) { err = d; bad = i; }  /* NaN-safe */",
                "                if (1.0 + a > scale) scale = 1.0 + a;",
                "            }",
                "            if (!(err <= worst_op)) worst_op = err;",
                "            if (!(err <= ops_tol * scale)) {",
                f"                if (r < {nb})",
                f'                    printf("VERIFY FAILED: fom_apply_{name} entry (%d, %d) is "',
                '                           "off by %.3e from its matrix\\n", bad, r, (double)err);',
                "                else",
                f'                    printf("VERIFY FAILED: fom_apply_{name} differs from its "',
                '                           "matrix by %.3e in row %d on random probe %d\\n",',
                f"                           (double)err, bad, r - {nb});",
                "                return 1;",
                "            }",
                "        }",
                f'        printf("OPSCHECK name={name} mode={mode} probes={nb + k} '
                'max_err=%.3e\\n", (double)worst_op);',
                "    }",
            ]
        return lines

    def _fused_basis_probes(self, ir: FomIR, block: FusedBlock) -> bool:
        """Probe every unit vector of every source, at every scale setting, when
        that fits the same work budget as the per-matrix check."""
        work = 0
        for t in block.terms:
            md = ir.matrices[t.matrix]
            rows, cols = md.shape[0], md.shape[1] if len(md.shape) > 1 else 1
            work += cols * ((md.nnz() if _is_csr(md) else rows * cols) + rows)
        return work * (1 + len(block.scales)) <= self._OPS_PROBE_BUDGET

    def _fused_check_data(self, ir: FomIR) -> list[str]:
        """Random sources and schedule values for the fused check, O(cols) each."""
        lines = []
        rng = np.random.default_rng(self.config.seed + 3)
        k = self._OPS_CHECK_RANDOM
        for block in self._fused(ir):
            up = block.name.upper()
            for si, source in enumerate(block.sources):
                n = _source_dim(ir, block, source)
                probes = rng.standard_normal((k, n))
                lines.append(_c_array_2d(f"FOM_VERIFY_FUSED_{up}_X{si}", probes, f"[{k}][{n}]"))
            if block.scales:
                scales = rng.uniform(-2.0, 2.0, (k, len(block.scales)))
                lines.append(
                    _c_array_2d(f"FOM_VERIFY_FUSED_{up}_SC", scales, f"[{k}][{len(block.scales)}]")
                )
            lines.append("")
        return lines

    def _fused_check_main(self, ir: FomIR) -> list[str]:
        """Check each ``fom_fused_<block>`` against the sum of its members' references.

        Unit-vector probes of one source at a time (the others zero), first with
        every schedule value at 0 and then with each alone at 1: at each setting
        the routine is linear in the probed source, so these recover, entry by
        entry, the unscaled members and each scaled member of every source. The
        random probes (random sources and schedule values) follow; they catch a
        routine that is not linear in its inputs or branches on a schedule value.
        Same starting ``out`` and tolerance as the per-matrix check.
        """
        lines = []
        k = self._OPS_CHECK_RANDOM
        for block in self._fused(ir):
            up = block.name.upper()
            fn = f"fom_fused_{block.name}"
            srcs = [_fused_src(s) for s in block.sources]
            dims = [_source_dim(ir, block, s) for s in block.sources]
            ns = len(block.scales)
            basis = self._fused_basis_probes(ir, block)
            settings = ["all schedule values 0"] + [f"{_fused_scale(c)} = 1" for c in block.scales]
            call_args = ", ".join(f"x{si}" for si in range(len(srcs)))
            call_args += "".join(f", sc[{c}]" for c in range(ns))
            rows = block.rows

            def check(fail: list[str]) -> list[str]:
                return [
                    f"            for (i = 0; i < {rows}; ++i) got[i] = ref[i] = (fom_float)(i % 7 - 3);",
                    f"            {fn}({call_args}, got);",
                    f"            fom_ref_fused_{block.name}({call_args}, ref);",
                    "            err = 0.0; scale = 1.0; bad = 0;",
                    f"            for (i = 0; i < {rows}; ++i) {{",
                    "                fom_float d = got[i] - ref[i];",
                    "                fom_float a = ref[i] - (fom_float)(i % 7 - 3);",
                    "                if (d < 0) d = -d;",
                    "                if (a < 0) a = -a;",
                    "                if (!(d <= err)) { err = d; bad = i; }  /* NaN-safe */",
                    "                if (1.0 + a > scale) scale = 1.0 + a;",
                    "            }",
                    "            if (!(err <= worst_op)) worst_op = err;",
                    "            if (!(err <= ops_tol * scale)) {",
                    *fail,
                    "                return 1;",
                    "            }",
                ]

            def zero_sources() -> list[str]:
                return [
                    f"            for (c = 0; c < {n}; ++c) x{si}[c] = 0.0;"
                    for si, n in enumerate(dims)
                ]

            lines += [
                "    {",
                f"        /* {fn}: direct check against its members, {_fused_formula(block)}. */",
                *[f"        static fom_float x{si}[{n}];" for si, n in enumerate(dims)],
                f"        static fom_float got[{rows}], ref[{rows}];",
                f"        fom_float sc[{max(ns, 1)}];",
                "        static const char *setting[] = {"
                + ", ".join(f'"{t}"' for t in settings)
                + "};",
                "        fom_float worst_op = 0.0, err, scale;",
                "        int r, i, c, j, bad, probes = 0;",
                "        (void)setting; (void)j; (void)sc;",
            ]
            if basis:
                for si, (src, n) in enumerate(zip(srcs, dims)):
                    lines += [
                        f"        for (j = 0; j < {n}; ++j) {{",
                        f"          for (r = 0; r <= {ns}; ++r) {{",
                        *zero_sources(),
                        f"            x{si}[j] = 1.0;",
                        *[f"            sc[{c}] = (fom_float)(r == {c + 1});" for c in range(ns)],
                        "            ++probes;",
                        *check(
                            [
                                f'                printf("VERIFY FAILED: {fn} entry (%d, %d) of '
                                f'{src} is off by %.3e from its matrices (%s)\\n",',
                                "                       bad, j, (double)err, setting[r]);",
                            ]
                        ),
                        "          }",
                        "        }",
                    ]
            lines += [
                f"        for (r = 0; r < {k}; ++r) {{",
                *[
                    f"            for (c = 0; c < {n}; ++c) x{si}[c] = FOM_VERIFY_FUSED_{up}_X{si}[r][c];"
                    for si, n in enumerate(dims)
                ],
                *[f"            sc[{c}] = FOM_VERIFY_FUSED_{up}_SC[r][{c}];" for c in range(ns)],
                "            ++probes;",
                *check(
                    [
                        f'                printf("VERIFY FAILED: {fn} differs from its matrices '
                        'by %.3e in row %d on random probe %d\\n", (double)err, bad, r);',
                    ]
                ),
                "        }",
                f'        printf("OPSCHECK name=fused_{block.name} mode={"basis" if basis else "random"} '
                'probes=%d max_err=%.3e\\n", probes, (double)worst_op);',
                "    }",
            ]
        return lines

    def _step_basis_probes(self, ir: FomIR, blocks: list[FusedBlock], settings: int) -> bool:
        """Whether unit-vector probes of ``blocks`` (members' work per probe, times
        ``settings`` schedule settings) fit the work budget."""
        work = 0
        for block in blocks:
            for t in block.terms:
                md = ir.matrices[t.matrix]
                rows, cols = md.shape[0], md.shape[1] if len(md.shape) > 1 else 1
                work += cols * ((md.nnz() if _is_csr(md) else rows * cols) + rows)
        return work * settings <= self._OPS_PROBE_BUDGET

    def _iterate_basis_probes(self, ir: FomIR, blocks: list[StepBlock]) -> bool:
        """Every unit vector of the state, at ``2 + #schedules`` settings, through
        every block of the iteration."""
        work = 0
        for block in blocks:
            for t in block.terms:
                md = ir.matrices[t.matrix]
                rows, cols = md.shape[0], md.shape[1] if len(md.shape) > 1 else 1
                work += (md.nnz() if _is_csr(md) else rows * cols) + rows
        settings = 2 + len(_iterate_scales(blocks))
        return ir.state_dim * settings * work <= self._OPS_PROBE_BUDGET

    def _step_check_data(self, ir: FomIR) -> list[str]:
        """Random sources, schedule values, biases and activation draws for the
        step, rest and iterate checks, O(rows + cols) each."""
        lines = []
        rng = np.random.default_rng(self.config.seed + 4)
        k = self._OPS_CHECK_RANDOM
        checked = {b.name: b for b in self._steps(ir)}
        checked.update({b.name: b for _, b, _ in self._rests(ir) if b.name not in checked})
        for name, block in checked.items():
            up = f"FOM_VERIFY_STEP_{name.upper()}"
            for si, source in enumerate(block.sources):
                n = _source_dim(ir, block, source)
                lines.append(_c_array_2d(f"{up}_X{si}", rng.standard_normal((k, n)), f"[{k}][{n}]"))
            if block.scales:
                ns = len(block.scales)
                lines.append(_c_array_2d(f"{up}_SC", rng.uniform(-2, 2, (k, ns)), f"[{k}][{ns}]"))
            r = block.rows
            lines.append(_c_array_2d(f"{up}_BIAS", rng.standard_normal((k, r)), f"[{k}][{r}]"))
            lines.append(_c_array_2d(f"{up}_R1", rng.standard_normal((k, r)), f"[{k}][{r}]"))
            lines.append(
                _c_array_2d(f"{up}_R2", np.abs(rng.standard_normal((k, r))), f"[{k}][{r}]")
            )
            lines.append("")
        blocks = self._iterate(ir)
        if blocks:
            n = ir.state_dim
            lines.append(
                _c_array_2d("FOM_VERIFY_ITER_X", rng.standard_normal((k, n)), f"[{k}][{n}]")
            )
            scales = _iterate_scales(blocks)
            ns = max(len(scales), 1)
            lines.append(
                _c_array_2d(
                    "FOM_VERIFY_ITER_SC", rng.uniform(-2, 2, (k + 1, ns)), f"[{k + 1}][{ns}]"
                )
            )
            for b in blocks:
                up = f"FOM_VERIFY_ITER_{b.name.upper()}"
                lines.append(
                    _c_array_2d(f"{up}_BIAS", rng.standard_normal((k, b.rows)), f"[{k}][{b.rows}]")
                )
                if b.activated:
                    lines.append(
                        _c_array_2d(
                            f"{up}_R1", rng.standard_normal((k, b.rows)), f"[{k}][{b.rows}]"
                        )
                    )
                    lines.append(
                        _c_array_2d(
                            f"{up}_R2", np.abs(rng.standard_normal((k, b.rows))), f"[{k}][{b.rows}]"
                        )
                    )
            lines.append("")
        return lines

    @staticmethod
    def _act_linear(block: StepBlock, act: str, lo: str, hi: str, indent: str) -> list[str]:
        """Set the harness ``act`` so the activation is the identity."""
        kind = block.act.kind
        if kind == ActivationKind.SOFT_THRESHOLD:
            return [f"{indent}{act}.lambd = 0.0;"]
        lines = [f"{indent}for (i = 0; i < {block.rows}; ++i) {lo}[i] = -FOM_VERIFY_HUGE;"]
        lines.append(f"{indent}{act}.lo = {lo};")
        if kind == ActivationKind.SATURATED_LINEAR:
            lines.append(f"{indent}for (i = 0; i < {block.rows}; ++i) {hi}[i] = FOM_VERIFY_HUGE;")
            lines.append(f"{indent}{act}.hi = {hi};")
        return lines

    @staticmethod
    def _act_random(
        block: StepBlock, act: str, lo: str, hi: str, pre: str, r1: str, r2: str, indent: str
    ) -> list[str]:
        """Set the harness ``act`` to random parameters around the pre-activation
        values ``pre``, so the activation is active on about half the rows."""
        n = block.rows
        lines = [
            f"{indent}m = 0.0;",
            f"{indent}for (i = 0; i < {n}; ++i) m += (({pre}[i] < 0) ? -{pre}[i] : {pre}[i]);",
            f"{indent}m = (m > 0) ? m / {n} : 1.0;",
        ]
        kind = block.act.kind
        if kind == ActivationKind.SOFT_THRESHOLD:
            return lines + [f"{indent}{act}.lambd = 0.5 * m;"]
        lines.append(f"{indent}for (i = 0; i < {n}; ++i) {lo}[i] = {pre}[i] + {r1}[i] * m;")
        lines.append(f"{indent}{act}.lo = {lo};")
        if kind == ActivationKind.SATURATED_LINEAR:
            lines.append(f"{indent}for (i = 0; i < {n}; ++i) {hi}[i] = {lo}[i] + {r2}[i] * m;")
            lines.append(f"{indent}{act}.hi = {hi};")
        return lines

    def _step_check_main(self, ir: FomIR) -> list[str]:
        """Check each step routine, and each ``__rest`` routine, against the reference.

        Step routines overwrite ``out``: the routine and the reference start from
        different garbage. With the harness activation switched off (``lambd = 0``,
        or bounds at +-``FOM_VERIFY_HUGE``) a step is linear in each source, so
        unit-vector probes of one source at a time (the others zero, a random
        bias), first with every schedule value at 0 and then with each alone at 1,
        recover every member and the bias pass-through entry by entry. Random
        probes follow with the activation off, with the package's
        ``FOM_ACT_<block>``, and with random parameters around the pre-activation
        values (active on about half the rows), so the activation itself is held
        to the reference. A ``__rest`` routine is held to the full reference on
        its rows and must leave every selection row bit-for-bit untouched.
        """
        lines = []
        k = self._OPS_CHECK_RANDOM
        routines = [(f"fom_step_{b.name}", b, frozenset()) for b in self._steps(ir)]
        routines += [(fn, b, skip) for fn, b, skip in self._rests(ir)]
        for fn, block, skip in routines:
            step = fn.startswith("fom_step_")
            up = f"FOM_VERIFY_STEP_{block.name.upper()}"
            srcs = [_fused_src(x) for x in block.sources]
            dims = [_source_dim(ir, block, x) for x in block.sources]
            ns = len(block.scales)
            rows = block.rows
            act = step and getattr(block, "activated", False)
            basis = self._step_basis_probes(ir, [block], 1 + ns)
            ref_fn = f"fom_ref_step_{block.name}" if step else f"fom_ref_fused_{block.name}"
            settings = ["all schedule values 0"] + [f"{_fused_scale(c)} = 1" for c in block.scales]
            modes = (
                ["activation off", "package activation", "random activation"] if act else ["random"]
            )
            args = ", ".join(f"x{si}" for si in range(len(srcs)))
            args += "".join(f", sc[{c}]" for c in range(ns))
            if step:
                args += ", bias"

            def call(f: str, use: str, out: str) -> str:
                return f"            {f}({args}{', ' + use if act else ''}, {out});"

            def check(use: str, fail: list[str]) -> list[str]:
                init_ref = "-(fom_float)(i % 5) - 0.25" if step else "got[i]"
                out = [
                    f"            for (i = 0; i < {rows}; ++i) {{",
                    "                got[i] = (fom_float)(i % 7) - 3.5;",
                    f"                ref[i] = {init_ref};",
                    "            }",
                    call(fn, use, "got"),
                    call(ref_fn, use, "ref"),
                    "            err = 0.0; scale = 1.0; bad = 0; touched = -1;",
                    f"            for (i = 0; i < {rows}; ++i) {{",
                ]
                if skip:
                    out += [
                        "                if (sel[i]) {",
                        "                    if (!(got[i] == (fom_float)(i % 7) - 3.5)) touched = i;",
                        "                    continue;",
                        "                }",
                    ]
                base = "" if step else " - ((fom_float)(i % 7) - 3.5)"
                out += [
                    "                {",
                    "                    fom_float d = got[i] - ref[i];",
                    f"                    fom_float a = ref[i]{base}, b = pre[i];",
                    "                    if (d < 0) d = -d;",
                    "                    if (a < 0) a = -a;",
                    "                    if (b < 0) b = -b;",
                    "                    if (b > a) a = b;",
                    "                    if (!(d <= err)) { err = d; bad = i; }  /* NaN-safe */",
                    "                    if (1.0 + a > scale) scale = 1.0 + a;",
                    "                }",
                    "            }",
                    "            if (!(err <= worst_op)) worst_op = err;",
                    "            if (touched >= 0) {",
                    f'                printf("VERIFY FAILED: {fn} changed selection row %d, which it '
                    'must leave untouched\\n", touched);',
                    "                return 1;",
                    "            }",
                    "            if (!(err <= ops_tol * scale)) {",
                    *fail,
                    "                return 1;",
                    "            }",
                ]
                return out

            sel = [1 if i in skip else 0 for i in range(rows)]
            lines += [
                "    {",
                f"        /* {fn}: direct check against its members"
                + (" and the reference activation" if act else "")
                + (", leaving its selection rows untouched" if skip else "")
                + ". */",
                *[f"        static fom_float x{si}[{n}];" for si, n in enumerate(dims)],
                f"        static fom_float bias[{rows}], got[{rows}], ref[{rows}], pre[{rows}];",
                f"        static fom_float act_lo[{rows}], act_hi[{rows}];",
                *(
                    [
                        f"        static const unsigned char sel[{rows}] = {{{', '.join(map(str, sel))}}};"
                    ]
                    if skip
                    else []
                ),
                *([f"        {_act_type(block)} hact;"] if act else []),
                f"        fom_float sc[{max(ns, 1)}];",
                "        static const char *setting[] = {"
                + ", ".join(f'"{t}"' for t in settings + modes)
                + "};",
                "        fom_float worst_op = 0.0, err, scale, m;",
                "        int r, i, c, j, bad, touched, mode, probes = 0;",
                "        (void)setting; (void)j; (void)sc; (void)m; (void)mode; (void)act_lo;",
                "        (void)act_hi; (void)bias;",
                f"        for (i = 0; i < {rows}; ++i) pre[i] = 0.0;",
            ]
            if act:
                lines += self._act_linear(block, "hact", "act_lo", "act_hi", "        ")
            if basis:
                for si, (src, n) in enumerate(zip(srcs, dims)):
                    lines += [
                        f"        for (i = 0; i < {rows}; ++i) bias[i] = {up}_BIAS[0][i];",
                        f"        for (j = 0; j < {n}; ++j) {{",
                        f"          for (r = 0; r <= {ns}; ++r) {{",
                        *[
                            f"            for (c = 0; c < {d}; ++c) x{sj}[c] = 0.0;"
                            for sj, d in enumerate(dims)
                        ],
                        f"            x{si}[j] = 1.0;",
                        *[f"            sc[{c}] = (fom_float)(r == {c + 1});" for c in range(ns)],
                        "            ++probes;",
                        *check(
                            "&hact",
                            [
                                f'                printf("VERIFY FAILED: {fn} entry (%d, %d) of '
                                f'{src} is off by %.3e from its matrices (%s)\\n",',
                                "                       bad, j, (double)err, setting[r]);",
                            ],
                        ),
                        "          }",
                        "        }",
                    ]
            uses = ["&hact", f"&FOM_ACT_{block.name}", "&hact"] if act else ["&hact"]
            for mode, use in enumerate(uses):
                lines += [
                    f"        for (r = 0; r < {k}; ++r) {{",
                    *[
                        f"            for (c = 0; c < {n}; ++c) x{si}[c] = {up}_X{si}[r][c];"
                        for si, n in enumerate(dims)
                    ],
                    *[f"            sc[{c}] = {up}_SC[r][{c}];" for c in range(ns)],
                    f"            for (i = 0; i < {rows}; ++i) bias[i] = {up}_BIAS[r][i];",
                ]
                if act:
                    lines += self._act_linear(block, "hact", "act_lo", "act_hi", "            ")
                    ref_step = f"fom_ref_step_{block.name}"
                    lines.append(f"            {ref_step}({args}, &hact, pre);")
                    if mode == 2:
                        lines += self._act_random(
                            block,
                            "hact",
                            "act_lo",
                            "act_hi",
                            "pre",
                            f"{up}_R1[r]",
                            f"{up}_R2[r]",
                            "            ",
                        )
                lines += [
                    "            ++probes;",
                    *check(
                        use,
                        [
                            f'                printf("VERIFY FAILED: {fn} differs from its reference '
                            f'by %.3e in row %d on random probe %d ({modes[mode]})\\n",',
                            "                       (double)err, bad, r);",
                        ],
                    ),
                    "        }",
                ]
                if act:
                    lines.append(f"        for (i = 0; i < {rows}; ++i) pre[i] = 0.0;")
                    lines += self._act_linear(block, "hact", "act_lo", "act_hi", "        ")
            name = fn[len("fom_") :]
            lines += [
                f'        printf("OPSCHECK name={name} mode={"basis" if basis else "random"} '
                'probes=%d max_err=%.3e\\n", probes, (double)worst_op);',
                "    }",
            ]
        return lines

    def _iterate_check_main(self, ir: FomIR) -> list[str]:
        """Check ``fom_iterate`` against the chained reference steps.

        Every output is compared: each temporary (which pins its block) and the
        new state. With every activation switched off and zero biases the
        iteration is linear in the state, so unit-vector probes at each schedule
        setting (all 0, each alone at 1, and one random setting, which catches
        products of schedule values across blocks) recover its matrix entry by
        entry. Random states then follow with random biases, with the activations
        off, at the package's ``FOM_ACT_<block>``, and at random parameters around
        each block's activation-off values.
        """
        blocks = self._iterate(ir)
        if not blocks:
            return []
        k = self._OPS_CHECK_RANDOM
        n = ir.state_dim
        scales = _iterate_scales(blocks)
        ns = len(scales)
        activated = [b for b in blocks if b.activated]
        outs = [(f"t_{b.name}", b.rows) for b in blocks if b.name != "su"]
        outs.append(("new_state", ir.state_dim))
        basis = self._iterate_basis_probes(ir, blocks)
        settings = ["all schedule values 0"]
        settings += [f"{_fused_scale(c)} = 1" for c in scales]
        settings.append("random schedule values")
        modes = ["activations off", "package activations", "random activations"]
        modes = modes if activated else modes[:1]

        def args(prefix: str, acts: str) -> str:
            a = ["x"] + [f"sc[{c}]" for c in range(ns)]
            a += [f"bias_{b.name}" for b in blocks]
            a += [f"{acts}{b.name}" for b in activated]
            a += [f"{prefix}_{name}" for name, _ in outs]
            return ", ".join(a)

        def check(acts: str, fail: list[str]) -> list[str]:
            lines = []
            for name, rows in outs:
                lines += [
                    f"            for (i = 0; i < {rows}; ++i) {{ g_{name}[i] = (fom_float)(i % 7) - 3.5; "
                    f"r_{name}[i] = -(fom_float)(i % 5) - 0.25; }}"
                ]
            lines += [
                f"            fom_iterate({args('g', acts)});",
                f"            fom_ref_iterate({args('r', acts)});",
                "            err = 0.0; scale = 1.0; bad = 0; which = 0;",
            ]
            for o, (name, rows) in enumerate(outs):
                lines += [
                    f"            for (i = 0; i < {rows}; ++i) {{",
                    f"                fom_float d = g_{name}[i] - r_{name}[i], a = r_{name}[i];",
                    "                if (d < 0) d = -d;",
                    "                if (a < 0) a = -a;",
                    f"                if (!(d <= err)) {{ err = d; bad = i; which = {o}; }}",
                    "                if (1.0 + a > scale) scale = 1.0 + a;",
                    "            }",
                ]
            lines += [
                "            if (!(err <= worst_op)) worst_op = err;",
                "            if (!(err <= ops_tol * scale)) {",
                *fail,
                "                return 1;",
                "            }",
            ]
            return lines

        names = "{" + ", ".join(f'"{name}"' for name, _ in outs) + "}"
        lines = [
            "    {",
            "        /* fom_iterate: direct check against the chained reference steps. */",
            f"        static fom_float x[{n}];",
            *[f"        static fom_float bias_{b.name}[{b.rows}];" for b in blocks],
            *[
                f"        static fom_float g_{name}[{rows}], r_{name}[{rows}];"
                for name, rows in outs
            ],
            *[
                f"        static fom_float lo_{b.name}[{b.rows}], hi_{b.name}[{b.rows}];"
                for b in activated
            ],
            *[f"        {_act_type(b)} hact_{b.name};" for b in activated],
            *[
                f"        const {_act_type(b)} *pact_{b.name} = &FOM_ACT_{b.name};"
                for b in activated
            ],
            f"        fom_float sc[{max(ns, 1)}];",
            "        static const char *setting[] = {"
            + ", ".join(f'"{t}"' for t in settings + modes)
            + "};",
            f"        static const char *output[] = {names};",
            "        fom_float worst_op = 0.0, err, scale, m;",
            "        int r, i, j, bad, which, probes = 0;",
            "        (void)setting; (void)output; (void)j; (void)sc; (void)m;",
        ]
        if activated:
            voids = [f"{v}_{b.name}" for b in activated for v in ("pact", "lo", "hi")]
            lines.append("        (void)" + "; (void)".join(voids) + ";")
        for b in activated:
            lines += self._act_linear(
                b, f"hact_{b.name}", f"lo_{b.name}", f"hi_{b.name}", "        "
            )
        hact = "&hact_"
        if basis:
            lines += [
                *[
                    f"        for (i = 0; i < {b.rows}; ++i) bias_{b.name}[i] = 0.0;"
                    for b in blocks
                ],
                f"        for (j = 0; j < {n}; ++j) {{",
                f"          for (r = 0; r <= {ns + 1}; ++r) {{",
                f"            for (i = 0; i < {n}; ++i) x[i] = 0.0;",
                "            x[j] = 1.0;",
                *[
                    f"            sc[{c}] = (r == {ns + 1}) ? FOM_VERIFY_ITER_SC[{k}][{c}] "
                    f": (fom_float)(r == {c + 1});"
                    for c in range(ns)
                ],
                "            ++probes;",
                *check(
                    hact,
                    [
                        '                printf("VERIFY FAILED: fom_iterate entry (%d, %d) of %s is '
                        'off by %.3e from the reference (%s)\\n",',
                        "                       bad, j, output[which], (double)err, setting[r]);",
                    ],
                ),
                "          }",
                "        }",
            ]
        for mode in range(len(modes)):
            lines += [
                f"        for (r = 0; r < {k}; ++r) {{",
                f"            for (i = 0; i < {n}; ++i) x[i] = FOM_VERIFY_ITER_X[r][i];",
                *[f"            sc[{c}] = FOM_VERIFY_ITER_SC[r][{c}];" for c in range(ns)],
                *[
                    f"            for (i = 0; i < {b.rows}; ++i) "
                    f"bias_{b.name}[i] = FOM_VERIFY_ITER_{b.name.upper()}_BIAS[r][i];"
                    for b in blocks
                ],
            ]
            acts = hact
            if mode == 1:
                acts = "pact_"
            if mode == 2:
                # Activation-off reference values of every block, then random
                # parameters around each.
                lines.append(f"            fom_ref_iterate({args('r', hact)});")
                for b in activated:
                    up = f"FOM_VERIFY_ITER_{b.name.upper()}"
                    lines += self._act_random(
                        b,
                        f"hact_{b.name}",
                        f"lo_{b.name}",
                        f"hi_{b.name}",
                        f"r_t_{b.name}",
                        f"{up}_R1[r]",
                        f"{up}_R2[r]",
                        "            ",
                    )
            lines += [
                "            ++probes;",
                *check(
                    acts,
                    [
                        '                printf("VERIFY FAILED: fom_iterate differs from the '
                        f'reference by %.3e in %s row %d on random probe %d ({modes[mode]})\\n",',
                        "                       (double)err, output[which], bad, r);",
                    ],
                ),
            ]
            if mode == 2:
                for b in activated:
                    lines += self._act_linear(
                        b, f"hact_{b.name}", f"lo_{b.name}", f"hi_{b.name}", "            "
                    )
            lines.append("        }")
        lines += [
            f'        printf("OPSCHECK name=iterate mode={"basis" if basis else "random"} '
            'probes=%d max_err=%.3e\\n", probes, (double)worst_op);',
            "    }",
        ]
        return lines

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
        lines += self._ops_check_data(ir)
        lines += self._fused_check_data(ir)
        lines += self._step_check_data(ir)
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
            "            if (!(d <= err)) err = d;  /* NaN-safe */",
            "        }",
            "        if (!(err <= tol)) {",
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
                *(
                    ['#include "fom_ops.h"', '#include "ops_ref.h"']
                    if self._checked_ops(ir)
                    else []
                ),
                *self._verify_check_includes(ir),
                '#include "verify_data.h"',
                "",
                *(
                    [
                        "/* Activation bounds far outside any iterate: the activation is off. */",
                        "#define FOM_VERIFY_HUGE ((fom_float)(sizeof(fom_float) == 4 ? 1e30 : 1e300))",
                        "",
                    ]
                    if self._steps(ir)
                    else []
                ),
                *(self._verify_stop_helper() if objective else []),
                "int main(void) {",
                "    FomWorkspace work;",
                "    FomSettings settings;",
                "    FomParams params;",
                "    int s, j;",
                f"    fom_float tol = (fom_float){tol};",
                *(
                    [f"    fom_float ops_tol = (fom_float){_fmt_float(self.config.tol_kernel)};"]
                    if self._checked_ops(ir)
                    else []
                ),
                "    fom_float worst = 0.0;",
                "    int failures = 0;",
                *(["    int stop_failures = 0;", "    fom_int stop_status;"] if objective else []),
                "",
                *self._ops_check_main(ir),
                *self._fused_check_main(ir),
                *self._step_check_main(ir),
                *self._iterate_check_main(ir),
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
                "            if (!(d <= err)) err = d;  /* NaN-safe: a NaN difference sticks */",
                "        }",
                "        if (!(err <= worst)) worst = err;",
                "        if (!(err <= tol)) {",
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
                "# Link-time optimization lets the compiler inline fom_apply_* (fom_ops.c)",
                "# into the solver; override with LTOFLAGS= to build without it.",
                "LTOFLAGS ?= -flto",
                "BENCHFLAGS ?= -O3 -Iinclude",
                "SRC = src/fom_solver.c src/fom_data.c src/fom_linsolve.c src/fom_ops.c "
                "src/fom_objective.c",
                "# Package-written reference matvecs for verify's direct check of fom_ops.c.",
                "VERIFY_SRC = $(wildcard test/ops_ref.c)",
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
                f"verify:{guard} $(SRC) $(VERIFY_SRC) test/verify_main.c",
                _KERNEL_CHECK_RECIPE,
                "\t$(CC) $(CFLAGS) $(LTOFLAGS) -Itest $(SRC) $(VERIFY_SRC) test/verify_main.c -o verify_bin -lm",
                "\t./verify_bin",
                "",
                "# Build the shared library from the marker-free production source.",
                "release: release/fom_solver.c src/fom_data.c release/fom_linsolve.c "
                "release/fom_ops.c src/fom_objective.c",
                "\t$(CC) $(CFLAGS) $(LTOFLAGS) -shared -fPIC release/fom_solver.c src/fom_data.c "
                "release/fom_linsolve.c release/fom_ops.c src/fom_objective.c "
                "-o libfomsolver.$(LIBEXT) -lm",
                "",
                "bench: $(SRC) test/bench_main.c",
                "\t$(CC) $(BENCHFLAGS) $(LTOFLAGS) -Itest $(SRC) test/bench_main.c -o bench_bin -lm",
                "\t./bench_bin",
                "",
                "profile: $(SRC) src/fom_profile.c test/bench_main.c",
                "\t$(CC) $(BENCHFLAGS) $(LTOFLAGS) -DFOM_PROFILE -Itest $(SRC) src/fom_profile.c "
                "test/bench_main.c -o profile_bin -lm",
                "\t./profile_bin",
                "",
                "lib: $(SRC)",
                "\t$(CC) $(CFLAGS) $(LTOFLAGS) -shared -fPIC $(SRC) -o libfomsolver.$(LIBEXT) -lm",
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
    def kernel_checks(
        self, rel: str, source: str, ir: FomIR, baseline: str | None = None
    ) -> list[str]:
        """The `kernel_rules` for a rewritten solver or ``fom_ops.c``.

        The solver's rules hold on every target (its products go through
        ``fom_apply_*``, generated or checked); ``fom_ops.c``'s only where it is
        editable, and so checked against the matrices.
        """
        kind = self._kernel_rule_kinds().get(rel)
        if kind is None:
            return []
        return kernel_rules.check(
            kind,
            source,
            _matvec_matrix_names(ir),
            baseline,
            self._fused_members(ir),
            self._exact_routines(ir),
            self._step_owners(ir),
            self._step_vectors(ir),
        )

    def _step_owners(self, ir: FomIR) -> dict[str, list[str]]:
        """Each step, ``__rest`` and iterate routine with the matrices it may read."""
        owners = {f"fom_step_{b.name}": b.members for b in self._steps(ir)}
        owners.update({fn: b.members for fn, b, _ in self._rests(ir)})
        blocks = self._iterate(ir)
        if blocks:
            owners["fom_iterate"] = list(dict.fromkeys(m for b in blocks for m in b.members))
        return owners

    def _step_vectors(self, ir: FomIR) -> list[str]:
        """Constant and activation-bound vectors, which ``fom_ops.c`` may not read
        once step routines exist (they come in as ``bias`` and ``act``)."""
        if not self._steps(ir):
            return []
        names = [lin.const for lin in [s.linear_op for s in ir.body] + [ir.state_update]]
        names += [b for s in ir.body for b in (s.activation.lower, s.activation.upper)]
        return [n for n in dict.fromkeys(names) if n]

    def _exact_routines(self, ir: FomIR) -> list[str]:
        """The ``fom_ops.c`` routines ``make verify`` probes on every unit vector
        (an exact check), which may therefore hold coefficients as literals."""
        exact = [
            f"fom_apply_{name}"
            for name in self._checked_ops(ir)
            if self._ops_basis_probes(ir.matrices[name])
        ]
        exact += [
            f"fom_fused_{block.name}"
            for block in self._fused(ir)
            if self._fused_basis_probes(ir, block)
        ]
        exact += [
            f"fom_step_{b.name}"
            for b in self._steps(ir)
            if self._step_basis_probes(ir, [b], 1 + len(b.scales))
        ]
        exact += [
            fn
            for fn, b, _ in self._rests(ir)
            if self._step_basis_probes(ir, [b], 1 + len(b.scales))
        ]
        blocks = self._iterate(ir)
        if blocks and self._iterate_basis_probes(ir, blocks):
            exact.append("fom_iterate")
        return exact

    def _fused_members(self, ir: FomIR) -> dict[str, list[str]]:
        """Each ``fom_fused_<block>`` this target emits, with its member matrices."""
        return {block.name: block.members for block in self._fused(ir)}

    def fused_routines(self, ir: FomIR) -> dict[str, list[str]]:
        return self._fused_members(ir)

    def exact_routines(self, ir: FomIR) -> list[str]:
        return self._exact_routines(ir) if self._ops_editable() else []

    def step_routines(self, ir: FomIR) -> dict[str, list[str]]:
        return {f"fom_step_{b.name}": b.members for b in self._steps(ir)}

    def iterate_routine(self, ir: FomIR) -> bool:
        return bool(self._iterate(ir))

    def selection_rows(self, ir: FomIR) -> dict[str, str]:
        return {
            name: _selection_rows_description(block, rows)
            for name, (block, rows) in self._selrow_blocks(ir).items()
        }

    def rest_routines(self, ir: FomIR) -> list[str]:
        return [fn for fn, _, _ in self._rests(ir)]

    def selection_matrices(self, ir: FomIR) -> dict[str, str]:
        return {
            name: selection_description(src, ir.matrices[name].shape[1])
            for name, src in self._selection_data(ir).items()
        }

    def ops_check_info(self, ir: FomIR, name: str) -> tuple[str, tuple, int]:
        """``(routine, shape, nnz)`` of the routine behind an ``OPSCHECK`` name:
        a matrix, or ``fused_<block>`` (its output rows by the summed member
        columns, and the members' total nonzeros)."""
        if name == "iterate":
            blocks = self._iterate(ir)
            nnz = sum(ir.matrices[m].nnz() for b in blocks for m in b.members)
            return "fom_iterate", (ir.state_dim, ir.state_dim), nnz
        if name.startswith(("fused_", "step_")):
            kind, rest = name.split("_", 1)
            base = rest[: -len("__rest")] if rest.endswith("__rest") else rest
            pool = self._fused(ir) if kind == "fused" else self._steps(ir)
            block = next(b for b in pool if b.name == base)
            cols = sum(_source_dim(ir, block, s) for s in block.sources)
            nnz = sum(ir.matrices[m].nnz() for m in block.members)
            return f"fom_{name}", (block.rows, cols), nnz
        md = ir.matrices[name]
        return f"fom_apply_{name}", tuple(md.shape), md.nnz()

    def _kernel_rule_kinds(self) -> dict[str, str]:
        kinds = {self.solver_source_path(): "solver"}
        if self._ops_editable():
            kinds[self.ops_source_path()] = "ops"
        return kinds

    def translation_checks(self, source: str, ir: FomIR) -> list[str]:
        return kernel_rules.check("translation", source, _matvec_matrix_names(ir))

    def write_kernel_check_config(
        self, out_dir, ir: FomIR, baselines: dict[str, str] | None
    ) -> None:
        kinds = self._kernel_rule_kinds()
        if baselines is None:  # translation: the solver only, with no baseline
            files = [{"path": self.solver_source_path(), "kind": "translation"}]
        else:
            files = [
                {"path": rel, "kind": kind, "baseline": baselines.get(rel)}
                for rel, kind in kinds.items()
            ]
        config = {
            "matvec": _matvec_matrix_names(ir),
            "fused": self._fused_members(ir),
            "exact": self._exact_routines(ir),
            "owners": self._step_owners(ir),
            "vectors": self._step_vectors(ir),
            "files": files,
        }
        (Path(out_dir) / _KERNEL_CHECK_CONFIG).write_text(json.dumps(config, indent=1))

    def remove_kernel_check_config(self, out_dir) -> None:
        (Path(out_dir) / _KERNEL_CHECK_CONFIG).unlink(missing_ok=True)

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
