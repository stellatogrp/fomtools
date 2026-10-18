"""Lower the verification objective onto the generated loop's vectors.

The verification problem measures performance with an objective set by
:meth:`~fomtools.verifier.Verifier.set_objective_norm`: a norm of linear
expressions in the iterates, written at iteration ``K`` in terms of ``w^K`` and
``w^{K-1}`` (a fixed-point residual ``||w^K - w^{K-1}||``, a KKT residual, ...).
:attr:`~fomtools.codegen.config.TerminationKind.OBJECTIVE` stops the generated
solver on exactly that quantity, so this module maps the objective onto what
the rolled loop holds at the end of iteration ``k``:

* ``new_state`` (``w^k``) and ``state`` (``w^{k-1}``),
* the parameters,
* iteration ``k``'s body temporaries, when the objective was written on the
  iteration the IR was extracted from, and
* ReLU steps recorded *outside* the loop on those quantities, for one-sided terms
  such as ``max(Ax - u, 0)``.

The mapping is structural and exact. A leaf coordinate is matched to a state
row only when that row of the iterate expression *is* the coordinate (a "pure
copy"), and anything that does not match raises
:class:`ObjectiveExtractionError` naming what to change. Nothing is ever fitted
or approximated, so the generated stopping test computes the expression the
user wrote.

Two frames are tried, in order. The *extracted iteration*: the state leaves as
``w^0``, ``next_state`` as ``w^1``, plus that iteration's temporaries. And, when
the recorded tape is available, the *last recorded iteration*: ``w^{K-1}`` and
``w^K`` from the tape, without temporaries (iteration ``K``'s steps are not the
ones the IR was built from). The second is what a verification-style recording
uses, setting the objective after its loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as spa

from fomtools.codegen.extract import (
    _dependency_cone,
    _row_slice_expr,
    _sparse,
    _step_source_activation,
    _zeros,
    fresh_leaf_of,
)
from fomtools.codegen.ir import (
    ActivationKind,
    ActivationOp,
    FomIR,
    LinearOp,
    MatrixData,
    NonStationaryFOMError,
    ObjectiveBlock,
    ObjectiveSpec,
    Term,
)
from fomtools.linearexpression import LinearExpression, stack

__all__ = [
    "ObjectiveExtractionError",
    "extract_objective",
    "norm_label",
    "objective_c_params",
    "objective_call",
]


class ObjectiveExtractionError(ValueError):
    """The verification objective cannot be evaluated by the generated loop."""


def norm_label(ord) -> str:
    """The IR's name (``"1"``, ``"2"``, ``"inf"``) for a verifier norm order."""
    if ord == np.inf or ord == "inf":
        return "inf"
    if ord in (1, 2):
        return str(int(ord))
    raise ObjectiveExtractionError(f"unsupported objective norm order {ord!r}; use 1, 2 or inf")


# --------------------------------------------------------------------------
# The C interface of the generated ``fom_objective``, shared by every target
# and by SPEC.md.
# --------------------------------------------------------------------------
def objective_c_source(source: str) -> str:
    """The ``fom_objective`` argument holding an objective term's source."""
    return source.split(":", 1)[-1]


def objective_c_params(ir: FomIR) -> list[str]:
    """The C parameter list of ``fom_objective`` for ``ir``.

    ``it`` (the 0-based index of the iteration that produced ``new_state``),
    ``state`` (``w^{k-1}``), ``new_state`` (``w^k``), then each parameter and
    body temporary the objective reads (see :meth:`FomIR.objective_inputs`).
    """
    params = ["fom_int it", "const fom_float *state", "const fom_float *new_state"]
    params += [f"const fom_float *{objective_c_source(s)}" for s in ir.objective_inputs()]
    return params


def objective_call(
    ir: FomIR, state: str = "work->state", new_state: str = "work->new_state"
) -> str:
    """An example ``fom_objective`` call reading the workspace buffers."""
    args = ["it", state, new_state]
    for source in ir.objective_inputs():
        kind, name = source.split(":", 1)
        args.append(f"work->params.{name}" if kind == "param" else f"work->{name}")
    return f"fom_objective({', '.join(args)})"


def _copy_rows(exprs) -> dict[LinearExpression, np.ndarray]:
    """For each leaf, the row of ``concat(exprs)`` that is a pure copy of each of
    its coordinates (``-1`` where none is).

    A row is a pure copy of ``leaf[j]`` when it reads that coordinate with
    coefficient exactly 1 and nothing else, constant included. Only then is
    the row's value the coordinate's value, whatever the other leaves hold.
    """
    copies: dict[LinearExpression, np.ndarray] = {}
    offset = 0
    for expr in exprs:
        rows = expr.ndim
        blocks = {leaf: _sparse(coeff) for leaf, coeff in expr.decomposition_dict.items()}
        nnz_per_row = np.zeros(rows, dtype=int)
        for block in blocks.values():
            block.eliminate_zeros()
            nnz_per_row += np.diff(block.indptr)
        const = np.asarray(expr.const, dtype=float).reshape(-1)
        const = np.broadcast_to(const, (rows,)) if const.size == 1 else const
        for leaf, block in blocks.items():
            coo = block.tocoo()
            sole = (nnz_per_row[coo.row] == 1) & (coo.data == 1.0) & (const[coo.row] == 0.0)
            if not np.any(sole):
                continue
            target = copies.setdefault(leaf, np.full(leaf.ndim, -1, dtype=int))
            for r, j in zip(coo.row[sole], coo.col[sole]):
                if target[j] < 0:  # the first copy wins; any copy holds the same value
                    target[j] = offset + r
        offset += rows
    return copies


@dataclass
class _Frame:
    """What the loop holds at the end of one iteration, keyed by leaf."""

    label: str
    state_dim: int
    new_rows: dict[LinearExpression, np.ndarray]
    prev_rows: dict[LinearExpression, np.ndarray]
    tmps: dict[LinearExpression, tuple[str, int]] = field(default_factory=dict)


@dataclass
class _Mapped:
    """One objective expression's rows, over the loop's sources."""

    rows: int
    blocks: dict[str, spa.csr_matrix]  # source -> coefficients
    const: np.ndarray


class _Mapper:
    def __init__(self, frame: _Frame, param_map, relu_steps: dict, outside_params=frozenset()):
        self.frame = frame
        self.param_map = param_map  # leaf -> (name, dim)
        self.relu_steps = relu_steps  # leaf -> producing ReLU step outside the loop
        self.outside_params = outside_params  # parameters the iteration does not read

    def source_dims(self) -> dict[str, int]:
        dims = {"new_state": self.frame.state_dim, "state": self.frame.state_dim}
        dims.update({f"param:{n}": d for n, d in self.param_map.values()})
        dims.update({f"tmp:{n}": d for n, d in self.frame.tmps.values()})
        return dims

    def map(self, expr: LinearExpression, allow_relu: bool) -> tuple[_Mapped, dict]:
        """Map ``expr``; ReLU-step leaves are returned unmapped, as
        ``{leaf: coeff}``, for the caller to lower row by row."""
        rows = expr.ndim
        dims = self.source_dims()
        acc = {src: _zeros(rows, dim) for src, dim in dims.items()}
        relu_terms = {}
        for leaf, coeff in expr.decomposition_dict.items():
            block = _sparse(coeff)
            if leaf in self.param_map:
                acc[f"param:{self.param_map[leaf][0]}"] += block
                continue
            if allow_relu and leaf in self.relu_steps:
                relu_terms[leaf] = block
                continue
            left = np.ones(leaf.ndim, dtype=bool)
            for src, table in (("new_state", self.frame.new_rows), ("state", self.frame.prev_rows)):
                idx = table.get(leaf)
                if idx is None:
                    continue
                cols = np.flatnonzero(left & (idx >= 0))
                if cols.size:
                    select = spa.csr_matrix(
                        (np.ones(cols.size), (cols, idx[cols])), shape=(leaf.ndim, dims[src])
                    )
                    acc[src] += block @ select
                    left[cols] = False
            if not left.any():
                continue
            if leaf in self.frame.tmps:
                keep = spa.diags(left.astype(float))
                acc[f"tmp:{self.frame.tmps[leaf][0]}"] += block @ keep
                continue
            # Unresolved coordinates are an error only where the objective reads them.
            if (block @ spa.diags(left.astype(float))).count_nonzero():
                raise ObjectiveExtractionError(self._unresolved(leaf, allow_relu))
        const = np.asarray(expr.const, dtype=float).reshape(-1)
        const = np.broadcast_to(const, (rows,)).copy() if const.size == 1 else const.copy()
        blocks = {}
        for src, m in acc.items():
            m = m.tocsr()
            m.eliminate_zeros()  # coefficients that cancel must not count as reads
            blocks[src] = m
        return _Mapped(rows, blocks, const), relu_terms

    def _unresolved(self, leaf, allow_relu: bool) -> str:
        where = self.frame.label
        if leaf in self.relu_steps and not allow_relu:
            return (
                f"in {where}: a ReLU term of the objective is applied to another ReLU "
                "term; nested one-sided terms are not supported."
            )
        if leaf in self.outside_params:
            return (
                f"in {where}: the objective reads a parameter that the iteration does not "
                "use, and the generated solver only takes the iteration's parameters. "
                "Write the objective in terms of the iterates and the parameters the "
                "method reads."
            )
        return (
            f"in {where}: the objective reads a quantity (a {leaf.ndim}-vector) that the "
            "generated loop does not carry: it is neither a parameter, nor exactly part "
            "of the iterate w^k or w^{k-1}, nor (on the extracted iteration) one of its "
            "step outputs, nor a ReLU of those. Typical causes: an iterate older than "
            "w^{k-1} (carry it in the state, e.g. state=[x_prev, x]), or an iterate "
            "that is a non-leaf expression accumulated over earlier iterations (record "
            "it with verifier.iterate_trajectory so each iterate is its own variable, "
            "or set the objective on the first recorded iteration)."
        )


def _register(prefix: str, mapped: _Mapped, matrices: dict) -> LinearOp:
    """Store the non-zero blocks of ``mapped`` as IR matrices; return its op."""
    terms = []
    for source, block in mapped.blocks.items():
        if block.count_nonzero() == 0:
            continue
        suffix = {"new_state": "new", "state": "state"}.get(source)
        if suffix is None:
            kind, name = source.split(":", 1)
            suffix = f"{'p' if kind == 'param' else 't'}_{name}"
        mname = f"{prefix}_{suffix}"
        matrices[mname] = MatrixData.from_coeff(mname, block)
        terms.append(Term(source=source, matrix=mname))
    const_name = None
    if np.any(mapped.const):
        const_name = f"{prefix}_c"
        matrices[const_name] = MatrixData(name=const_name, array=mapped.const)
    return LinearOp(out_dim=mapped.rows, terms=tuple(terms), const=const_name)


def _relu_rows(expr, mapped: _Mapped, relu_terms: dict, mapper: _Mapper, index: int):
    """Split ``expr``'s rows into linear rows and single-ReLU rows.

    A row holding a ReLU term must hold nothing else: ``a * relu(z_j)`` is
    lowered to ``relu(|a| z_j)``, exact under every norm since ``a`` only
    flips the sign. A row mixing a ReLU with other terms has no such form.
    """
    rows = mapped.rows
    count = np.zeros(rows, dtype=int)
    for block in relu_terms.values():
        block.eliminate_zeros()
        count += np.diff(block.indptr)
    relu_rows = np.flatnonzero(count)
    if relu_rows.size == 0:
        return np.arange(rows), None
    if np.any(count[relu_rows] > 1):
        raise ObjectiveExtractionError(
            f"objective expression {index}: a row sums several ReLU terms; each row may "
            "hold at most one (e.g. max(Ax - u, 0)). Put the terms in separate rows or "
            "separate objective expressions."
        )
    other = np.zeros(rows, dtype=int)
    for block in mapped.blocks.values():
        other += np.diff(block.indptr)
    if np.any(other[relu_rows] > 0) or np.any(mapped.const[relu_rows] != 0.0):
        raise ObjectiveExtractionError(
            f"objective expression {index}: a row adds a ReLU term to other terms. A row "
            "holding a ReLU (e.g. max(Ax - u, 0)) must hold only that term; put the "
            "linear part in its own row or objective expression."
        )

    args = []
    for r in relu_rows:
        for leaf, block in relu_terms.items():
            row = block.getrow(r)
            if row.nnz:
                j, a = int(row.indices[0]), float(row.data[0])
                step = mapper.relu_steps[leaf]
                source, _, _ = _step_source_activation(step)
                args.append(abs(a) * _row_slice_expr(source, [j]))
                break
    arg_expr = stack(args) if len(args) > 1 else args[0]
    linear_rows = np.setdiff1d(np.arange(rows), relu_rows)
    return linear_rows, arg_expr


def _slice_rows(mapped: _Mapped, rows: np.ndarray) -> _Mapped:
    return _Mapped(
        rows=int(rows.size),
        blocks={s: b[rows] for s, b in mapped.blocks.items()},
        const=mapped.const[rows],
    )


def _lower(exprs, mapper: _Mapper, matrices: dict) -> list[ObjectiveBlock]:
    blocks = []
    staged: dict[str, MatrixData] = {}
    for i, expr in enumerate(exprs):
        mapped, relu_terms = mapper.map(expr, allow_relu=True)
        linear_rows, arg_expr = _relu_rows(expr, mapped, relu_terms, mapper, i)
        if linear_rows.size:
            prefix = f"obj{i}"
            op = _register(prefix, _slice_rows(mapped, linear_rows), staged)
            blocks.append(
                ObjectiveBlock(prefix, op.out_dim, op, ActivationOp(ActivationKind.IDENTITY))
            )
        if arg_expr is not None:
            arg_mapped, _ = mapper.map(arg_expr, allow_relu=False)
            prefix = f"obj{i}r"
            op = _register(prefix, arg_mapped, staged)
            blocks.append(ObjectiveBlock(prefix, op.out_dim, op, ActivationOp(ActivationKind.RELU)))
    # Only once every expression mapped: a failed frame must leave no matrices behind.
    matrices.update(staged)
    return blocks


def _relu_steps_outside(verifier, loop_steps) -> dict:
    """ReLU steps recorded outside the loop, keyed by the leaf they produce."""
    inside = {id(step) for step in loop_steps}
    out = {}
    for step in verifier.steps:
        if id(step) in inside:
            continue
        try:
            _, activation, _ = _step_source_activation(step)
            leaf = fresh_leaf_of(step)
        except NonStationaryFOMError:
            continue  # not a step an objective row can lower to
        if activation.kind == ActivationKind.RELU:
            out[leaf] = step
    return out


def extract_objective(
    verifier,
    state_layout,
    next_state_exprs,
    param_map,
    tmp_map,
    matrices: dict,
    iterates=None,
) -> ObjectiveSpec:
    """Map the objective set on ``verifier`` onto the loop extracted by
    :func:`~fomtools.codegen.extract.extract_ir`.

    Parameters
    ----------
    verifier:
        The recording, with its objective set.
    state_layout, next_state_exprs, param_map, tmp_map:
        :func:`~fomtools.codegen.extract.extract_ir`'s view of the iteration:
        ``(leaf, slot, dim, offset)`` per state leaf, the iterate after one
        iteration, and the parameter / temporary leaf maps.
    matrices:
        The IR's matrix table; the objective's coefficient blocks are added to it.
    iterates:
        The recorded iterate sequence (one entry per iteration, normalized), if
        any. Enables the last-recorded-iteration frame.

    Raises
    ------
    ObjectiveExtractionError
        If no objective is set, or it reads something the loop does not carry.
    """
    if verifier.objective is None:
        raise ObjectiveExtractionError(
            "termination kind 'objective' stops on the verification objective, but none "
            "is set on this recording. Call verifier.set_objective_norm(expr, ord) (e.g. "
            "the fixed-point residual x_K - x_{K-1}, ord=np.inf) before generating code, "
            "or use TerminationKind.INF_NORM_CHANGE / MAX_ITERS."
        )
    exprs, ord = verifier.objective
    norm = norm_label(ord)
    state_dim = sum(dim for (_, _, dim, _) in state_layout)

    groups = [list(g) for g in iterates] if iterates else []
    loop_steps = _dependency_cone(
        verifier.steps,
        {leaf for g in [next_state_exprs, *groups] for e in g for leaf in e.decomposition_dict},
    )
    relu_steps = _relu_steps_outside(verifier, loop_steps)
    # Parameters are the leaves no step produces and no iterate declares, as in
    # extraction's own classification.
    produced = {leaf for step in verifier.steps for leaf in step.output_leaves}
    iterates_declared = set(verifier.iterates)
    outside_params = {
        leaf
        for expr in exprs
        for leaf in expr.decomposition_dict
        if leaf not in produced and leaf not in iterates_declared and leaf not in param_map
    }

    state_rows = {leaf: np.arange(off, off + dim) for (leaf, _, dim, off) in state_layout}
    frames = [
        _Frame(
            label="the extracted iteration (w^0 = the state leaves, w^1 = next_state)",
            state_dim=state_dim,
            new_rows=_copy_rows(next_state_exprs),
            prev_rows=state_rows,
            tmps=dict(tmp_map),
        )
    ]
    if len(groups) >= 2:
        K = len(groups)
        frames.append(
            _Frame(
                label=f"the last recorded iteration (w^{K - 1} and w^{K} from the tape)",
                state_dim=state_dim,
                new_rows=_copy_rows(groups[-1]),
                prev_rows=_copy_rows(groups[-2]),
            )
        )

    failures = []
    for frame in frames:
        try:
            mapper = _Mapper(frame, param_map, relu_steps, outside_params)
            blocks = _lower(exprs, mapper, matrices)
        except ObjectiveExtractionError as err:
            failures.append(str(err))
            continue
        return ObjectiveSpec(blocks=tuple(blocks), norm=norm)
    hint = (
        ""
        if len(frames) > 1
        else " To set the objective at the last recorded iteration instead, also pass "
        "iterates= (the recorded iterate sequence) so that iteration can be located."
    )
    raise ObjectiveExtractionError(
        "the verification objective cannot be evaluated by the generated loop.\n- "
        + "\n- ".join(failures)
        + hint
    )
