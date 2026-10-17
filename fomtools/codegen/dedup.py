"""Merge coefficient blocks that hold identical numbers.

An unrolled recording gives every iteration its own coefficient blocks, because
nothing in the extraction assumes the iterations agree. When they *do* agree the
same array is emitted once per iteration, which dominates the artifact: a
stationary ISTA loop unrolled to 200 iterations emits 401 blocks holding 3
distinct arrays, and a Douglas-Rachford loop that accumulates through a non-leaf
expression emits 13121 blocks holding 10.

This pass keeps one block per distinct array and repoints every reference at it,
so the emitted data is proportional to the *distinct* coefficients rather than to
the iteration count. It is purely a naming change: no array is altered, so the
generated solver computes exactly the same numbers.

Blocks are bucketed by a content hash and then compared exactly, so a hash
collision cannot merge two different arrays. The earliest name in insertion order
survives, which keeps emitted symbol names stable and deterministic; if a later
member of a group carries a sparsity pattern the survivor lacks, that pattern
moves to the survivor, since the arrays are equal and the pattern is what selects
sparse matrix-vector emission.

Applied on the *unrolled* path only. Merging a block removes its
``fom_apply_<matrix>`` helper from the emitted ops, and a rolled recording may
legitimately hold duplicate blocks that a hand-written solver body already calls
by name: of the fixtures in ``tests/codegen/conftest.py``, both the
indefinite-KKT and the CG-QP IRs hold three blocks that collapse to one, so
enabling this for rolled emission would stop their goldens compiling. Nothing
here is unsafe for a rolled IR mathematically -- the constraint is that the
solver body is written against the emitted names.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from fomtools.codegen.ir import ActivationOp, FomIR, LinearOp, MatrixData


def _canonical(md: MatrixData) -> tuple:
    """The values of a block in one canonical form, dense or sparse alike:
    ``(shape, rowptr, colidx, vals)`` of its exact nonzeros."""
    if md.array.ndim < 2:
        arr = np.ascontiguousarray(md.array, dtype=np.float64)
        return (arr.shape, arr)
    rowptr, colidx, vals = md.csr(tol=0.0)
    return (tuple(md.shape), rowptr, colidx, vals)


def _content_key(md: MatrixData) -> tuple:
    """Hash bucket for a block: shape plus a digest of its canonical values."""
    shape, *parts = _canonical(md)
    digest = hashlib.sha1()
    for part in parts:
        digest.update(np.ascontiguousarray(part).tobytes())
    return (shape, digest.hexdigest())


def _same_values(a: MatrixData, b: MatrixData) -> bool:
    ca, cb = _canonical(a), _canonical(b)
    return ca[0] == cb[0] and all(np.array_equal(x, y) for x, y in zip(ca[1:], cb[1:]))


def _rewrite_linop(lin: LinearOp, alias: dict[str, str]) -> LinearOp:
    return replace(
        lin,
        terms=tuple(replace(t, matrix=alias[t.matrix]) for t in lin.terms),
        const=None if lin.const is None else alias[lin.const],
    )


def _rewrite_activation(act: ActivationOp, alias: dict[str, str]) -> ActivationOp:
    if act.lower is None and act.upper is None:
        return act
    return replace(
        act,
        lower=None if act.lower is None else alias[act.lower],
        upper=None if act.upper is None else alias[act.upper],
    )


def dedup_matrices(ir: FomIR) -> FomIR:
    """Return ``ir`` with content-identical coefficient blocks merged.

    Returns the input unchanged when every block is already distinct, so this is
    safe to call unconditionally.
    """
    buckets: dict[tuple, list[str]] = {}
    kept: dict[str, MatrixData] = {}
    alias: dict[str, str] = {}

    for name, md in ir.matrices.items():
        bucket = buckets.setdefault(_content_key(md), [])
        for other in bucket:
            # Exact comparison: the digest only buckets, it never decides.
            if _same_values(kept[other], md):
                alias[name] = other
                if kept[other].sparse_pattern is None and md.sparse_pattern is not None:
                    kept[other] = replace(kept[other], sparse_pattern=md.sparse_pattern)
                break
        else:
            alias[name] = name
            kept[name] = md
            bucket.append(name)

    if len(kept) == len(ir.matrices):
        return ir

    return replace(
        ir,
        matrices=kept,
        body=tuple(
            replace(
                step,
                linear_op=_rewrite_linop(step.linear_op, alias),
                activation=_rewrite_activation(step.activation, alias),
            )
            for step in ir.body
        ),
        state_update=_rewrite_linop(ir.state_update, alias),
        objective=None
        if ir.objective is None
        else replace(
            ir.objective,
            blocks=tuple(
                replace(block, linear_op=_rewrite_linop(block.linear_op, alias))
                for block in ir.objective.blocks
            ),
        ),
    )
