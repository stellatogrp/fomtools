"""Tests for merging content-identical coefficient blocks."""

from dataclasses import replace

import numpy as np
import pytest

from fomtools.codegen.config import CodegenConfig, TerminationSpec
from fomtools.codegen.dedup import dedup_matrices
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import (
    ActivationKind,
    ActivationOp,
    FomIR,
    IterationStep,
    LinearOp,
    MatrixData,
    StateSpec,
    Term,
)
from fomtools.codegen.oracle import ReferenceOracle, run_recorded_tape
from fomtools.verifier import Verifier

ONE_PASS = CodegenConfig(termination=TerminationSpec(max_iters=1))


def _ir(matrices, body, state_update, state_dim=2):
    return FomIR(
        params=(),
        state=StateSpec(dim=state_dim, slots=(("s0", state_dim),), init=np.zeros(state_dim)),
        body=tuple(body),
        state_update=state_update,
        termination=TerminationSpec(max_iters=1),
        matrices=matrices,
    )


def _block(name, array, pattern=None):
    return MatrixData(name=name, array=np.asarray(array, dtype=float), sparse_pattern=pattern)


def test_identical_blocks_merge_and_references_repoint():
    same = [[1.0, 2.0], [3.0, 4.0]]
    matrices = {
        "a": _block("a", same),
        "b": _block("b", same),
        "c": _block("c", [[9.0, 0.0], [0.0, 9.0]]),
    }
    body = [
        IterationStep(
            name="t0",
            out_dim=2,
            linear_op=LinearOp(out_dim=2, terms=(Term(source="state", matrix="b"),), const="a"),
            activation=ActivationOp(ActivationKind.RELU),
        )
    ]
    state_update = LinearOp(out_dim=2, terms=(Term(source="tmp:t0", matrix="c"),))

    out = dedup_matrices(_ir(matrices, body, state_update))

    assert set(out.matrices) == {"a", "c"}, "the earliest name survives"
    assert out.body[0].linear_op.terms[0].matrix == "a"
    assert out.body[0].linear_op.const == "a"
    assert out.state_update.terms[0].matrix == "c"


def test_distinct_blocks_are_returned_unchanged():
    matrices = {
        "a": _block("a", [[1.0, 0.0], [0.0, 1.0]]),
        "b": _block("b", [[2.0, 0.0], [0.0, 2.0]]),
    }
    body = [
        IterationStep(
            name="t0",
            out_dim=2,
            linear_op=LinearOp(out_dim=2, terms=(Term(source="state", matrix="a"),)),
            activation=ActivationOp(ActivationKind.IDENTITY),
        )
    ]
    ir = _ir(matrices, body, LinearOp(out_dim=2, terms=(Term(source="tmp:t0", matrix="b"),)))
    assert dedup_matrices(ir) is ir


def test_saturated_linear_bounds_repoint():
    bound = [0.0, 0.0]
    matrices = {"lo": _block("lo", bound), "hi": _block("hi", bound)}
    body = [
        IterationStep(
            name="t0",
            out_dim=2,
            linear_op=LinearOp(out_dim=2, terms=()),
            activation=ActivationOp(ActivationKind.SATURATED_LINEAR, lower="lo", upper="hi"),
        )
    ]
    out = dedup_matrices(_ir(matrices, body, LinearOp(out_dim=2, terms=())))
    assert set(out.matrices) == {"lo"}
    assert out.body[0].activation.lower == "lo"
    assert out.body[0].activation.upper == "lo"


def test_survivor_inherits_a_sparsity_pattern_from_a_merged_block():
    """The pattern selects sparse matvec emission, so it must not be dropped."""
    same = [[1.0, 0.0], [0.0, 1.0]]
    pattern = (np.array([0, 1]), np.array([0, 1]))
    matrices = {"a": _block("a", same), "b": _block("b", same, pattern)}
    body = [
        IterationStep(
            name="t0",
            out_dim=2,
            linear_op=LinearOp(
                out_dim=2,
                terms=(Term(source="state", matrix="a"), Term(source="state", matrix="b")),
            ),
            activation=ActivationOp(ActivationKind.IDENTITY),
        )
    ]
    out = dedup_matrices(_ir(matrices, body, LinearOp(out_dim=2, terms=())))

    assert set(out.matrices) == {"a"}
    assert out.matrices["a"].sparse_pattern is not None


def test_scaled_terms_keep_their_own_schedule():
    """Merging a block must not merge the schedules that scale it."""
    same = [[1.0, 0.0], [0.0, 1.0]]
    matrices = {"a": _block("a", same), "b": _block("b", same)}
    body = [
        IterationStep(
            name="t0",
            out_dim=2,
            linear_op=LinearOp(
                out_dim=2,
                terms=(
                    Term(source="state", matrix="a"),
                    Term(source="state", matrix="b", scale="gamma"),
                ),
            ),
            activation=ActivationOp(ActivationKind.IDENTITY),
        )
    ]
    ir = replace(
        _ir(matrices, body, LinearOp(out_dim=2, terms=())),
        schedules={"gamma": np.linspace(0.0, 1.0, 4)},
    )
    out = dedup_matrices(ir)

    assert set(out.matrices) == {"a"}
    assert [(t.matrix, t.scale) for t in out.body[0].linear_op.terms] == [
        ("a", None),
        ("a", "gamma"),
    ]


@pytest.mark.parametrize("steps", [10, 50])
def test_unrolled_stationary_ista_collapses_to_three_blocks(steps):
    """A stationary loop unrolled emits one block set per iteration; all agree."""
    rng = np.random.default_rng(0)
    m, n = 6, 10
    dmat = rng.normal(size=(m, n)) / np.sqrt(m)
    eta = 1.0 / np.max(np.linalg.eigvalsh(dmat.T @ dmat))
    at, bt = np.eye(n) - eta * dmat.T @ dmat, eta * dmat.T

    v = Verifier(method="simulator", obbt_solver=None)
    x = v.parameter(m, lb=-np.ones(m), ub=np.ones(m))
    z = v.iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    iterates = []
    for _ in range(steps):
        z = v.soft_threshold(at @ z + bt @ x, lambd=0.05 * eta)
        iterates.append(z)

    ir = extract_ir(v, iterates[-1], config=ONE_PASS)
    assert len(ir.matrices) == 2 * steps + 1

    deduped = dedup_matrices(ir)
    assert len(deduped.matrices) == 3

    # Purely a naming change: the numbers must be identical.
    xval = rng.uniform(-1.0, 1.0, size=m)
    truth = run_recorded_tape(v, iterates, {x: xval})
    before = ReferenceOracle(ir).solve(param_values={"p0": xval}, x0=truth[0]).solution
    after = ReferenceOracle(deduped).solve(param_values={"p0": xval}, x0=truth[0]).solution
    assert np.array_equal(before, after)
    assert np.allclose(after, truth[-1], atol=1e-12)
