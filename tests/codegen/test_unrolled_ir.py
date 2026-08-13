"""Tests for extracting a whole recorded run as one unrolled body.

These recordings are written the way a user writes them to *verify*: a plain
Python loop over ``verifier.step.*``, with no restructuring for code generation.
"""

from dataclasses import replace

import numpy as np
import pytest

from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec
from fomtools.codegen.extract import normalize_iterates
from fomtools.codegen.extract_unrolled import extract_unrolled_ir
from fomtools.codegen.ir import MatrixData, UnrolledBlowupError, UnrolledFidelityError
from fomtools.codegen.oracle import (
    ReferenceOracle,
    check_unrolled,
    run_recorded_tape,
    sample_param_values,
)
from tests.codegen.conftest import (
    build_dr_loop,
    build_fista_loop,
    build_halpern_loop,
    build_ista_loop,
    build_mixed_step_loop,
)


def _assert_matches_tape(v, iterates, ir, param_leaf, atol=1e-12):
    """The unrolled body's final state must equal the tape's final iterate."""
    groups = normalize_iterates(v, iterates)
    pvals = sample_param_values(v, groups)
    truth = run_recorded_tape(v, groups, pvals)
    got = ReferenceOracle(ir).solve(param_values={"p0": pvals[param_leaf]}, x0=truth[0]).solution
    assert np.allclose(got, truth[-1], atol=atol)
    return got


# --- the three recordings no existing path accepts --------------------------


def test_manual_fista_loop_reproduces_the_tape():
    v, iterates, x, d = build_fista_loop()
    ir = extract_unrolled_ir(v, iterates, name="fista")

    assert ir.unrolled is True
    assert len(ir.body) == len(iterates)
    assert ir.state_dim == d["n"]
    _assert_matches_tape(v, iterates, ir, x)


def test_halpern_anchor_resolves_as_a_state_slot():
    """The anchor is an ``add_iterate`` leaf nothing updates, so it *is* the state."""
    v, iterates, x, d = build_halpern_loop()
    ir = extract_unrolled_ir(v, iterates, name="halpern")

    assert len(ir.state.slots) == 1
    assert ir.state_dim == d["n"]
    # Every iteration reads the anchor, each with its own coefficient block.
    anchored = [s for s in ir.body if any(t.source == "state" for t in s.linear_op.terms)]
    assert len(anchored) == len(iterates)
    _assert_matches_tape(v, iterates, ir, x)


def test_step_type_may_change_with_the_iteration():
    from fomtools.codegen.ir import ActivationKind

    v, iterates, x, _ = build_mixed_step_loop()
    ir = extract_unrolled_ir(v, iterates, name="mixed")

    kinds = [s.activation.kind for s in ir.body]
    assert set(kinds) == {ActivationKind.RELU, ActivationKind.SOFT_THRESHOLD}
    _assert_matches_tape(v, iterates, ir, x)


# --- rolling back up --------------------------------------------------------


def test_a_stationary_loop_rolls_back_up():
    """Writing a stationary method as a loop must cost nothing."""
    v, iterates, x, _ = build_ista_loop(steps=40)
    ir = extract_unrolled_ir(v, iterates, name="ista")

    assert ir.unrolled is False
    assert len(ir.body) == 1
    assert len(ir.matrices) == 3


def test_rolling_preserves_the_recorded_horizon():
    """Rolling changes the artifact, never the computation -- including how long."""
    v, iterates, _, _ = build_ista_loop(steps=12)
    ir = extract_unrolled_ir(
        v, iterates, config=CodegenConfig(termination=TerminationSpec(max_iters=999))
    )
    assert ir.unrolled is False
    assert ir.termination.max_iters == 12, "the recording fixes the horizon, not the config"


@pytest.mark.parametrize("builder", [build_fista_loop, build_halpern_loop, build_mixed_step_loop])
def test_a_varying_loop_does_not_roll(builder):
    v, iterates, x, _ = builder(steps=10)
    ir = extract_unrolled_ir(v, iterates)

    assert ir.unrolled is True
    assert len(ir.body) == 10
    _assert_matches_tape(v, iterates, ir, x)


def test_a_stationary_accumulating_loop_rolls_rather_than_blowing_up():
    """Rolling is attempted first, so quadratic growth never gets built."""
    v, iterates, _ = build_dr_loop(steps=80, collapse_expr=True)
    ir = extract_unrolled_ir(v, iterates, name="dr")

    assert ir.unrolled is False
    assert len(ir.body) == 2
    assert ir.termination.max_iters == 80


def test_roll_can_be_disabled():
    v, iterates, x, _ = build_ista_loop(steps=8)
    ir = extract_unrolled_ir(v, iterates, roll=False)

    assert ir.unrolled is True
    assert len(ir.body) == 8
    _assert_matches_tape(v, iterates, ir, x)


# --- shape of the result ----------------------------------------------------


def test_unrolled_body_is_a_single_pass():
    v, iterates, _, _ = build_ista_loop(steps=5)
    ir = extract_unrolled_ir(v, iterates, roll=False)
    assert ir.termination.kind == TerminationKind.MAX_ITERS
    assert ir.termination.max_iters == 1


def test_configured_max_iters_is_overridden_by_the_recorded_length():
    """The recording fixes the horizon; a configured loop count is meaningless."""
    v, iterates, _, _ = build_ista_loop(steps=5)
    ir = extract_unrolled_ir(
        v,
        iterates,
        roll=False,
        config=CodegenConfig(termination=TerminationSpec(max_iters=999)),
    )
    assert ir.termination.max_iters == 1
    assert len(ir.body) == 5


@pytest.mark.parametrize("roll", [True, False])
def test_tolerance_based_termination_is_rejected(roll):
    """Rejected up front, so the roll path cannot slip past the check."""
    v, iterates, _, _ = build_ista_loop(steps=5)
    config = CodegenConfig(
        termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=50, tol=1e-6)
    )
    with pytest.raises(ValueError, match="recorded tape"):
        extract_unrolled_ir(v, iterates, config=config, roll=roll)


# --- iterate sequence conventions ------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"trajectory": True}, {"include_initial": True}, {"include_initial": False}]
)
def test_a_leading_initial_iterate_is_recognized_and_dropped(kwargs):
    """Slot 0 of a trajectory is the initial iterate; hand-built lists go both ways."""
    steps = 6
    v, iterates, x, _ = build_ista_loop(steps=steps, **kwargs)
    ir = extract_unrolled_ir(v, iterates, roll=False)
    assert len(ir.body) == steps
    _assert_matches_tape(v, iterates, ir, x)


def test_empty_iterates_raises():
    v, _, _, _ = build_ista_loop(steps=3)
    with pytest.raises(ValueError, match="at least one recorded iteration"):
        extract_unrolled_ir(v, [])


# --- size ------------------------------------------------------------------


def test_dedup_collapses_a_stationary_unrolled_loop():
    v, iterates, _, _ = build_ista_loop(steps=40)
    plain = extract_unrolled_ir(v, iterates, roll=False, dedup=False)
    merged = extract_unrolled_ir(v, iterates, roll=False, dedup=True)

    assert len(plain.matrices) == 2 * 40 + 1
    assert len(merged.matrices) == 3


def test_repeated_solves_share_one_factorization():
    v, iterates, _ = build_dr_loop(steps=20, collapse_expr=False)
    ir = extract_unrolled_ir(v, iterates, name="dr", roll=False)
    assert len(ir.body) == 60, "three steps per iteration, unrolled"
    assert len(ir.factorizations) == 1


# --- accumulation -----------------------------------------------------------


def _varying_relaxation(k):
    return 1.0 + 0.05 * k


def test_a_varying_accumulating_loop_is_rejected_with_an_actionable_message():
    """The realistic path: it cannot roll, and collapsed it cannot be unrolled either."""
    v, iterates, _ = build_dr_loop(steps=20, collapse_expr=True, relaxation=_varying_relaxation)
    with pytest.raises(UnrolledBlowupError) as excinfo:
        extract_unrolled_ir(v, iterates, name="dr")

    message = str(excinfo.value)
    assert "square of the iteration count" in message
    assert "iterate_trajectory" in message


def test_a_trajectory_handles_accumulation_with_no_user_action():
    """The default 'auto' policy latches, so this needs no collapse_expr at all."""
    v, iterates, q = build_dr_loop(steps=20, collapse_expr="auto", relaxation=_varying_relaxation)
    ir = extract_unrolled_ir(v, iterates, name="dr")

    assert ir.unrolled is True
    terms = sum(len(s.linear_op.terms) for s in ir.body) + len(ir.state_update.terms)
    assert terms < 200
    _assert_matches_tape(v, iterates, ir, q, atol=1e-11)


def test_the_same_varying_loop_extracts_linearly_when_materialized():
    v, iterates, q = build_dr_loop(steps=20, collapse_expr=False, relaxation=_varying_relaxation)
    ir = extract_unrolled_ir(v, iterates, name="dr")

    assert ir.unrolled is True, "a varying relaxation cannot be rolled"
    terms = sum(len(s.linear_op.terms) for s in ir.body) + len(ir.state_update.terms)
    assert terms < 200, "materializing each iterate must keep the term count linear"
    assert len(ir.state_update.terms) == 1, "the answer is the last materialized iterate"
    _assert_matches_tape(v, iterates, ir, q, atol=1e-11)


def test_blowup_is_detected_independently_of_rolling():
    v, iterates, _ = build_dr_loop(steps=20, collapse_expr=True)
    with pytest.raises(UnrolledBlowupError):
        extract_unrolled_ir(v, iterates, name="dr", roll=False)


def test_a_short_accumulating_loop_is_not_rejected():
    """Growth only matters once the artifact is big enough to care about."""
    v, iterates, _ = build_dr_loop(steps=4, collapse_expr=True)
    ir = extract_unrolled_ir(v, iterates, name="dr", roll=False)
    assert ir.unrolled is True


# --- the gate ---------------------------------------------------------------


def test_the_gate_catches_a_corrupted_body():
    """Proves the check is not vacuous."""
    v, iterates, _, d = build_ista_loop(steps=6)
    ir = extract_unrolled_ir(v, iterates, roll=False)

    victim = ir.body[3].linear_op.terms[0].matrix
    broken = replace(
        ir,
        matrices={
            **ir.matrices,
            victim: MatrixData(name=victim, array=np.zeros_like(ir.matrices[victim].array)),
        },
    )
    with pytest.raises(UnrolledFidelityError, match="does not reproduce the recorded tape"):
        check_unrolled(broken, v, list(iterates))


def test_the_gate_accepts_a_trajectory_directly():
    """A trajectory keeps its initial iterate in slot 0; the gate must cope."""
    v, traj, _, _ = build_ista_loop(steps=6, trajectory=True)
    ir = extract_unrolled_ir(v, traj, check=False, roll=False)
    check_unrolled(ir, v, traj)


def test_check_can_be_skipped():
    v, iterates, _, _ = build_ista_loop(steps=6)
    ir = extract_unrolled_ir(v, iterates, check=False, roll=False)
    assert ir.unrolled is True


def test_check_false_does_not_disable_the_roll_decision():
    """``check=False`` applies to the unrolled result only.

    On the roll path the check *is* the decision, so it cannot be skipped without
    changing the answer. Observable both ways: a stationary recording still rolls,
    and a varying one still refuses to.
    """
    v, iterates, _, _ = build_ista_loop(steps=10)
    assert extract_unrolled_ir(v, iterates, check=False).unrolled is False

    v, iterates, _, _ = build_halpern_loop(steps=10)
    assert extract_unrolled_ir(v, iterates, check=False).unrolled is True


def test_sampled_param_values_are_deterministic_and_inside_the_box():
    v, iterates, x, d = build_ista_loop(steps=4)
    first = sample_param_values(v, list(iterates))
    second = sample_param_values(v, list(iterates))

    assert np.array_equal(first[x], second[x])
    assert first[x].shape == (d["m"],)
    assert np.all(first[x] >= -1.0) and np.all(first[x] <= 1.0)
    # Not the box midpoint, which would be zero and make soft-thresholding trivial.
    assert np.any(np.abs(first[x]) > 1e-3)
