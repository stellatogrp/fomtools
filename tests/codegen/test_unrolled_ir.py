"""Tests for extracting a whole recorded run as one unrolled body.

These recordings are written the way a user writes them to *verify*: a plain
Python loop over ``verifier.step.*``, with no restructuring for code generation.
"""

from dataclasses import replace

import numpy as np
import pytest

from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec
from fomtools.codegen.extract import normalize_iterates
from fomtools.codegen.ir import MatrixData, UnrolledBlowupError, UnrolledFidelityError
from fomtools.codegen.oracle import (
    ReferenceOracle,
    check_unrolled,
    run_recorded_tape,
    sample_param_values,
)
from fomtools.codegen.unrolled import extract_unrolled_ir
from fomtools.verifier import Verifier


def _lasso_data(m=6, n=10, seed=0):
    rng = np.random.default_rng(seed)
    dmat = rng.normal(size=(m, n)) / np.sqrt(m)
    eta = 1.0 / np.max(np.linalg.eigvalsh(dmat.T @ dmat))
    return {
        "m": m,
        "n": n,
        "at": np.eye(n) - eta * dmat.T @ dmat,
        "bt": eta * dmat.T,
        "lambd": 0.05 * eta,
    }


def build_ista_loop(steps=8, trajectory=False, include_initial=False):
    """A stationary loop, written as a loop rather than as one iteration."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    z0 = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    if trajectory:
        traj = v.iterate_trajectory(steps + 1)
        traj[0] = z0
        for k in range(1, steps + 1):
            traj[k] = v.step.soft_threshold(d["at"] @ traj[k - 1] + d["bt"] @ x, lambd=d["lambd"])
        return v, traj, x, d

    z, iterates = z0, [z0] if include_initial else []
    for _ in range(steps):
        z = v.step.soft_threshold(d["at"] @ z + d["bt"] @ x, lambd=d["lambd"])
        iterates.append(z)
    return v, iterates, x, d


def build_fista_loop(steps=8):
    """FISTA written the natural way: one iterate leaf, momentum in Python."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    s = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    y_prev, extrap, beta, iterates = s, s, 1.0, []
    for _ in range(steps):
        y = v.step.soft_threshold(d["at"] @ extrap + d["bt"] @ x, lambd=d["lambd"])
        beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
        extrap = y + ((beta - 1.0) / beta_next) * (y - y_prev)
        beta, y_prev = beta_next, y
        iterates.append(y)
    return v, iterates, x, d


def build_halpern_loop(steps=8):
    """An anchored method: every iteration refers back to ``z_0``."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    z0 = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    z, iterates = z0, []
    for k in range(steps):
        alpha = 1.0 / (k + 2.0)
        tz = v.step.soft_threshold(d["at"] @ z + d["bt"] @ x, lambd=d["lambd"])
        z = alpha * z0 + (1.0 - alpha) * tz
        iterates.append(z)
    return v, iterates, x, d


def build_mixed_step_loop(steps=8):
    """The step *type* changes with the iteration index."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    z = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    iterates = []
    for k in range(steps):
        arg = d["at"] @ z + d["bt"] @ x
        z = v.step.relu(arg) if k % 2 else v.step.soft_threshold(arg, lambd=0.01 * (k + 1))
        iterates.append(z)
    return v, iterates, x, d


def build_dr_loop(steps=20, collapse_expr="auto"):
    """Douglas-Rachford: the iterate is carried through a non-leaf accumulation."""
    rng = np.random.default_rng(0)
    n, m = 8, 4
    amat = rng.normal(size=(m, n))
    kkt = np.block([[np.eye(n), amat.T], [amat, -np.eye(m)]])
    dim = n + m

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(dim, lb=-np.ones(dim), ub=np.ones(dim))
    z = v.iterate_trajectory(steps + 1, collapse_expr=collapse_expr)
    z[0] = v.add_iterate(dim, lb=np.zeros(dim), ub=np.zeros(dim))
    for k in range(1, steps + 1):
        u = v.step.implicit_linear(z[k - 1] - q, output_mat=kkt)
        ut = v.step.relu(2 * u - z[k - 1], proj_ranges=(n, dim))
        z[k] = z[k - 1] + ut - u
    return v, z, q


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


# --- shape of the result ----------------------------------------------------


def test_body_is_a_single_pass():
    v, iterates, _, _ = build_ista_loop(steps=5)
    ir = extract_unrolled_ir(v, iterates)
    assert ir.termination.kind == TerminationKind.MAX_ITERS
    assert ir.termination.max_iters == 1


def test_configured_max_iters_is_overridden_by_the_recorded_length():
    """The recording fixes the horizon; a configured loop count is meaningless."""
    v, iterates, _, _ = build_ista_loop(steps=5)
    ir = extract_unrolled_ir(
        v, iterates, config=CodegenConfig(termination=TerminationSpec(max_iters=999))
    )
    assert ir.termination.max_iters == 1
    assert len(ir.body) == 5


def test_tolerance_based_termination_is_rejected():
    v, iterates, _, _ = build_ista_loop(steps=5)
    config = CodegenConfig(
        termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=50, tol=1e-6)
    )
    with pytest.raises(ValueError, match="whole run executed once"):
        extract_unrolled_ir(v, iterates, config=config)


# --- iterate sequence conventions ------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"trajectory": True}, {"include_initial": True}, {"include_initial": False}]
)
def test_a_leading_initial_iterate_is_recognized_and_dropped(kwargs):
    """Slot 0 of a trajectory is the initial iterate; hand-built lists go both ways."""
    steps = 6
    v, iterates, x, _ = build_ista_loop(steps=steps, **kwargs)
    ir = extract_unrolled_ir(v, iterates)
    assert len(ir.body) == steps
    _assert_matches_tape(v, iterates, ir, x)


def test_empty_iterates_raises():
    v, _, _, _ = build_ista_loop(steps=3)
    with pytest.raises(ValueError, match="at least one recorded iteration"):
        extract_unrolled_ir(v, [])


# --- size ------------------------------------------------------------------


def test_dedup_collapses_a_stationary_unrolled_loop():
    v, iterates, _, _ = build_ista_loop(steps=40)
    plain = extract_unrolled_ir(v, iterates, dedup=False)
    merged = extract_unrolled_ir(v, iterates, dedup=True)

    assert len(plain.matrices) == 2 * 40 + 1
    assert len(merged.matrices) == 3


def test_repeated_solves_share_one_factorization():
    v, iterates, _ = build_dr_loop(steps=20, collapse_expr=False)
    ir = extract_unrolled_ir(v, iterates, name="dr")
    assert len(ir.factorizations) == 1


# --- accumulation -----------------------------------------------------------


def test_accumulating_loop_is_rejected_with_an_actionable_message():
    v, iterates, _ = build_dr_loop(steps=20, collapse_expr=True)
    with pytest.raises(UnrolledBlowupError) as excinfo:
        extract_unrolled_ir(v, iterates, name="dr")

    message = str(excinfo.value)
    assert "square of the iteration count" in message
    assert "iterate_trajectory" in message


def test_materialized_accumulating_loop_extracts_linearly():
    v, iterates, q = build_dr_loop(steps=20, collapse_expr=False)
    ir = extract_unrolled_ir(v, iterates, name="dr")

    terms = sum(len(s.linear_op.terms) for s in ir.body) + len(ir.state_update.terms)
    assert terms < 200, "materializing each iterate must keep the term count linear"
    assert len(ir.state_update.terms) == 1, "the answer is the last materialized iterate"
    _assert_matches_tape(v, iterates, ir, q, atol=1e-11)


def test_a_short_accumulating_loop_is_not_rejected():
    """Growth only matters once the artifact is big enough to care about."""
    v, iterates, _ = build_dr_loop(steps=4, collapse_expr=True)
    ir = extract_unrolled_ir(v, iterates, name="dr")
    assert ir.unrolled is True


# --- the gate ---------------------------------------------------------------


def test_the_gate_catches_a_corrupted_body():
    """Proves the check is not vacuous."""
    v, iterates, _, d = build_ista_loop(steps=6)
    ir = extract_unrolled_ir(v, iterates)

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
    ir = extract_unrolled_ir(v, traj, check=False)
    check_unrolled(ir, v, traj)


def test_check_can_be_skipped():
    v, iterates, _, _ = build_ista_loop(steps=6)
    ir = extract_unrolled_ir(v, iterates, check=False)
    assert ir.unrolled is True


def test_sampled_param_values_are_deterministic_and_inside_the_box():
    v, iterates, x, d = build_ista_loop(steps=4)
    first = sample_param_values(v, list(iterates))
    second = sample_param_values(v, list(iterates))

    assert np.array_equal(first[x], second[x])
    assert first[x].shape == (d["m"],)
    assert np.all(first[x] >= -1.0) and np.all(first[x] <= 1.0)
    # Not the box midpoint, which would be zero and make soft-thresholding trivial.
    assert np.any(np.abs(first[x]) > 1e-3)
