"""Extraction of a FOM whose coefficients vary with the iteration index.

The whole approach rests on one fact: a schedule scalar enters the recorded
``LinearExpression`` linearly, so every extracted coefficient block is affine in
it and can be recovered exactly from a few probes. These tests pin that fact
down, and pin down what happens when it does not hold.
"""

import numpy as np
import pytest

from fomtools.codegen import (
    CodegenConfig,
    ReferenceOracle,
    ScheduleFitError,
    TerminationSpec,
    extract_ir,
    extract_scheduled_ir,
)
from fomtools.codegen.extract_scheduled import Recording
from fomtools.codegen.ir import NonStationaryFOMError
from fomtools.codegen.schedule import Schedule, as_schedule
from fomtools.verifier import Verifier
from tests.codegen.conftest import build_scheduled, fista_reference


def _varying(k: int) -> float:
    """Some coefficient that changes with the iteration; the values do not matter.

    Written here rather than imported: what a coefficient *is* belongs to the
    method being generated, not to the code generator.
    """
    return k / (k + 3.0)


def test_fista_extracts_to_a_single_stationary_body():
    """Over a two-slot state, FISTA is one step: only its coefficient moves."""
    case = build_scheduled(max_iters=50)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])

    assert len(ir.body) == 1
    assert ir.state.dim == 4
    assert [dim for _, dim in ir.state.slots] == [2, 2]
    assert set(ir.schedules) == {"gamma"}
    assert len(ir.schedules["gamma"]) == 50


def test_only_the_momentum_coefficient_is_scheduled():
    """Exactly one block varies, and it is split into a base plus a scaled slope."""
    case = build_scheduled(max_iters=20)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])

    scaled = {t.matrix: t.scale for t in ir.body[0].linear_op.terms if t.scale is not None}
    assert scaled == {"t0_state__d_gamma": "gamma"}
    # The parameter block and the whole state update are iteration-independent.
    assert all(t.scale is None for t in ir.state_update.terms)
    assert all(
        t.scale is None for t in ir.body[0].linear_op.terms if t.matrix != "t0_state__d_gamma"
    )


def test_affine_recovery_is_exact():
    """base + gamma * slope must reproduce a directly extracted IR at any gamma."""
    case = build_scheduled(max_iters=20)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    base = ir.matrices["t0_state"].dense()
    slope = ir.matrices["t0_state__d_gamma"].dense()

    for gamma in (0.05, 0.5, 0.9, 1.7):
        direct = case["build"](gamma=gamma)
        ref = extract_ir(
            direct.verifier,
            direct.next_state,
            state=direct.state,
            config=case["config"],
        )
        assert np.abs(base + gamma * slope - ref.matrices["t0_state"].dense()).max() < 1e-14


def test_first_momentum_value_is_zero_but_the_schedule_is_not_frozen():
    """The degeneracy that made FISTA look like ISTA is present but harmless.

    ``gamma_0 = 0``, so a single recorded iteration carries no momentum at all.
    Tabulating the coefficient rather than reading it off one iteration is
    precisely what keeps that from collapsing the method.
    """
    case = build_scheduled(max_iters=20)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    gamma = ir.schedules["gamma"]

    assert gamma[0] == pytest.approx(0.0)
    assert np.all(np.diff(gamma) > 0)  # increasing towards 1
    assert gamma[-1] > 0.8


def test_scheduled_oracle_matches_numpy_fista():
    """The rolled IR reproduces FISTA at every iteration, and is not ISTA."""
    case = build_scheduled(max_iters=100)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    b_val = case["data"]["b"]

    rolled = ReferenceOracle(ir).solve(param_values={"p0": b_val}).trajectory
    truth = fista_reference(case["data"], b_val, n_iters=100)
    assert max(np.abs(a - b).max() for a, b in zip(rolled, truth)) < 1e-12

    # Both methods converge to the same solution, so they can only be told apart
    # along the way — which is exactly where the silent degradation showed up.
    plain = fista_reference(case["data"], b_val, n_iters=100, momentum=False)
    assert max(np.abs(a - b).max() for a, b in zip(truth, plain)) > 1e-2


def test_non_affine_dependence_is_rejected():
    """A scalar entering nonlinearly cannot be tabulated; the held-out probe says so."""
    rng = np.random.default_rng(0)
    n, m = 3, 4
    A = rng.normal(size=(m, n))
    z0 = np.zeros(n)

    def build(gamma):
        v = Verifier(method="simulator", obbt_solver=None)
        p = v.parameter(m, lb=-np.ones(m), ub=np.ones(m))
        y_prev = v.iterate(n, lb=z0, ub=z0)
        y_curr = v.iterate(n, lb=z0, ub=z0)
        # gamma**2 is not affine in gamma.
        extrap = y_curr + (gamma**2) * (y_curr - y_prev)
        y_next = v.relu(extrap + A.T @ p)
        return Recording(v, next_state=[y_curr, y_next], state=[y_prev, y_curr])

    config = CodegenConfig(termination=TerminationSpec(max_iters=10))
    with pytest.raises(ScheduleFitError, match="does not depend affinely"):
        extract_scheduled_ir(build, {"gamma": _varying}, config=config)


def test_scheduled_factorization_is_rejected():
    """A factorization is computed once, so its system matrix may not vary."""
    rng = np.random.default_rng(1)
    n = 4
    M = rng.normal(size=(n, n))
    M = M @ M.T + n * np.eye(n)
    z0 = np.zeros(n)

    def build(rho):
        v = Verifier(method="simulator", obbt_solver=None)
        p = v.parameter(n, lb=-np.ones(n), ub=np.ones(n))
        z = v.iterate(n, lb=z0, ub=z0)
        # The schedule reaches the *system matrix*, not just a coefficient.
        z_next = v.implicit_linear(z + p, output_mat=M + rho * np.eye(n))
        return Recording(v, next_state=[z_next], state=[z])

    config = CodegenConfig(termination=TerminationSpec(max_iters=10))
    with pytest.raises(ScheduleFitError, match="linear system"):
        extract_scheduled_ir(build, {"rho": lambda k: 0.5 + 0.01 * k}, config=config)


def test_a_weakly_varying_factorization_is_still_rejected():
    """The check is only useful at a tight tolerance.

    A system matrix can depend on the schedule scalar too faintly to notice at
    ``np.allclose``'s default 1e-5 and still be a different matrix. Freezing it at
    the base factorization would emit a solver that quietly drifts from the
    recorded algorithm, so the comparison uses the same tolerance as every other
    numeric decision in the module.
    """
    rng = np.random.default_rng(4)
    n = 4
    M = rng.normal(size=(n, n))
    M = M @ M.T + n * np.eye(n)
    z0 = np.zeros(n)

    def build(rho):
        v = Verifier(method="simulator", obbt_solver=None)
        p = v.parameter(n, lb=-np.ones(n), ub=np.ones(n))
        z = v.iterate(n, lb=z0, ub=z0)
        z_next = v.implicit_linear(z + p, output_mat=M + rho * 1e-8 * np.eye(n))
        return Recording(v, next_state=[z_next], state=[z])

    config = CodegenConfig(termination=TerminationSpec(max_iters=10))
    with pytest.raises(ScheduleFitError, match="linear system"):
        extract_scheduled_ir(build, {"rho": lambda k: 0.5 + 0.01 * k}, config=config)


def test_shape_change_between_probes_is_rejected():
    """The build function must record the same algorithm at every probe value."""
    rng = np.random.default_rng(2)
    n, m = 3, 4
    A = rng.normal(size=(m, n))
    z0 = np.zeros(n)

    def build(gamma):
        v = Verifier(method="simulator", obbt_solver=None)
        p = v.parameter(m, lb=-np.ones(m), ub=np.ones(m))
        y_prev = v.iterate(n, lb=z0, ub=z0)
        y_curr = v.iterate(n, lb=z0, ub=z0)
        extrap = y_curr + gamma * (y_curr - y_prev)
        y_next = v.relu(extrap + A.T @ p)
        # Branching on the probe value records a different number of steps.
        if gamma > 0.5:
            y_next = v.relu(y_next)
        return Recording(v, next_state=[y_curr, y_next], state=[y_prev, y_curr])

    config = CodegenConfig(termination=TerminationSpec(max_iters=10))
    with pytest.raises(NonStationaryFOMError, match="changes shape"):
        extract_scheduled_ir(build, {"gamma": _varying}, config=config)


def test_a_vanishing_state_slot_is_reported_helpfully():
    """gamma = 0 would delete a state slot, so the probes never use it.

    This is the degeneracy behind the original bug: recorded at its first value,
    FISTA's momentum term disappears from the expression entirely. Forcing it
    with an explicit zero schedule should produce a shape error naming the state,
    not a wrong solver.
    """
    rng = np.random.default_rng(3)
    n, m = 3, 4
    A = rng.normal(size=(m, n))
    z0 = np.zeros(n)

    def build(gamma):
        v = Verifier(method="simulator", obbt_solver=None)
        p = v.parameter(m, lb=-np.ones(m), ub=np.ones(m))
        y_prev = v.iterate(n, lb=z0, ub=z0)
        y_curr = v.iterate(n, lb=z0, ub=z0)
        extrap = y_curr + gamma * (y_curr - y_prev)
        y_next = v.relu(extrap + A.T @ p)
        # No explicit state=, so the layout is inferred from what is referenced.
        return Recording(v, next_state=[y_curr, y_next])

    config = CodegenConfig(termination=TerminationSpec(max_iters=10))
    # The probes are all non-zero, so this lowers fine.
    ir = extract_scheduled_ir(build, {"gamma": _varying}, config=config)
    assert ir.state.dim == 2 * n


def test_unused_schedule_is_rejected():
    """An ignored scalar is almost always a misspelled keyword."""
    case = build_scheduled(max_iters=10)

    def build(gamma, unused):
        return case["build"](gamma=gamma)

    with pytest.raises(ScheduleFitError, match="do not affect any coefficient"):
        extract_scheduled_ir(
            build,
            {"gamma": _varying, "unused": lambda k: 1.0 + k},
            config=case["config"],
        )


def test_extract_scheduled_ir_needs_a_schedule():
    case = build_scheduled(max_iters=10)
    with pytest.raises(ValueError, match="at least one schedule"):
        extract_scheduled_ir(case["build"], {}, config=case["config"])


# -- schedule construction -------------------------------------------------


def test_as_schedule_accepts_callable_sequence_and_schedule():
    from_callable = as_schedule("g", lambda k: 2.0 * k, 5)
    assert from_callable.values.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]

    from_sequence = as_schedule("g", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 5)
    assert from_sequence.values.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]

    from_schedule = as_schedule("g", Schedule("g", np.arange(9.0)), 5)
    assert from_schedule.values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_as_schedule_rejects_a_table_shorter_than_the_loop():
    """A short table would be read past its end by a loop that runs further."""
    with pytest.raises(ValueError, match="can run 10 iterations"):
        as_schedule("g", [0.0, 1.0, 2.0], 10)


def test_as_schedule_calls_a_callable_with_increasing_indices():
    """A coefficient defined by a recurrence may cache, so the order is a contract.

    ``scheduled_example``'s momentum is written that way, and would return wrong
    values if the index ever went backwards.
    """
    seen = []

    def spec(k: int) -> float:
        seen.append(k)
        return float(k)

    values = as_schedule("g", spec, 6).values

    assert seen == [0, 1, 2, 3, 4, 5]
    assert values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
