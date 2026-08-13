"""The gate that catches a recording which cannot be rolled into a loop.

Extraction reads one iteration and repeats it. Before this gate existed nothing
compared that against the algorithm the user actually recorded, and every other
evaluator in the package is itself rolled, so all of them agreed with a wrongly
rolled IR. A FISTA recording therefore lowered to plain ISTA and passed every
check on the way out.

These tests pin the gate down on the recording that produced that bug.
"""

import re

import numpy as np
import pytest

from fomtools.codegen import (
    CodegenConfig,
    NonStationaryFOMError,
    ReferenceOracle,
    TerminationSpec,
    check_stationary,
    extract_ir,
    generate_code,
    run_recorded_tape,
)
from fomtools.verifier import Verifier
from tests.codegen.conftest import build_dr, build_ista, build_pdhg_lp, gate


def _record_lasso(momentum, K=6, n=3, m=4, seed=0):
    """Record a lasso method the way a user naturally would: one iterate leaf.

    With ``momentum=True`` this is FISTA, and it is *not* stationary: the
    coefficient on the previous iterate changes every step.
    """
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(m, n))
    eta = 1.0 / np.max(np.linalg.eigvalsh(D.T @ D))
    lambd = 0.05
    s0 = np.zeros(n)

    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(m, lb=-np.ones(m), ub=np.ones(m))
    s = v.add_iterate(n, lb=s0, ub=s0)

    y_prev, extrap, beta = s, s, 1.0
    iterates = []
    for _ in range(K):
        y = v.step.soft_threshold(extrap - eta * D.T @ (D @ extrap - x), lambd=lambd * eta)
        if momentum:
            beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
            extrap = y + ((beta - 1.0) / beta_next) * (y - y_prev)
            beta = beta_next
        else:
            extrap = y
        y_prev = y
        iterates.append(y)
    return v, iterates, x, rng.uniform(-1.0, 1.0, size=m)


def _config(max_iters=50):
    return CodegenConfig(termination=TerminationSpec(max_iters=max_iters))


def test_gate_rejects_the_fista_recording():
    """The regression test for the bug: FISTA must not lower silently."""
    v, iterates, x, x_val = _record_lasso(momentum=True)
    ir = extract_ir(v, iterates[0], config=_config())

    with pytest.raises(NonStationaryFOMError) as excinfo:
        check_stationary(ir, v, iterates, {x: x_val})

    message = str(excinfo.value)
    assert "does not reproduce the recorded algorithm" in message
    # It says where they part company and points at the fix.
    assert re.search(r"diverge at iteration \d+ of 6", message)
    assert "schedule" in message


def test_gate_accepts_the_ista_recording():
    """The same code path, stationary, must pass."""
    v, iterates, x, x_val = _record_lasso(momentum=False)
    ir = extract_ir(v, iterates[0], config=_config())
    check_stationary(ir, v, iterates, {x: x_val})


def test_the_wrongly_rolled_fista_really_is_ista():
    """Document the failure the gate prevents, rather than just asserting it fails.

    Extracted from its first iteration, where the momentum coefficient is zero,
    FISTA's IR is indistinguishable from plain ISTA — which is why the
    degradation was invisible.
    """
    v_f, iters_f, x_f, x_val = _record_lasso(momentum=True)
    v_i, iters_i, x_i, _ = _record_lasso(momentum=False)

    ir_f = extract_ir(v_f, iters_f[0], config=_config())
    ir_i = extract_ir(v_i, iters_i[0], config=_config())

    for name, md in ir_i.matrices.items():
        assert np.abs(ir_f.matrices[name].array - md.array).max() < 1e-14

    rolled = ReferenceOracle(ir_f).solve(param_values={"p0": x_val}).solution
    truth = run_recorded_tape(v_f, iters_f, {x_f: x_val})
    plain = ReferenceOracle(ir_i).solve(param_values={"p0": x_val}).solution

    assert np.abs(rolled - plain).max() < 1e-14  # it is ISTA
    assert np.abs(rolled - truth[len(iters_f)]).max() > 1e-3  # it is not FISTA


@pytest.mark.parametrize("builder", [build_ista, build_dr, build_pdhg_lp])
def test_recorded_tape_matches_the_rolled_ir_for_stationary_methods(builder):
    """Every stationary fixture must survive the gate."""
    case = builder(max_iters=50)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    kwargs = gate(case)
    check_stationary(ir, case["verifier"], kwargs["iterates"], kwargs["param_values"])


def test_generate_code_warns_when_stationarity_is_unchecked(tmp_path):
    case = build_ista(max_iters=20)
    with pytest.warns(UserWarning, match="did not check that the recorded method is stationary"):
        with pytest.raises(Exception):
            # No driver is configured; we only care that the warning fired first.
            generate_code(case["verifier"], case["next_state"], tmp_path, config=case["config"])


def test_generate_code_requires_param_values_with_iterates(tmp_path):
    case = build_ista(max_iters=20)
    with pytest.raises(ValueError, match="requires param_values"):
        generate_code(
            case["verifier"],
            case["next_state"],
            tmp_path,
            config=case["config"],
            iterates=case["iterates"][1:],
        )


def test_generate_code_rejects_a_build_function_without_schedules(tmp_path):
    from tests.codegen.conftest import build_scheduled

    case = build_scheduled(max_iters=20)
    with pytest.raises(TypeError, match="no schedules"):
        generate_code(case["build"], None, tmp_path, config=case["config"])
