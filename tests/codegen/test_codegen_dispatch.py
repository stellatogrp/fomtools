"""Which generation path a call selects, and what a wrong call says.

``Verifier.codegen`` takes the run you recorded; ``Verifier.codegen_rolled`` takes
one stationary iteration. ``iterates=`` means the run to generate when it arrives
without ``next_state``, and the tape to check against when it arrives with one --
so the tests below pin both readings.
"""

import warnings
from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.config import CodegenConfig, TerminationSpec
from tests.codegen.conftest import build_ista, gate
from tests.codegen.test_unrolled_ir import build_halpern_loop, build_ista_loop

HALPERN_GOLDEN = Path(__file__).parent / "golden" / "halpern" / "embedded_c" / "fom_solver.c"
ISTA_GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"


# --- Verifier.codegen: the recorded run ------------------------------------


def test_codegen_generates_the_recorded_run(tmp_path, requires_compiler):
    """The whole point: verify with a loop, then generate that same loop."""
    v, iterates, _, _ = build_halpern_loop(steps=5)
    result = v.codegen(
        tmp_path, iterates=iterates, driver=StubAgentDriver(HALPERN_GOLDEN), name="halpern"
    )

    assert result.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    assert result.ir.unrolled is True
    assert len(result.ir.body) == 5


def test_codegen_accepts_a_trajectory(tmp_path):
    v, traj, _, _ = build_ista_loop(steps=6, trajectory=True)
    result = v.codegen(tmp_path, iterates=traj, driver=StubAgentDriver(ISTA_GOLDEN), name="ista")
    assert result.ir.unrolled is False, "a stationary recording rolls back up"


def test_codegen_rolls_a_stationary_recording(tmp_path):
    v, iterates, _, _ = build_ista_loop(steps=12)
    result = v.codegen(tmp_path, iterates=iterates, driver=StubAgentDriver(ISTA_GOLDEN))

    assert result.ir.unrolled is False
    assert len(result.ir.body) == 1
    assert result.ir.termination.max_iters == 12, "the recorded horizon is preserved"


def test_codegen_roll_false_forces_the_unrolled_body(tmp_path):
    v, iterates, _, _ = build_ista_loop(steps=6)
    result = v.codegen(tmp_path, iterates=iterates, driver=StubAgentDriver(ISTA_GOLDEN), roll=False)
    assert result.ir.unrolled is True
    assert len(result.ir.body) == 6


def test_codegen_needs_no_param_values_to_gate_itself(tmp_path):
    """Unlike the rolled path, which has to warn when given nothing to check with."""
    v, iterates, _, _ = build_halpern_loop(steps=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v.codegen(tmp_path, iterates=iterates, driver=StubAgentDriver(HALPERN_GOLDEN))

    stationarity = [w for w in caught if "did not check that the recorded" in str(w.message)]
    assert not stationarity, "the unrolled path gates itself, so it must not warn"


# --- Verifier.codegen_rolled: one stationary iteration ---------------------


def test_codegen_rolled_keeps_the_previous_behavior(tmp_path, requires_compiler):
    case = build_ista(max_iters=200)
    result = case["verifier"].codegen_rolled(
        tmp_path,
        case["next_state"],
        config=case["config"],
        driver=StubAgentDriver(ISTA_GOLDEN),
        name="ista",
        **gate(case),
    )
    assert result.success, result.translate.log
    assert result.ir.unrolled is False
    assert result.ir.termination.max_iters == 200, "the configured horizon is honored"


def test_codegen_rolled_still_warns_without_a_tape_to_check(tmp_path):
    case = build_ista()
    with pytest.warns(UserWarning, match="did not check that the recorded method"):
        case["verifier"].codegen_rolled(
            tmp_path,
            case["next_state"],
            config=case["config"],
            driver=StubAgentDriver(ISTA_GOLDEN),
        )


def test_iterates_with_next_state_still_means_check_this_tape(tmp_path):
    """The pre-existing reading of ``iterates=`` is unchanged."""
    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(ISTA_GOLDEN),
        **gate(case),
    )
    assert result.ir.unrolled is False
    assert len(result.ir.body) == 1


# --- wrong calls ------------------------------------------------------------


def test_passing_next_state_where_iterates_is_expected_is_caught(tmp_path):
    """The rename's migration hazard: a LinearExpression iterates as its rows."""
    v, iterates, _, _ = build_ista_loop(steps=4)
    with pytest.raises(TypeError, match="codegen_rolled"):
        v.codegen(tmp_path, iterates[0])


def test_neither_iterates_nor_next_state_is_caught(tmp_path):
    v, _, _, _ = build_ista_loop(steps=4)
    with pytest.raises(TypeError, match="either iterates= .*or next_state="):
        generate_code(v, None, tmp_path)


def test_a_build_function_without_schedules_is_caught(tmp_path):
    with pytest.raises(TypeError, match="no schedules="):
        generate_code(lambda gamma: None, None, tmp_path, iterates=[])


def test_tolerance_termination_is_rejected_on_the_recorded_run_path(tmp_path):
    from fomtools.codegen.config import TerminationKind

    v, iterates, _, _ = build_ista_loop(steps=4)
    config = CodegenConfig(
        termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=20, tol=1e-6)
    )
    with pytest.raises(ValueError, match="recorded tape"):
        v.codegen(tmp_path, iterates=iterates, config=config)
