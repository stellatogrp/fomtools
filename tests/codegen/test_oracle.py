"""Tests for the reference oracle: it must reproduce the FOM exactly."""

import numpy as np
import pytest

from fomtools.codegen.extract import extract_ir
from fomtools.codegen.oracle import ReferenceOracle, run_faithful
from tests.codegen.conftest import build_dr, build_ista, dr_reference, ista_reference

TEST_EPS = 1e-7


@pytest.mark.parametrize("offset", [0.0, 0.3, -0.5])
def test_ista_oracle_matches_reference(offset):
    case = build_ista(max_iters=300)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    b_val = case["data"]["b"] + offset

    res = ReferenceOracle(ir).solve(param_values={"p0": b_val})
    ref = ista_reference(case["data"], b_val, ir.state.init, n_iters=ir.termination.max_iters)
    assert np.allclose(res.solution, ref[-1], atol=TEST_EPS)


def test_ista_oracle_matches_faithful():
    case = build_ista(max_iters=120)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    b_val = case["data"]["b"]
    res = ReferenceOracle(ir).solve(param_values={"p0": b_val})
    traj = run_faithful(
        case["verifier"],
        case["next_state"],
        {case["param"]: b_val},
        n_iters=ir.termination.max_iters,
    )
    assert np.allclose(res.solution, traj[-1], atol=TEST_EPS)


def test_dr_oracle_matches_reference():
    case = build_dr(max_iters=300)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    q_val = np.hstack([case["data"]["c"], case["data"]["b"]])
    res = ReferenceOracle(ir).solve(param_values={"p0": q_val})
    ref = dr_reference(case["data"], q_val, ir.state.init, n_iters=ir.termination.max_iters)
    assert np.allclose(res.solution, ref[-1], atol=1e-6)


def test_dr_oracle_matches_faithful():
    case = build_dr(max_iters=120)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    q_val = np.hstack([case["data"]["c"], case["data"]["b"]])
    res = ReferenceOracle(ir).solve(param_values={"p0": q_val})
    traj = run_faithful(
        case["verifier"],
        case["next_state"],
        {case["param"]: q_val},
        n_iters=ir.termination.max_iters,
    )
    assert np.allclose(res.solution, traj[-1], atol=1e-7)


def test_inf_norm_change_termination_stops_early():
    case = build_ista(max_iters=1000, tol=1e-8)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    res = ReferenceOracle(ir).solve(param_values={"p0": case["data"]["b"]})
    assert res.iters < ir.termination.max_iters
    assert res.residual <= 1e-8


def test_oracle_trajectory_length():
    case = build_ista(max_iters=10)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    res = ReferenceOracle(ir).solve(param_values={"p0": case["data"]["b"]})
    assert len(res.trajectory) == res.iters + 1


def test_missing_param_raises():
    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    with pytest.raises(ValueError):
        ReferenceOracle(ir).solve(param_values={})
