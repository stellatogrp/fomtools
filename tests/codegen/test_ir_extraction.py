"""Tests for IR extraction from a built verifier."""

import numpy as np
import pytest

from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import ActivationKind, NonStationaryFOMError
from tests.codegen.conftest import build_dr, build_ista


def test_ista_ir_structure():
    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="ista")
    data = case["data"]

    assert ir.state_dim == data["n"]
    assert [p.dim for p in ir.params] == [data["m"]]

    # One soft-threshold body step.
    assert len(ir.body) == 1
    step = ir.body[0]
    assert step.activation.kind == ActivationKind.SOFT_THRESHOLD
    assert np.isclose(step.activation.lambd, data["lambd"] * data["t"])

    # The linear op reads state (At) and the parameter (Bt).
    sources = {t.source for t in step.linear_op.terms}
    assert sources == {"state", "param:p0"}
    state_mat = ir.matrices[[t.matrix for t in step.linear_op.terms if t.source == "state"][0]]
    param_mat = ir.matrices[[t.matrix for t in step.linear_op.terms if t.source == "param:p0"][0]]
    assert np.allclose(state_mat.array, data["At"])
    assert np.allclose(param_mat.array, data["Bt"])

    # State update is just the soft-threshold output.
    assert [t.source for t in ir.state_update.terms] == ["tmp:t0"]
    assert np.allclose(ir.state.init, 2.0)


def test_dr_ir_structure():
    case = build_dr()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="dr")
    n, m = case["data"]["n"], case["data"]["m"]

    assert ir.state_dim == n + m
    # Period-2 body: affine (identity activation) then partial-projection ReLU.
    assert len(ir.body) == 2
    assert ir.body[0].activation.kind == ActivationKind.IDENTITY
    assert ir.body[0].out_dim == n + m
    assert ir.body[1].activation.kind == ActivationKind.RELU
    assert ir.body[1].out_dim == m  # only the projected coordinates

    # State update mixes the previous state and both temporaries.
    sources = {t.source for t in ir.state_update.terms}
    assert sources == {"state", "tmp:t0", "tmp:t1"}


def test_next_state_with_no_steps_raises():
    case = build_ista()
    # The initial iterate depends on no recorded step.
    with pytest.raises(NonStationaryFOMError):
        extract_ir(case["verifier"], case["iterates"][0], config=case["config"])


def test_explicit_state_and_params():
    case = build_ista()
    ir = extract_ir(
        case["verifier"],
        case["next_state"],
        state=[case["iterates"][0]],
        params=[case["param"]],
        config=case["config"],
    )
    assert ir.state_dim == case["data"]["n"]
    assert [p.name for p in ir.params] == ["p0"]


def test_max_rhs_dim_covers_intermediates():
    case = build_dr()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    assert ir.max_rhs_dim >= ir.state_dim
