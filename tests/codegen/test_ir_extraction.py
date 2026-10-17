"""Tests for IR extraction from a built verifier."""

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.codegen.config import CodegenConfig, TerminationSpec
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import ActivationKind, NonStationaryFOMError
from fomtools.codegen.oracle import ReferenceOracle, run_recorded_tape
from fomtools.verifier import Verifier
from tests.codegen.conftest import build_dr, build_ista

ONE_PASS = CodegenConfig(termination=TerminationSpec(max_iters=1))


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
    assert np.allclose(state_mat.dense(), data["At"])
    assert np.allclose(param_mat.dense(), data["Bt"])

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


def test_identity_affine_is_a_copy_not_a_solve():
    """``M w = rhs`` with ``M = I`` must not become a factorized solve.

    ``IterateTrajectory`` materializes an iterate with exactly this step, so
    factorizing it would cost a stored LDL^T plus two triangular solves per
    iteration for what is a copy.
    """
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(3, lb=-np.ones(3), ub=np.ones(3))
    z = v.add_iterate(3, lb=np.zeros(3), ub=np.zeros(3))
    w = v.step.implicit_linear(z + x, output_mat=spa.eye(3))
    out = v.step.relu(w)

    ir = extract_ir(v, out, config=ONE_PASS)

    assert ir.factorizations == {}
    assert [s.solve for s in ir.body] == [None, None]
    assert ir.body[0].activation.kind == ActivationKind.IDENTITY

    xval = np.array([0.5, -0.25, 0.75])
    truth = run_recorded_tape(v, [w, out], {x: xval})
    got = ReferenceOracle(ir).solve(param_values={"p0": xval}, x0=truth[0]).solution
    assert np.allclose(got, truth[-1], atol=1e-12)


def test_a_nearly_identity_operator_is_still_solved():
    """The identity check is exact: near-identity changes numerics, not codegen."""
    almost = np.eye(3)
    almost[1, 1] = 1.0 + 1e-9
    v = Verifier(method="simulator", obbt_solver=None)
    z = v.add_iterate(3, lb=np.zeros(3), ub=np.zeros(3))
    out = v.step.implicit_linear(z, output_mat=almost)

    ir = extract_ir(v, out, config=ONE_PASS)

    assert list(ir.factorizations) == ["t0"]
    assert ir.body[0].solve == "t0"


def test_repeated_affine_operator_shares_one_factorization():
    """Two solves against the same operator share a factorization, dense or sparse."""
    mat = np.array([[2.0, 1.0], [1.0, 3.0]])
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(2, lb=-np.ones(2), ub=np.ones(2))
    z = v.add_iterate(2, lb=np.zeros(2), ub=np.zeros(2))
    a = v.step.implicit_linear(z + x, output_mat=mat)
    b = v.step.implicit_linear(a + x, output_mat=spa.csr_matrix(mat))

    ir = extract_ir(v, b, config=ONE_PASS)

    assert list(ir.factorizations) == ["t0"]
    assert [s.solve for s in ir.body] == ["t0", "t0"]

    xval = np.array([0.3, -0.7])
    truth = run_recorded_tape(v, [a, b], {x: xval})
    got = ReferenceOracle(ir).solve(param_values={"p0": xval}, x0=truth[0]).solution
    assert np.allclose(got, truth[-1], atol=1e-12)


def test_distinct_affine_operators_keep_separate_factorizations():
    v = Verifier(method="simulator", obbt_solver=None)
    z = v.add_iterate(2, lb=np.zeros(2), ub=np.zeros(2))
    a = v.step.implicit_linear(z, output_mat=np.array([[2.0, 1.0], [1.0, 3.0]]))
    b = v.step.implicit_linear(a, output_mat=np.array([[4.0, 0.0], [0.0, 5.0]]))

    ir = extract_ir(v, b, config=ONE_PASS)

    assert sorted(ir.factorizations) == ["t0", "t1"]
    assert [s.solve for s in ir.body] == ["t0", "t1"]
