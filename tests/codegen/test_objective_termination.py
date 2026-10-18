"""Termination on the verification objective (``TerminationKind.OBJECTIVE``).

The generated solver stops at the first checked iteration ``k`` (every
``check_termination``-th, and the last) with ``r^k <= tol``, where ``r^k`` is the
objective set with ``Verifier.set_objective_norm``. These tests cover the
configuration, the mapping of the objective onto the loop's vectors (and its
errors), the oracle against brute force, the spec round trip, the emitted
artifacts, and compiled end-to-end runs on the rolled, scheduled and unrolled
paths.
"""

import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen import spec as spec_mod
from fomtools.codegen.agent.prompts import code_optimization_instructions, translation_instructions
from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.extract_scheduled import extract_scheduled_ir
from fomtools.codegen.extract_unrolled import extract_unrolled_ir
from fomtools.codegen.generate import extract_fom_ir
from fomtools.codegen.harness import build_and_verify, build_lib
from fomtools.codegen.ir import ActivationKind
from fomtools.codegen.kernel_rules import strip_c_comments_and_strings
from fomtools.codegen.objective import ObjectiveExtractionError, norm_label
from fomtools.codegen.oracle import ReferenceOracle
from fomtools.codegen.targets import get_target
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from fomtools.codegen.targets.large_scale_cuda import CudaTarget
from fomtools.codegen.targets.large_scale_openmp import OpenMpTarget
from fomtools.verifier import Verifier
from tests.codegen.conftest import (
    build_dr,
    build_ista,
    build_ista_objective,
    build_scheduled_objective,
    fista_example,
    gate,
    ista_reference,
    objective_config,
)

GOLDEN = Path(__file__).parent / "golden"
ISTA_GOLDEN = GOLDEN / "ista_objective" / "embedded_c" / "fom_solver.c"
SCHEDULED_GOLDEN = GOLDEN / "scheduled_objective" / "embedded_c" / "fom_solver.c"
FISTA_GOLDEN = GOLDEN / "fista_objective" / "embedded_c" / "fom_solver.c"
PLAIN_ISTA_GOLDEN = GOLDEN / "ista" / "embedded_c" / "fom_solver.c"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def brute_stop(values, check, max_iter, tol):
    """``(k, r^k, solved)`` of the first tested iteration with ``r^k <= tol``.

    ``values[k - 1]`` is ``r^k``. Tested iterations are every ``check``-th and
    the last, which is the semantics the oracle and the C must share.
    """
    for k in range(1, max_iter + 1):
        if k == max_iter or (check > 0 and k % check == 0):
            if values[k - 1] <= tol or k == max_iter:
                return k, values[k - 1], bool(values[k - 1] <= tol)
    raise AssertionError("unreachable")


def ista_values(case, b_val, n_iters):
    """``||x_k - x_{k-1}||_inf`` for k = 1..n_iters, in plain NumPy."""
    traj = ista_reference(case["data"], b_val, np.full(case["data"]["n"], 2.0), n_iters)
    return [float(np.max(np.abs(traj[k] - traj[k - 1]))) for k in range(1, n_iters + 1)]


def with_settings(ir, **update):
    """The IR as the C runs it at other runtime settings."""
    return replace(ir, termination=ir.termination.model_copy(update=update))


def ista_ir(**kw):
    case = build_ista_objective(**kw)
    return case, extract_fom_ir(
        case["verifier"], case["next_state"], config=case["config"], **gate(case)
    )


def _load_wrapper(out_dir):
    spec = importlib.util.spec_from_file_location(
        f"fom_solver_wrapper_{abs(hash(str(out_dir)))}", out_dir / "python" / "fom_solver.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_nnls(*, K=3, seed=1, n=4, m=6):
    """Projected gradient for nonnegative least squares, ``x <- relu(x - t A'(Ax - b))``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    b0 = rng.normal(size=m)
    t = 1.0 / np.linalg.norm(A, 2) ** 2
    v = Verifier(method="simulator", obbt_solver=None)
    b = v.add_param(m, lb=b0 - 0.5, ub=b0 + 0.5)
    xs = [v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))]
    for _ in range(K):
        xs.append(v.step.relu(xs[-1] - t * (A.T @ (A @ xs[-1] - b))))
    return dict(verifier=v, iterates=xs, param=b, data=dict(A=A, b0=b0, t=t, n=n, m=m))


def nnls_kkt(case):
    """Set the KKT-type objective [x_K - x_{K-1}; max(-A'(A x_K - b), 0)]."""
    v, xs, b = case["verifier"], case["iterates"], case["param"]
    A = case["data"]["A"]
    dual_infeas = v.step.relu(-1.0 * (A.T @ (A @ xs[-1] - b)))
    return xs[-1] - xs[-2], dual_infeas


def nnls_values(case, b_val, n_iters, norm):
    d = case["data"]
    A, t = d["A"], d["t"]
    x = np.zeros(d["n"])
    out = []
    for _ in range(n_iters):
        new = np.maximum(x - t * A.T @ (A @ x - b_val), 0.0)
        r = np.concatenate([new - x, np.maximum(-(A.T @ (A @ new - b_val)), 0.0)])
        out.append({"inf": np.max(np.abs(r)), "1": np.sum(np.abs(r)), "2": np.sum(r * r)}[norm])
        x = new
    return out


# --------------------------------------------------------------------------
# configuration and the verifier's record
# --------------------------------------------------------------------------
def test_objective_kind_defaults_and_validation():
    spec = TerminationSpec(kind=TerminationKind.OBJECTIVE, tol=1e-6)
    assert spec.check_termination == 25
    assert spec.uses_objective and not spec.uses_tolerance
    # The other criteria keep their default interval.
    assert TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, tol=1e-6).check_termination == 1
    assert TerminationSpec().check_termination == 1
    assert TerminationSpec(kind="objective", tol=1.0, check_termination=3).check_termination == 3
    assert TerminationSpec(kind="objective", tol=1.0, check_termination=0).check_termination == 0
    for tol in (None, 0.0, -1.0):
        with pytest.raises(ValueError, match="tol must be a positive float for the objective"):
            TerminationSpec(kind=TerminationKind.OBJECTIVE, tol=tol)


def test_check_iterations_include_the_last():
    spec = TerminationSpec(kind="objective", tol=1.0, max_iters=60, check_termination=25)
    assert [k for k in range(1, 61) if spec.is_check_iteration(k)] == [25, 50, 60]
    assert [k for k in range(1, 11) if spec.is_check_iteration(k, max_iter=10)] == [10]
    off = TerminationSpec(kind="objective", tol=1.0, max_iters=5, check_termination=0)
    assert [k for k in range(1, 6) if off.is_check_iteration(k)] == [5]


def test_verifier_records_the_objective():
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_iterate(2, lb=0, ub=1)
    assert v.objective is None
    for ord, label in ((1, "1"), (2, "2"), (np.inf, "inf"), ("inf", "inf")):
        v.set_objective_norm(x, ord)
        exprs, recorded = v.objective
        assert exprs == (x,) and norm_label(recorded) == label
    v.set_objective_norm([x, 2 * x], 2)
    assert len(v.objective[0]) == 2
    v.set_zero_objective()
    assert v.objective is None


def test_norm_label_rejects_unknown_orders():
    with pytest.raises(ObjectiveExtractionError, match="unsupported objective norm"):
        norm_label(3)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def test_fixed_point_residual_at_the_last_recorded_iteration():
    case, ir = ista_ir()
    (block,) = ir.objective.blocks
    assert ir.objective.norm == "inf"
    assert block.activation.kind == ActivationKind.IDENTITY
    terms = {t.source: ir.matrices[t.matrix].dense() for t in block.linear_op.terms}
    n = case["data"]["n"]
    np.testing.assert_array_equal(terms["new_state"], np.eye(n))
    np.testing.assert_array_equal(terms["state"], -np.eye(n))
    assert ir.objective_inputs() == []


def test_objective_at_the_last_iteration_needs_the_tape():
    case = build_ista_objective()
    with pytest.raises(ObjectiveExtractionError, match="pass\\s+iterates="):
        extract_ir(case["verifier"], case["next_state"], config=case["config"])


def test_objective_on_the_extracted_iteration_needs_no_tape():
    case = build_ista(K=3)
    its = case["iterates"]
    case["verifier"].set_objective_norm(its[1] - its[0], 1)
    ir = extract_ir(case["verifier"], case["next_state"], config=objective_config())
    assert ir.objective.norm == "1"
    sources = {t.source for t in ir.objective.blocks[0].linear_op.terms}
    assert sources == {"new_state", "state"}


def test_partial_state_objective_reads_one_slot():
    """FISTA carries (y_{k-1}, y_k); the objective reads only the y half."""
    case = build_scheduled_objective(extrapolated=False)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    (block,) = ir.objective.blocks
    n = case["data"]["n"]
    (term,) = block.linear_op.terms
    assert term.source == "new_state" and term.scale is None
    # new_state = (y_k, y_{k+1}): the residual is its second half minus its first.
    np.testing.assert_array_equal(
        ir.matrices[term.matrix].dense(), np.hstack([-np.eye(n), np.eye(n)])
    )


def test_scheduled_objective_reads_the_schedule():
    case = build_scheduled_objective(extrapolated=True)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    scaled = [t for b in ir.objective.blocks for t in b.linear_op.terms if t.scale]
    assert {t.scale for t in scaled} == {"gamma"}


def test_kkt_residual_with_a_relu_block():
    case = build_nnls()
    linear, relu = nnls_kkt(case)
    case["verifier"].set_objective_norm([linear, relu], np.inf)
    ir = extract_fom_ir(
        case["verifier"],
        case["iterates"][1],
        config=objective_config(),
        iterates=case["iterates"][1:],
        param_values={case["param"]: case["data"]["b0"]},
    )
    kinds = [b.activation.kind for b in ir.objective.blocks]
    assert kinds == [ActivationKind.IDENTITY, ActivationKind.RELU]
    assert [b.out_dim for b in ir.objective.blocks] == [case["data"]["n"]] * 2
    relu_sources = {t.source for t in ir.objective.blocks[1].linear_op.terms}
    assert relu_sources == {"new_state", "param:p0"}


def test_mixed_rows_in_one_expression_and_scaled_relu():
    """One expression may hold linear rows and single-ReLU rows; ``a relu(z)``
    is lowered to ``relu(|a| z)``, which every norm treats identically."""
    from fomtools.linearexpression import stack

    case = build_nnls()
    linear, relu = nnls_kkt(case)
    case["verifier"].set_objective_norm(stack([linear, -2.0 * relu]), 1)
    ir = extract_fom_ir(
        case["verifier"],
        case["iterates"][1],
        config=objective_config(),
        iterates=case["iterates"][1:],
        param_values={case["param"]: case["data"]["b0"]},
    )
    assert [b.activation.kind for b in ir.objective.blocks] == [
        ActivationKind.IDENTITY,
        ActivationKind.RELU,
    ]
    b_val = case["data"]["b0"] + 0.1
    oracle = ReferenceOracle(with_settings(ir, check_termination=1, max_iters=30))
    got = [r for _, r in oracle.solve(param_values={"p0": b_val}).objectives]
    d = case["data"]
    x = np.zeros(d["n"])
    for k, value in enumerate(got):
        new = np.maximum(x - d["t"] * d["A"].T @ (d["A"] @ x - b_val), 0.0)
        relu_part = np.maximum(-(d["A"].T @ (d["A"] @ new - b_val)), 0.0)
        expected = np.sum(np.abs(new - x)) + np.sum(2.0 * relu_part)
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-14), k
        x = new


@pytest.mark.parametrize(
    "objective, match",
    [
        (lambda xs, relu, relu2: relu + xs[-1], "adds a ReLU term to other terms"),
        (lambda xs, relu, relu2: relu + relu2, "sums several ReLU terms"),
        (lambda xs, relu, relu2: xs[-1] - xs[-3], "older than"),
    ],
)
def test_objective_rows_the_loop_cannot_evaluate(objective, match):
    case = build_nnls()
    _, relu = nnls_kkt(case)
    relu2 = case["verifier"].step.relu(case["iterates"][-1] - 1.0)
    case["verifier"].set_objective_norm(objective(case["iterates"], relu, relu2), np.inf)
    with pytest.raises(ObjectiveExtractionError, match=match):
        extract_ir(
            case["verifier"],
            case["iterates"][1],
            config=objective_config(),
            iterates=case["iterates"][1:],
        )


def test_nested_relu_is_rejected():
    case = build_nnls()
    _, relu = nnls_kkt(case)
    nested = case["verifier"].step.relu(relu - 1.0)
    case["verifier"].set_objective_norm(nested, np.inf)
    with pytest.raises(ObjectiveExtractionError, match="nested"):
        extract_ir(
            case["verifier"],
            case["iterates"][1],
            config=objective_config(),
            iterates=case["iterates"][1:],
        )


def test_other_steps_outside_the_loop_are_rejected():
    case = build_ista(K=3)
    v, its = case["verifier"], case["iterates"]
    v.set_objective_norm(v.step.soft_threshold(its[-1], lambd=0.1), np.inf)
    with pytest.raises(ObjectiveExtractionError, match="does not carry"):
        extract_fom_ir(v, case["next_state"], config=objective_config(), **gate(case))


def test_parameters_the_iteration_does_not_read_are_rejected():
    case = build_ista(K=3)
    v, its = case["verifier"], case["iterates"]
    extra = v.add_param(case["data"]["n"], lb=-1.0, ub=1.0)
    v.set_objective_norm(its[-1] - its[-2] + extra, np.inf)
    with pytest.raises(ObjectiveExtractionError, match="parameter that the iteration does not"):
        extract_fom_ir(v, case["next_state"], config=objective_config(), **gate(case))


def test_objective_kind_without_an_objective():
    case = build_ista(K=3)
    with pytest.raises(ObjectiveExtractionError, match="none is set"):
        extract_ir(case["verifier"], case["next_state"], config=objective_config())


def test_ir_requires_an_objective_for_the_objective_kind():
    _, ir = ista_ir()
    with pytest.raises(ValueError, match="needs the verification objective"):
        replace(ir, objective=None)
    # Any other criterion may carry one (the stationarity probes do).
    replace(ir, termination=TerminationSpec(max_iters=3))


def test_accumulated_iterate_maps_on_the_extracted_iteration_through_temporaries():
    """Douglas-Rachford carries z = z + ut - u, a non-leaf expression. Written on
    the extracted iteration, its fixed-point residual reads that iteration's
    step outputs; written at the last recorded iteration it cannot be read."""
    case = build_dr(K=3)
    z = case["iterates"]
    case["verifier"].set_objective_norm(z[1] - z[0], np.inf)
    ir = extract_ir(case["verifier"], case["next_state"], config=objective_config())
    assert ir.objective_inputs() == ["tmp:t0", "tmp:t1"]

    q = 0.5 * (case["param"].lb + case["param"].ub)
    from tests.codegen.conftest import dr_reference

    traj = dr_reference(case["data"], q, np.zeros(case["data"]["n"] + case["data"]["m"]), 60)
    values = [float(np.max(np.abs(traj[k] - traj[k - 1]))) for k in range(1, 61)]
    oracle = ReferenceOracle(with_settings(ir, check_termination=1, max_iters=60, tol=1e-12))
    got = [r for _, r in oracle.solve(param_values={"p0": q}).objectives]
    np.testing.assert_allclose(got, values[: len(got)], rtol=1e-9, atol=1e-12)

    late = build_dr(K=3)
    zz = late["iterates"]
    late["verifier"].set_objective_norm(zz[-1] - zz[-2], np.inf)
    with pytest.raises(ObjectiveExtractionError, match="iterate_trajectory"):
        extract_fom_ir(
            late["verifier"], late["next_state"], config=objective_config(), **gate(late)
        )


def test_trajectory_iterates_map_at_the_last_iteration():
    from tests.codegen.conftest import build_dr_loop

    v, z, q = build_dr_loop(steps=6)
    v.set_objective_norm(z[-1] - z[-2], np.inf)
    ir = extract_unrolled_ir(v, z, config=objective_config(tol=1e-4))
    assert not ir.unrolled and ir.termination.max_iters == 6
    assert ir.termination.uses_objective
    assert ir.objective_inputs() == []


# --------------------------------------------------------------------------
# oracle against brute force
# --------------------------------------------------------------------------
@pytest.mark.parametrize("check", [1, 7, 25, 0])
@pytest.mark.parametrize("shift", [0.0, 0.3, -0.4])
def test_oracle_stop_matches_brute_force_ista(check, shift):
    case, ir = ista_ir(tol=1e-6)
    b_val = case["data"]["b"] + shift
    for max_iter in (200, 13):
        oracle = ReferenceOracle(with_settings(ir, check_termination=check, max_iters=max_iter))
        res = oracle.solve(param_values={"p0": b_val})
        k, value, solved = brute_stop(ista_values(case, b_val, max_iter), check, max_iter, 1e-6)
        assert (res.iters, res.solved) == (k, solved)
        assert res.residual == pytest.approx(value, rel=1e-9, abs=1e-15)
        assert res.objectives[-1][0] == k
        # max_iters caps: never past it, and the last iteration is always tested.
        assert res.iters <= max_iter


@pytest.mark.parametrize("norm", ["inf", "1", "2"])
def test_oracle_stop_matches_brute_force_kkt(norm):
    case = build_nnls()
    linear, relu = nnls_kkt(case)
    case["verifier"].set_objective_norm([linear, relu], {"inf": np.inf, "1": 1, "2": 2}[norm])
    ir = extract_fom_ir(
        case["verifier"],
        case["iterates"][1],
        config=objective_config(max_iters=400, tol=1e-5),
        iterates=case["iterates"][1:],
        param_values={case["param"]: case["data"]["b0"]},
    )
    for check in (1, 7, 25):
        b_val = case["data"]["b0"] - 0.2
        res = ReferenceOracle(with_settings(ir, check_termination=check)).solve(
            param_values={"p0": b_val}
        )
        k, value, solved = brute_stop(nnls_values(case, b_val, 400, norm), check, 400, 1e-5)
        assert (res.iters, res.solved) == (k, solved)
        assert res.residual == pytest.approx(value, rel=1e-9, abs=1e-15)


def test_oracle_stop_matches_brute_force_scheduled():
    case = build_scheduled_objective(extrapolated=True, tol=1e-7)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    d = case["data"]
    gamma = case["schedules"]["gamma"]
    for shift in (0.0, 0.5):
        b_val = d["b"] + shift
        y_prev = y = d["z0"].copy()
        values = []
        for k in range(300):
            extrap = y + gamma(k) * (y - y_prev)
            r = d["At"] @ extrap + d["Bt"] @ b_val
            y_next = np.sign(r) * np.maximum(np.abs(r) - d["lambd"] * d["t"], 0.0)
            values.append(float(np.max(np.abs(y_next - extrap))))
            y_prev, y = y, y_next
        for check in (1, 7, 25):
            res = ReferenceOracle(with_settings(ir, check_termination=check)).solve(
                param_values={"p0": b_val}
            )
            k, value, solved = brute_stop(values, check, 300, 1e-7)
            assert (res.iters, res.solved) == (k, solved)
            assert res.residual == pytest.approx(value, rel=1e-8, abs=1e-14)


def test_unrolled_objective_is_evaluated_once_at_the_end():
    ex = fista_example()
    v, its = ex["verifier"], ex["iterates"]
    v.set_objective_norm(its[-1] - its[-2], np.inf)
    ir = extract_unrolled_ir(v, its, config=objective_config(tol=1e-3))
    assert ir.unrolled and ir.termination.max_iters == 1
    assert ir.objective_inputs() == ["tmp:t4"]
    b_val = ex["data"]["b"]
    res = ReferenceOracle(ir).solve(param_values={"p0": b_val})
    traj = ex_trajectory(ex, b_val)
    expected = float(np.max(np.abs(traj[-1] - traj[-2])))
    assert res.iters == 1 and res.objectives == [(1, pytest.approx(expected, rel=1e-12))]
    assert res.solved == (expected <= 1e-3)


def ex_trajectory(ex, b_val):
    from tests.codegen.conftest import fista_loop_reference

    d = ex["data"]
    return [d["z0"], *fista_loop_reference(d, b_val, d["K"])]


def test_infinity_norm_change_still_rejected_on_the_tape():
    ex = fista_example()
    with pytest.raises(ValueError, match="OBJECTIVE"):
        extract_unrolled_ir(
            ex["verifier"],
            ex["iterates"],
            config=CodegenConfig(
                termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, tol=1e-6)
            ),
        )


# --------------------------------------------------------------------------
# spec round trip
# --------------------------------------------------------------------------
def test_spec_round_trip_keeps_the_objective(tmp_path):
    case = build_scheduled_objective(extrapolated=True)
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    spec_mod.save(ir, tmp_path)
    loaded = spec_mod.load(tmp_path)
    assert loaded.objective == ir.objective
    assert loaded.termination == ir.termination
    for name in (m for b in ir.objective.blocks for m in [t.matrix for t in b.linear_op.terms]):
        assert loaded.matrices[name].is_sparse
        np.testing.assert_array_equal(loaded.matrices[name].dense(), ir.matrices[name].dense())
    b_val = case["data"]["b"]
    a = ReferenceOracle(ir).solve(param_values={"p0": b_val})
    b = ReferenceOracle(loaded).solve(param_values={"p0": b_val})
    assert (a.iters, a.residual) == (b.iters, b.residual)

    # A spec written before objective termination has no objective entry.
    raw = json.loads((tmp_path / "spec.json").read_text())
    raw.pop("objective")
    raw["termination"]["kind"] = "max_iters"
    (tmp_path / "spec.json").write_text(json.dumps(raw))
    assert spec_mod.load(tmp_path).objective is None


def test_spec_round_trip_relu_block(tmp_path):
    case = build_nnls()
    case["verifier"].set_objective_norm(list(nnls_kkt(case)), 2)
    ir = extract_fom_ir(
        case["verifier"],
        case["iterates"][1],
        config=objective_config(),
        iterates=case["iterates"][1:],
        param_values={case["param"]: case["data"]["b0"]},
    )
    spec_mod.save(ir, tmp_path)
    assert spec_mod.load(tmp_path).objective == ir.objective


# --------------------------------------------------------------------------
# emitted artifacts
# --------------------------------------------------------------------------
def test_emitted_artifacts(tmp_path):
    case, ir = ista_ir()
    scaffold(ir, tmp_path, case["config"])
    header = (tmp_path / "include" / "fom_workspace.h").read_text()
    assert "#define FOM_USE_OBJECTIVE 1" in header
    assert "#define FOM_USE_TOL 0" in header
    assert "#define FOM_CHECK_TERMINATION 25" in header
    ops_h = (tmp_path / "include" / "fom_ops.h").read_text()
    assert (
        "fom_float fom_objective(fom_int it, const fom_float *state, "
        "const fom_float *new_state);" in ops_h
    )
    obj_c = (tmp_path / "src" / "fom_objective.c").read_text()
    assert "FOM_OBJ_OBJ0_NEW_ROWPTR" in obj_c and "fom_objective(" in obj_c
    # The metric has its own source: the editable fom_ops.c holds none of it.
    assert "objective" not in (tmp_path / "src" / "fom_ops.c").read_text()
    data_c = (tmp_path / "src" / "fom_data.c").read_text()
    assert "obj0" not in data_c  # objective blocks live with fom_objective
    spec_md = (tmp_path / "SPEC.md").read_text()
    assert "## Termination: the verification objective" in spec_md
    assert "r = fom_objective(it, work->state, work->new_state);" in spec_md
    verify = (tmp_path / "test" / "verify_data.h").read_text()
    assert "#define FOM_VERIFY_OBJECTIVE 1" in verify
    assert "FOM_VERIFY_CASE_CHECK[3] = {1, 7, 25}" in verify
    makefile = (tmp_path / "Makefile").read_text()
    assert "verify: objective_guard" in makefile
    assert makefile.index("verify:") < makefile.index("objective_guard:")  # default goal
    assert "src/fom_objective.c" in makefile.split("SRC = ", 1)[1].split("\n# ", 1)[0]


def test_objective_source_is_division_and_allocation_free():
    case = build_nnls()
    case["verifier"].set_objective_norm(list(nnls_kkt(case)), 2)
    config = objective_config()
    ir = extract_fom_ir(
        case["verifier"],
        case["iterates"][1],
        config=config,
        iterates=case["iterates"][1:],
        param_values={case["param"]: case["data"]["b0"]},
    )
    code = strip_c_comments_and_strings("\n".join(EmbeddedCTarget(config)._objective_source(ir)))
    assert "/" not in code
    assert "alloc" not in code


def test_other_criteria_emit_no_objective(tmp_path):
    case = build_ista(K=3)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    scaffold(ir, tmp_path, case["config"])
    assert "fom_objective" not in (tmp_path / "include" / "fom_ops.h").read_text()
    # Still written (the build and the CVXPYgen sources list it), with no code.
    assert "fom_objective" not in (tmp_path / "src" / "fom_objective.c").read_text()
    assert "objective_guard:" not in (tmp_path / "Makefile").read_text()
    assert "#define FOM_USE_OBJECTIVE 0" in (tmp_path / "include" / "fom_workspace.h").read_text()


def test_prompts_require_fom_objective():
    case, ir = ista_ir()
    target = get_target(case["config"])
    for text in (translation_instructions(ir, target), code_optimization_instructions(ir, target)):
        assert "fom_objective" in text and "check_termination" in text
    plain = build_ista(K=3)
    plain_ir = extract_ir(plain["verifier"], plain["next_state"], config=plain["config"])
    assert "fom_objective" not in translation_instructions(plain_ir, target)


def _dr_objective_ir(config):
    """Douglas-Rachford (a factorized affine solve) stopping on its residual."""
    case = build_dr(K=3)
    z = case["iterates"]
    case["verifier"].set_objective_norm(z[1] - z[0], np.inf)
    return extract_ir(case["verifier"], case["next_state"], config=config)


@pytest.mark.parametrize("target_cls", [EmbeddedCTarget, OpenMpTarget, CudaTarget])
def test_optimization_prompt_keeps_the_objective_rule_with_an_editable_linsolve(target_cls):
    """Where the optimizer may also rewrite the linear solves (and the matrix-vector
    products), the objective stays fixed in a source it may not edit and must
    still gate the stop."""
    config = objective_config()
    ir = _dr_objective_ir(config)
    assert ir.factorizations and ir.termination.uses_objective
    target = target_cls(config)
    text = code_optimization_instructions(ir, target)
    editable = target.linsolve_source_path() in target.optimization_editable_paths()
    assert ("You may edit" in text) == editable
    assert "Keep calling `fom_objective` exactly where it is called now" in text
    assert f"It lives in `{target.objective_source_path()}`, which stays fixed" in text
    assert target.objective_source_path() not in target.optimization_editable_paths()


def test_verify_holds_both_linsolve_and_objective_checks(tmp_path):
    config = objective_config()
    ir = _dr_objective_ir(config)
    scaffold(ir, tmp_path, config)
    data = (tmp_path / "test" / "verify_data.h").read_text()
    main = (tmp_path / "test" / "verify_main.c").read_text()
    assert "FOM_VERIFY_LS_T0_B" in data and "FOM_VERIFY_CASE_CHECK" in data
    assert "fom_lin_solve_t0(FOM_VERIFY_LS_T0_B[r], x);" in main
    assert "check_stop(" in main and "FOM_VERIFY_NUM_CASES > 0" in main


@pytest.mark.parametrize("target_cls", [EmbeddedCTarget, OpenMpTarget, CudaTarget])
def test_static_checks_enforce_the_objective_contract(target_cls):
    case, ir = ista_ir()
    target = target_cls(case["config"])
    assert target.static_checks(ISTA_GOLDEN.read_text(), ir) == []
    missing_call = PLAIN_ISTA_GOLDEN.read_text().replace("check_termination", "")
    violations = target.static_checks(missing_call, ir)
    assert any("fom_objective" in v for v in violations)
    assert any("check_termination" in v for v in violations)
    # Mentions in comments do not count.
    commented = "/* fom_objective(it, s, n); check_termination */\n" + missing_call
    assert len(target.static_checks(commented, ir)) >= 2
    # Without the objective kind there is nothing to enforce.
    assert not any("objective" in v for v in target.static_checks(missing_call, None))


def test_cuda_emission(tmp_path):
    """No GPU here: check the device objective is emitted and compiles in
    principle (kernels, host fold, guard on the compile target)."""
    case = build_scheduled_objective(extrapolated=True, ord=2)
    config = case["config"].model_copy(
        update={"version": Version.LARGE_SCALE, "backend": Backend.CUDA}
    )
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=config)
    scaffold(ir, tmp_path, config)
    ops = (tmp_path / "src" / "fom_ops.cu").read_text()
    assert "__global__ void fom_objective_obj0_kern(" in ops
    assert 'extern "C" fom_float fom_objective(fom_int it,' in ops
    assert "fom_sched_gamma(it)" in ops  # schedule read on the host
    assert "sh[threadIdx.x] += sh[threadIdx.x + w];" in ops  # sum reduction for ord 2
    assert "DEVICE pointer" in (tmp_path / "include" / "fom_ops.h").read_text()
    assert "compile: objective_guard" in (tmp_path / "Makefile").read_text()

    case_inf, ir_inf = ista_ir()
    cuda = case_inf["config"].model_copy(
        update={"version": Version.LARGE_SCALE, "backend": Backend.CUDA}
    )
    scaffold(ir_inf, tmp_path / "inf", cuda)
    ops = (tmp_path / "inf" / "src" / "fom_ops.cu").read_text()
    assert "r = r > v ? r : v;" in ops and "(void)it;" in ops


# --------------------------------------------------------------------------
# end to end, compiled
# --------------------------------------------------------------------------
def _generate(case, tmp_path, golden, config=None, **kwargs):
    return generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=config or case["config"],
        driver=StubAgentDriver(golden),
        name="ista",
        **gate(case),
        **kwargs,
    )


@pytest.mark.parametrize(
    "version, backend", [(Version.EMBEDDED, Backend.C), (Version.LARGE_SCALE, Backend.OPENMP)]
)
def test_rolled_ista_verifies(tmp_path, requires_compiler, version, backend):
    case = build_ista_objective(version=version, backend=backend)
    result = _generate(case, tmp_path, ISTA_GOLDEN)
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_runtime_settings_drive_the_stop(tmp_path, requires_compiler):
    """The compiled solver stops where the oracle does at any runtime
    check interval and max_iter, and reports the objective at the stop."""
    case = build_ista_objective()
    result = _generate(case, tmp_path, ISTA_GOLDEN)
    assert result.translate.success, result.translate.log
    build_lib(tmp_path)
    solver = _load_wrapper(tmp_path).Solver()
    rng = np.random.default_rng(7)
    for check in (1, 7, 25):
        for max_iter in (200, 10):
            solver.update_settings(check_termination=check, max_iter=max_iter)
            oracle = ReferenceOracle(
                with_settings(result.ir, check_termination=check, max_iters=max_iter)
            )
            for _ in range(4):
                b_val = case["data"]["b"] + rng.uniform(-1.0, 1.0, size=case["data"]["m"])
                out = solver.solve(p0=b_val)
                expected = oracle.solve(param_values={"p0": b_val})
                assert out["iter"] == expected.iters
                assert out["status"] == (0 if expected.solved else 1)
                assert out["residual"] == pytest.approx(expected.residual, rel=1e-9, abs=1e-15)
                np.testing.assert_allclose(out["x"], expected.solution, atol=1e-12)
    # A runtime tolerance moves the stop as well.
    solver.update_settings(check_termination=1, max_iter=200, tol=1e-2)
    b_val = case["data"]["b"]
    expected = ReferenceOracle(with_settings(result.ir, check_termination=1, tol=1e-2)).solve(
        param_values={"p0": b_val}
    )
    assert solver.solve(p0=b_val)["iter"] == expected.iters


def test_check_termination_zero_tests_only_the_last_iteration(tmp_path, requires_compiler):
    """0 disables early exit but still evaluates the objective at max_iter, so
    the generated default must be 0, not clamped to 1."""
    case = build_ista_objective(check_termination=0, max_iters=40)
    result = _generate(case, tmp_path, ISTA_GOLDEN)
    assert result.translate.success, result.translate.log
    header = (tmp_path / "include" / "fom_workspace.h").read_text()
    assert "#define FOM_CHECK_TERMINATION 0" in header
    res = ReferenceOracle(result.ir).solve(param_values={"p0": case["data"]["b"]})
    assert res.iters == 40 and [k for k, _ in res.objectives] == [40]


def test_float32_rolled_ista_verifies(tmp_path, requires_compiler):
    # 1e-4 keeps every checked objective well clear of the single-precision
    # rounding of the threshold test.
    case = build_ista_objective(dtype="float32", tol=1e-4)
    result = _generate(case, tmp_path, ISTA_GOLDEN)
    assert result.translate.success, result.translate.log
    assert "typedef float fom_float;" in (tmp_path / "include" / "fom_workspace.h").read_text()


@pytest.mark.parametrize(
    "version, backend", [(Version.EMBEDDED, Backend.C), (Version.LARGE_SCALE, Backend.OPENMP)]
)
def test_scheduled_objective_verifies(tmp_path, requires_compiler, version, backend):
    case = build_scheduled_objective(extrapolated=True, version=version, backend=backend)
    result = generate_code(
        case["build"],
        out_dir=tmp_path,
        config=case["config"],
        driver=StubAgentDriver(SCHEDULED_GOLDEN),
        name="scheduled",
        schedules=case["schedules"],
    )
    assert result.translate.success, result.translate.log
    assert result.ir.is_scheduled


@pytest.mark.parametrize(
    "version, backend", [(Version.EMBEDDED, Backend.C), (Version.LARGE_SCALE, Backend.OPENMP)]
)
def test_unrolled_objective_verifies(tmp_path, requires_compiler, version, backend):
    ex = fista_example()
    v, its = ex["verifier"], ex["iterates"]
    v.set_objective_norm(its[-1] - its[-2], np.inf)
    result = generate_code(
        v,
        None,
        tmp_path,
        config=objective_config(tol=1e-3, version=version, backend=backend),
        driver=StubAgentDriver(FISTA_GOLDEN),
        name="fista",
        iterates=its,
    )
    assert result.translate.success, result.translate.log
    assert result.ir.unrolled


def test_tape_that_rolls_stops_within_the_recorded_horizon(tmp_path, requires_compiler):
    """Generated from the tape, a stationary method rolls back up with the
    recorded horizon as max_iter: the certified worst-case stopping time."""
    case = build_ista(K=30)
    its = case["iterates"]
    case["verifier"].set_objective_norm(its[-1] - its[-2], np.inf)
    config = objective_config(tol=1e-6, check_termination=4)
    result = generate_code(
        case["verifier"],
        None,
        tmp_path,
        config=config,
        driver=StubAgentDriver(ISTA_GOLDEN),
        name="ista",
        iterates=its,
    )
    assert result.translate.success, result.translate.log
    assert not result.ir.unrolled and result.ir.termination.max_iters == 30
    res = ReferenceOracle(result.ir).solve(param_values={"p0": case["data"]["b"]})
    assert res.iters <= 30


@pytest.mark.parametrize("norm", ["inf", "2"])
def test_openmp_reductions_compile(tmp_path, requires_compiler, norm):
    """The objective's OpenMP reductions are real pragmas, not just comments:
    compile the objective source with OpenMP on (front end only where the
    toolchain has no runtime, as Apple clang)."""
    import subprocess

    case = build_scheduled_objective(
        extrapolated=True,
        ord={"inf": np.inf, "2": 2}[norm],
        version=Version.LARGE_SCALE,
        backend=Backend.OPENMP,
    )
    ir = extract_scheduled_ir(case["build"], case["schedules"], config=case["config"])
    scaffold(ir, tmp_path, case["config"])
    obj = (tmp_path / "src" / "fom_objective.c").read_text()
    assert ("reduction(max : r)" in obj) == (norm == "inf")
    for flags in (["-fopenmp"], ["-Xclang", "-fopenmp"]):
        cmd = [requires_compiler, "-Werror", "-Wall", "-Iinclude", *flags, "-c"]
        cmd += ["src/fom_objective.c"]
        proc = subprocess.run(cmd + ["-o", "obj.o"], cwd=tmp_path, capture_output=True, text=True)
        if proc.returncode == 0:
            return
    pytest.skip(f"no OpenMP-capable compiler front end: {proc.stderr[-300:]}")


def test_verify_rejects_a_solver_that_never_calls_fom_objective(tmp_path, requires_compiler):
    case = build_ista_objective()
    result = _generate(case, tmp_path, PLAIN_ISTA_GOLDEN)
    assert not result.translate.success
    assert "never calls fom_objective" in result.translate.log


@pytest.mark.parametrize(
    "edit, message",
    [
        # Ignores the runtime interval: the other check intervals stop elsewhere.
        (
            (
                "const fom_int check = work->settings.check_termination;",
                "const fom_int check = FOM_CHECK_TERMINATION;",
            ),
            "check_termination=1",
        ),
        # Skips the test at the last iteration: the capped run reports a stale value.
        (("|| it + 1 == max_iter", ""), "max_iter="),
    ],
)
def test_verify_rejects_a_solver_that_bends_the_protocol(
    tmp_path, requires_compiler, edit, message
):
    case = build_ista_objective()
    source = ISTA_GOLDEN.read_text().replace(*edit)
    assert source != ISTA_GOLDEN.read_text()
    result = _generate(case, tmp_path, source)
    assert not result.translate.success
    assert message in result.translate.log
    assert "VERIFY FAILED" in result.translate.log


def test_verify_passes_after_scaffold_restore(tmp_path, requires_compiler):
    """Re-scaffolding (as the pipeline does before every verify) keeps the
    objective artifacts deterministic."""
    case = build_ista_objective()
    result = _generate(case, tmp_path, ISTA_GOLDEN)
    before = (tmp_path / "src" / "fom_objective.c").read_text()
    result.target.scaffold(result.ir, tmp_path)
    assert (tmp_path / "src" / "fom_objective.c").read_text() == before
    assert build_and_verify(tmp_path, result.target).success


def test_the_optimizer_cannot_rewrite_the_objective(tmp_path, requires_compiler):
    """fom_ops.c is editable in the code optimization phase, fom_objective.c is
    not: a rewrite that makes the objective stop at once is discarded before
    verify, and the shipped objective is the generated one."""
    case = build_ista_objective()
    rel = "src/fom_objective.c"

    def stop_at_once(text: str) -> str:
        return text.replace("    return r;", "    return 0.0;")

    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(ISTA_GOLDEN, extra_sources={rel: stop_at_once}),
        name="ista",
        code_optimization=True,
        **gate(case),
    )
    assert result.success, result.optimize.log
    # Restored before verify ran: the tampered objective would stop at once.
    assert "VERIFY OK" in result.optimize.log
    generated = (tmp_path / rel).read_text()
    assert "return r;" in generated and stop_at_once(generated) != generated
    assert rel not in result.optimization_diff
    assert not (tmp_path / "release" / "fom_objective.c").exists()


# --------------------------------------------------------------------------
# hardware runs: a GPU, and threaded OpenMP (skipped where unavailable; run on
# the cluster through slurm/codegen_hw.slurm)
# --------------------------------------------------------------------------
CUDA_GOLDENS = {
    "ista": GOLDEN / "ista_objective" / "large_scale_cuda" / "fom_solver.cu",
    "scheduled": GOLDEN / "scheduled_objective" / "large_scale_cuda" / "fom_solver.cu",
}


def _hw_cases(backend):
    """(label, generate_code args) for the rolled ISTA, a large ISTA whose
    objective has enough rows to spread over threads / thread blocks, and the
    scheduled objective in both reduction kinds (max and sum)."""
    hw = dict(version=Version.LARGE_SCALE, backend=backend)
    small = build_ista_objective(**hw)
    large = build_ista_objective(**hw)
    large_ista = build_ista(K=3, n=300, m=600, seed=3)
    its = large_ista["iterates"]
    large_ista["verifier"].set_objective_norm(its[-1] - its[-2], np.inf)
    large.update(
        {k: large_ista[k] for k in ("verifier", "next_state", "iterates", "param", "data")}
    )
    out = [("ista", small), ("ista_large", large)]
    for ord, label in ((np.inf, "scheduled_inf"), (2, "scheduled_sq")):
        out.append((label, build_scheduled_objective(extrapolated=True, ord=ord, **hw)))
    return out


def _generate_hw(case, out_dir, golden):
    if "build" in case:
        return generate_code(
            case["build"],
            out_dir=out_dir,
            config=case["config"],
            driver=StubAgentDriver(golden),
            name="scheduled",
            schedules=case["schedules"],
        )
    return _generate(case, out_dir, golden)


def _openmp_links(compiler, tmp_path) -> bool:
    src = tmp_path / "omp_probe.c"
    src.write_text(
        "#include <omp.h>\nint main(void) { return omp_get_max_threads() > 0 ? 0 : 1; }\n"
    )
    import subprocess

    proc = subprocess.run(
        [compiler, "-fopenmp", str(src), "-o", str(tmp_path / "omp_probe")], capture_output=True
    )
    return proc.returncode == 0


@pytest.mark.parametrize("threads", [1, 4, 16])
def test_threaded_openmp_objective_verifies(tmp_path, requires_compiler, monkeypatch, threads):
    """The fom_objective reductions give the oracle's stop with threads on."""
    if not _openmp_links(requires_compiler, tmp_path):
        pytest.skip("the C compiler cannot build and link with -fopenmp")
    monkeypatch.setenv("OMPFLAGS", "-fopenmp")  # the Makefile reads OMPFLAGS ?=
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    for label, case in _hw_cases(Backend.OPENMP):
        golden = SCHEDULED_GOLDEN if "build" in case else ISTA_GOLDEN
        result = _generate_hw(case, tmp_path / label, golden)
        assert result.translate.success, (label, result.translate.log)
        assert "-fopenmp" in result.translate.log, label


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs nvcc (CUDA toolkit)")
def test_cuda_objective_builds_and_verifies(tmp_path):
    """Builds every objective case with nvcc; on a GPU (detected, or forced
    with FOMTOOLS_CUDA_RUN=1) the verify harness also runs, runtime cases
    included."""
    on_gpu = CudaTarget._can_run_on_gpu()
    for label, case in _hw_cases(Backend.CUDA):
        golden = CUDA_GOLDENS["scheduled" if "build" in case else "ista"]
        result = _generate_hw(case, tmp_path / label, golden)
        assert result.translate.success, (label, result.translate.log)
        if on_gpu:
            assert "VERIFY OK" in result.translate.log, (label, result.translate.log)
