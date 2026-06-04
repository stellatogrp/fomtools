"""Saturated-linear (vector clip bounds) + the MPC/OSQP example.

box-QP exercises the vector-bound saturated-linear step end-to-end (golden C);
the MPC example checks the OSQP-ADMM FOM extracts correctly and converges to the
QP solution.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.config import CodegenConfig, TerminationSpec
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.harness import build_lib
from fomtools.codegen.ir import ActivationKind
from fomtools.codegen.oracle import run_faithful
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import boxqp_reference, build_boxqp

BOXQP_GOLDEN = Path(__file__).parent / "golden" / "boxqp" / "embedded_c" / "fom_solver.c"


# --- box-QP: vector saturated-linear --------------------------------------
def test_boxqp_ir_has_vector_clip():
    case = build_boxqp()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="boxqp")
    assert len(ir.body) == 1
    act = ir.body[0].activation
    assert act.kind == ActivationKind.SATURATED_LINEAR
    # Bounds are stored as per-coordinate IR arrays, not scalars.
    assert act.lower is not None and act.upper is not None
    assert ir.matrices[act.lower].array.shape == (case["data"]["n"],)
    assert np.allclose(ir.matrices[act.lower].array, case["data"]["lo"])


def test_boxqp_oracle_matches_reference():
    case = build_boxqp(max_iters=300)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    q = case["data"]["q0"]
    res = ReferenceOracle(ir).solve(param_values={"p0": q})
    ref = boxqp_reference(case["data"], q, ir.state.init, n_iters=ir.termination.max_iters)
    assert np.allclose(res.solution, ref[-1], atol=1e-9)
    # and matches the faithful step-level evaluation
    traj = run_faithful(
        case["verifier"], case["next_state"], {case["param"]: q}, n_iters=ir.termination.max_iters
    )
    assert np.allclose(res.solution, traj[-1], atol=1e-9)


def test_boxqp_golden_static_checks():
    target = EmbeddedCTarget(build_boxqp()["config"])
    assert target.static_checks(BOXQP_GOLDEN.read_text()) == []


def test_boxqp_pipeline_and_wrapper(tmp_path, requires_compiler):
    case = build_boxqp(max_iters=300)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(BOXQP_GOLDEN),
        name="boxqp",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    build_lib(tmp_path)
    spec = importlib.util.spec_from_file_location("w_box", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)
    rng = np.random.default_rng(0)
    for _ in range(4):
        q = case["data"]["q0"] + rng.uniform(-0.5, 0.5, size=case["data"]["n"])
        out = solver.solve(p0=q)
        assert np.allclose(out["x"], oracle.solve(param_values={"p0": q}).solution, atol=1e-6)


# --- MPC / OSQP -----------------------------------------------------------
def test_mpc_ir_structure():
    from fomtools.codegen.examples import mpc_osqp_example

    ex = mpc_osqp_example(nx=3, nu=2, N=5)
    d = ex["data"]
    ir = extract_ir(ex["verifier"], ex["next_state"], config=CodegenConfig(), name="mpc")
    assert ir.state_dim == d["nv"] + 2 * d["nc"]
    kinds = [s.activation.kind for s in ir.body]
    assert ActivationKind.IDENTITY in kinds  # KKT solve (factorized; no inverse)
    assert ActivationKind.SATURATED_LINEAR in kinds  # box projection
    assert any(s.solve is not None for s in ir.body)  # affine lowered to a solve


def test_mpc_admm_converges_to_qp_solution():
    cp = pytest.importorskip("cvxpy")
    from fomtools.codegen.examples import mpc_osqp_example

    ex = mpc_osqp_example(nx=3, nu=2, N=5)
    d = ex["data"]
    ir = extract_ir(
        ex["verifier"],
        ex["next_state"],
        config=CodegenConfig(termination=TerminationSpec(max_iters=4000)),
        name="mpc",
    )
    q = d["q_nom"]
    res = ReferenceOracle(ir).solve(param_values={"p0": q})
    x_admm = res.solution[: d["nv"]]

    z = cp.Variable(d["nv"])
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(d["P"])) + q @ z),
        [d["M"] @ z >= d["l"], d["M"] @ z <= d["u"]],
    )
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal"
    assert np.linalg.norm(x_admm - z.value, np.inf) < 1e-2


# --- factorized solve: never an inverse -----------------------------------
def test_affine_lowered_to_factorized_solve_not_inverse():
    from fomtools.codegen.examples import mpc_osqp_example

    ex = mpc_osqp_example(nx=3, nu=2, N=4)
    ir = extract_ir(ex["verifier"], ex["next_state"], config=CodegenConfig(), name="mpc")
    # The affine step carries a factorization and a solve, not a folded inverse.
    affine = next(s for s in ir.body if s.solve is not None)
    fac = ir.factorizations[affine.solve]
    assert fac.method == "ldlt"  # symmetric KKT -> LDL^T
    assert fac.nnz < fac.dim**2  # genuinely sparse, not a dense inverse
    # The linear op assembles the *original* rhs (its state block is identity-like
    # selection of the KKT rhs), never M^{-1}.
    assert affine.activation.kind == ActivationKind.IDENTITY


def test_emitted_linsolve_is_division_free(tmp_path):
    from fomtools.codegen.emit import scaffold
    from fomtools.codegen.examples import mpc_osqp_example
    from fomtools.codegen.targets.base import strip_c_comments_and_strings

    ex = mpc_osqp_example(nx=3, nu=2, N=4)
    ir = extract_ir(ex["verifier"], ex["next_state"], config=CodegenConfig(), name="mpc")
    scaffold(ir, tmp_path, CodegenConfig())
    src = (tmp_path / "src" / "fom_linsolve.c").read_text()
    code = strip_c_comments_and_strings(src)
    assert "/" not in code  # reciprocal diagonals, no division
    assert "fom_lin_solve_t0" in src
    # No dense inverse is emitted anywhere.
    for f in (tmp_path / "src").glob("*.c"):
        assert "inv(" not in f.read_text().lower()


def test_structured_matvec_emitted_sparse(tmp_path):
    from fomtools.codegen.emit import scaffold
    from fomtools.codegen.examples import mpc_osqp_example
    from fomtools.codegen.targets.base import strip_c_comments_and_strings

    ex = mpc_osqp_example(nx=4, nu=2, N=8)
    ir = extract_ir(ex["verifier"], ex["next_state"], config=CodegenConfig(), name="mpc")
    scaffold(ir, tmp_path, CodegenConfig())
    ops = (tmp_path / "src" / "fom_ops.c").read_text()
    data = (tmp_path / "src" / "fom_data.c").read_text()
    # The structured OSQP matrices are emitted as CSR (rowptr), not dense, and
    # are NOT duplicated as dense FOM_MAT arrays in fom_data.c.
    assert "FOM_OP_T0_STATE_ROWPTR" in ops
    assert "fom_apply_t0_state" in ops
    assert "FOM_MAT_t0_state[" not in data
    # The matvec routines are division-free.
    assert "/" not in strip_c_comments_and_strings(ops)


def test_mpc_golden_pipeline_and_wrapper(tmp_path, requires_compiler):
    import importlib.util

    from fomtools.codegen.examples import mpc_osqp_example

    golden = Path(__file__).parent / "golden" / "mpc" / "embedded_c" / "fom_solver.c"
    ex = mpc_osqp_example(nx=3, nu=2, N=5)
    result = generate_code(
        ex["verifier"],
        ex["next_state"],
        tmp_path,
        config=CodegenConfig(termination=TerminationSpec(max_iters=400)),
        driver=StubAgentDriver(golden),
        name="mpc",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log

    build_lib(tmp_path)
    spec = importlib.util.spec_from_file_location("w_mpc", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)
    q = ex["data"]["q_nom"]
    out = solver.solve(p0=q)
    assert np.allclose(out["x"], oracle.solve(param_values={"p0": q}).solution, atol=1e-6)
