"""CVXPYgen builders stop on OSQP's unscaled residuals.

Each built-in builder sets ``max(||Ax - clamp(Ax, l, u)||_inf, ||Px + q +
A'y||_inf)`` as its verification objective, so ``TerminationKind.OBJECTIVE``
is OSQP's test with ``eps_abs = tol``, ``eps_rel = 0`` and ``scaled_termination
= False``. These tests check the objective against the residuals computed
directly (as the C bridge does), in the original units when the QP is
equilibrated, and run the generated solver end to end.
"""

import dataclasses
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as spa

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.codegen.agent.stub import StubAgentDriver  # noqa: E402
from fomtools.codegen.config import (  # noqa: E402
    Backend,
    CodegenConfig,
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.extract_scheduled import extract_scheduled_ir  # noqa: E402
from fomtools.codegen.generate import extract_fom_ir  # noqa: E402
from fomtools.codegen.ir import ActivationKind  # noqa: E402
from fomtools.codegen.oracle import ReferenceOracle  # noqa: E402
from fomtools.cvxpygen import generate_solver, interface  # noqa: E402
from fomtools.cvxpygen.builders import (  # noqa: E402
    build_admm_qp,
    build_pdhg_lp,
    build_relaxed_admm_qp,
)
from fomtools.cvxpygen.qpdata import CVXPYGEN_INF  # noqa: E402
from tests.codegen.test_cvxpygen_dispatch import _admm_qp, _pdhg_lp  # noqa: E402
from tests.codegen.test_cvxpygen_e2e import _build_problem  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "cvxpygen_qp_objective" / "embedded_c" / "fom_solver.c"


def objective_config(max_iters=400, tol=1e-6, check_termination=None, **kw):
    return CodegenConfig(
        termination=TerminationSpec(
            kind=TerminationKind.OBJECTIVE,
            max_iters=max_iters,
            tol=tol,
            check_termination=check_termination,
        ),
        **kw,
    )


def bridge_residuals(qp, pieces, x, y):
    """``(pri_res, dua_res)`` exactly as the generated bridge computes them
    (``fomtools.cvxpygen.glue._residual_block``), on the canonical data."""
    n_eq = qp.n_eq
    l = np.concatenate([pieces["b"], np.full(qp.n_ineq, -CVXPYGEN_INF)])
    u = np.concatenate([pieces["b"], pieces["u_ineq"]])
    Ax = qp.A @ x
    viol = np.maximum(Ax - u, l - Ax)
    pri = max(0.0, float(np.max(viol))) if qp.m else 0.0
    dua = float(np.max(np.abs(qp.P @ x + pieces["q"] + qp.A.T @ y)))
    assert n_eq == len(pieces["b"])
    return pri, dua


def draw_pieces(qp, rng):
    """Canonical vectors, with the parametric ones drawn inside their boxes."""
    pieces = {"q": qp.q, "b": qp.b, "u_ineq": qp.u_ineq}
    for piece, (lb, ub) in qp.bounds.items():
        pieces[piece] = rng.uniform(lb, ub)
    return pieces


def param_values(ir, param_copy, pieces):
    return {p.name: pieces[piece] for p, piece in zip(ir.params, param_copy)}


def rolled_ir(spec, config):
    return extract_fom_ir(
        spec.verifier,
        spec.next_state,
        config=config,
        state=list(spec.state),
        params=list(spec.params),
        name=spec.name,
        iterates=list(spec.iterates),
        param_values=spec.param_values,
    )


def check_trajectory(ir, spec, qp, pieces, unscale=None, unscaled_qp=None, unscaled=None):
    """Every r^k the oracle tests equals the bridge's max(pri_res, dua_res)."""
    oracle = ReferenceOracle(
        dataclasses.replace(
            ir, termination=ir.termination.model_copy(update={"check_termination": 1})
        )
    )
    res = oracle.solve(param_values=param_values(ir, spec.param_copy, pieces))
    assert len(res.objectives) > 5
    for k, value in res.objectives:
        state = res.trajectory[k]
        x = state[spec.primal_offset : spec.primal_offset + spec.primal_dim]
        y = state[spec.dual_offset : spec.dual_offset + spec.dual_dim]
        if unscale is not None:
            D, E, c = unscale
            x, y = D * x, E * y / c  # the bridge's unscaling of the solution
        pri, dua = bridge_residuals(unscaled_qp or qp, unscaled or pieces, x, y)
        assert value == pytest.approx(max(pri, dua), rel=1e-9, abs=1e-12), k
    return res


# --------------------------------------------------------------------------
# the objective is OSQP's residual
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "build, make_qp", [(build_admm_qp, _admm_qp), (build_pdhg_lp, _pdhg_lp)], ids=["admm", "pdhg"]
)
def test_builder_objective_is_the_bridge_residual(build, make_qp):
    qp = make_qp()
    spec = build(qp)
    ir = rolled_ir(spec, objective_config(tol=1e-9))
    assert ir.objective.norm == "inf"
    assert ActivationKind.RELU in [b.activation.kind for b in ir.objective.blocks]
    rng = np.random.default_rng(0)
    for _ in range(3):
        check_trajectory(ir, spec, qp, draw_pieces(qp, rng))


def test_scheduled_builder_objective_is_the_bridge_residual():
    qp = _admm_qp()
    spec = build_relaxed_admm_qp(qp, ramp_iters=20)
    ir = extract_scheduled_ir(spec.build, spec.schedules, config=objective_config(tol=1e-9))
    # The relaxation enters the iterate the objective reads, so it is scheduled.
    assert any(t.scale for b in ir.objective.blocks for t in b.linear_op.terms)
    check_trajectory(ir, spec, qp, draw_pieces(qp, np.random.default_rng(1)))


def _equilibrated(qp, rng):
    """A positively-scaled copy of ``qp`` carrying its ``(D, E, c)``, as Ruiz
    equilibration produces: ``P~ = c D P D``, ``q~ = c D q``, ``A~ = E A D``."""
    D = rng.uniform(0.3, 3.0, qp.n)
    E = rng.uniform(0.3, 3.0, qp.m)
    c = 0.37
    scale = {"q": c * D, "b": E[: qp.n_eq], "u_ineq": E[qp.n_eq :]}
    return dataclasses.replace(
        qp,
        P=spa.csc_matrix(c * spa.diags(D) @ qp.P @ spa.diags(D)),
        q=c * D * qp.q,
        A=spa.csc_matrix(spa.diags(E) @ qp.A @ spa.diags(D)),
        b=scale["b"] * qp.b,
        u_ineq=scale["u_ineq"] * qp.u_ineq,
        bounds={k: (scale[k] * lb, scale[k] * ub) for k, (lb, ub) in qp.bounds.items()},
        unscale=(D, E, c),
    ), scale


def test_objective_is_measured_in_the_original_units():
    qp = _admm_qp(seed=4)
    scaled, scale = _equilibrated(qp, np.random.default_rng(2))
    spec = build_admm_qp(scaled)
    ir = rolled_ir(spec, objective_config(tol=1e-9))
    rng = np.random.default_rng(3)
    for _ in range(3):
        original = draw_pieces(qp, rng)
        pieces = {k: scale[k] * v for k, v in original.items()}
        check_trajectory(
            ir, spec, scaled, pieces, unscale=scaled.unscale, unscaled_qp=qp, unscaled=original
        )


def test_stop_agrees_with_the_bridge_residuals():
    """At the oracle's stop, the residuals the bridge reports are within tol
    exactly when the solver says FOM_SOLVED, and their max is the reported
    objective."""
    qp = _admm_qp()
    spec = build_admm_qp(qp)
    ir = rolled_ir(spec, objective_config(max_iters=2000, tol=1e-5))
    rng = np.random.default_rng(5)
    for _ in range(4):
        # A drawn equality right-hand side can make the QP infeasible (the
        # residual then plateaus, as it should); keep the feasible nominal one.
        pieces = {**draw_pieces(qp, rng), "b": qp.b}
        res = ReferenceOracle(ir).solve(param_values=param_values(ir, spec.param_copy, pieces))
        x = res.solution[spec.primal_offset : spec.primal_offset + spec.primal_dim]
        y = res.solution[spec.dual_offset : spec.dual_offset + spec.dual_dim]
        pri, dua = bridge_residuals(qp, pieces, x, y)
        assert res.residual == pytest.approx(max(pri, dua), rel=1e-9, abs=1e-12)
        assert res.solved == (max(pri, dua) <= 1e-5)
        assert res.solved and res.iters % 25 == 0


def test_runtime_settings_for_the_objective_kind():
    settings = interface._settings_for(
        "rolled", TerminationSpec(kind=TerminationKind.OBJECTIVE, max_iters=300, tol=1e-5)
    )
    assert set(settings) == {"max_iter", "tol", "check_termination"}
    assert settings["check_termination"].default == "25"
    assert float(settings["tol"].default) == 1e-5
    assert interface._settings_for("unrolled", TerminationSpec()) == {}


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
@pytest.mark.requires_compiler
@pytest.mark.parametrize(
    "version, backend", [(Version.EMBEDDED, Backend.C), (Version.LARGE_SCALE, Backend.OPENMP)]
)
@pytest.mark.parametrize("check_termination", [1, 7, 25])
def test_generated_solver_verifies_with_objective_termination(
    requires_compiler, tmp_path, monkeypatch, version, backend, check_termination
):
    monkeypatch.chdir(tmp_path)
    problem, q, x = _build_problem()
    result = generate_solver(
        problem,
        "cpg_obj",
        fom="admm",
        param_bounds={q: (-1.0, 1.0)},
        config=objective_config(
            max_iters=4000,
            tol=1e-7,
            check_termination=check_termination,
            version=version,
            backend=backend,
        ),
        driver=StubAgentDriver(GOLDEN),
        wrapper=False,
    )
    assert result.success, result.translate.log
    assert result.ir.termination.uses_objective
    cpg_solve = (tmp_path / "cpg_obj" / "c" / "src" / "cpg_solve.c").read_text()
    assert "cpg_set_solver_tol" in cpg_solve
    assert "cpg_set_solver_check_termination" in cpg_solve


@pytest.mark.requires_compiler
def test_cvxpy_solve_stops_on_the_residuals(requires_compiler, tmp_path, monkeypatch):
    """Through the compiled Python wrapper: the solve stops where the oracle
    does, and the residuals the bridge reports are the objective at the stop."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")
    if importlib.util.find_spec("pip") is None:
        pytest.skip("pip not available (cvxpygen's wrapper build runs 'python -m pip')")
    monkeypatch.chdir(tmp_path)
    problem, q, x = _build_problem()
    result = generate_solver(
        problem,
        "cpg_obj_wrap",
        fom="admm",
        param_bounds={q: (-1.0, 1.0)},
        config=objective_config(max_iters=4000, tol=1e-6),
        driver=StubAgentDriver(GOLDEN),
        wrapper=True,
        # The extension's pybind types are named after the prefix; another test
        # in the same process may already have registered the unprefixed ones.
        prefix="objterm",
    )
    assert result.success
    for q_val, tol, check in (
        (0.1 * np.ones(3), 1e-6, 25),
        (np.array([0.8, -0.5, 0.3]), 1e-6, 7),
        (np.array([0.8, -0.5, 0.3]), 1e-3, 1),
    ):
        q.value = q_val
        problem.solve(method="CPG", tol=tol, check_termination=check)
        stats = problem.solver_stats.extra_stats
        iters = problem.solver_stats.num_iters
        expected = ReferenceOracle(
            dataclasses.replace(
                result.ir,
                termination=result.ir.termination.model_copy(
                    update={"tol": tol, "check_termination": check}
                ),
            )
        ).solve(param_values={"p0": q_val})
        assert iters == expected.iters
        assert max(stats["pri_res"], stats["dua_res"]) == pytest.approx(
            expected.residual, rel=1e-6, abs=1e-12
        )
        assert max(stats["pri_res"], stats["dua_res"]) <= tol
