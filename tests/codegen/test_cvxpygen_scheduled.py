"""Math gate for the scheduled built-in: relaxed ADMM.

Extracts the IR through the scheduled path and runs the pure-NumPy reference
oracle; the fixed point must match what CVXPY computes for the same problem. No
C compiler, no code generation, no agent involved.

The tests also pin the two things that make the scheduled path worth having: the
relaxation really is read from a table at the loop counter (freezing it changes
the trajectory), and a canonical *constant* reaching the state update does not
break extraction — the trap that rules out anchored methods here, since
``extract_scheduled`` cannot scale a constant offset by a schedule.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.codegen.config import CodegenConfig, TerminationSpec  # noqa: E402
from fomtools.codegen.extract_scheduled import extract_scheduled_ir  # noqa: E402
from fomtools.codegen.oracle import ReferenceOracle  # noqa: E402
from fomtools.codegen.spec import render_markdown  # noqa: E402
from fomtools.cvxpygen.builders import (  # noqa: E402
    BUILDERS,
    ScheduledFomSpec,
    build_relaxed_admm_qp,
    fom_kind,
)
from fomtools.cvxpygen.qpdata import extract_qp_data  # noqa: E402

from .conftest import cvxpygen_canonicalize  # noqa: E402


def _parametric_qp(seed=0, n=4, m_eq=2):
    """Strongly convex QP; both the linear cost and the equality rhs vary."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m_eq, n))
    q = cp.Parameter(n, name="q")
    beq = cp.Parameter(m_eq, name="beq")
    x = cp.Variable(n, name="x")
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
        [A @ x == beq, x <= 1.0, -1.0 <= x],
    )
    q.value = 0.1 * rng.standard_normal(n)
    beq.value = A @ rng.uniform(-0.3, 0.3, size=n)
    canon, _ = cvxpygen_canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={q: (-1.0, 1.0), beq: (-2.0, 2.0)})
    return problem, qp, x


def _constant_rhs_lp(seed=3, n=4, m_eq=2):
    """An LP whose equality rhs is a plain constant, not a cvxpy Parameter.

    That constant lands in the canonical state update, which is exactly what an
    anchored (Halpern) schedule cannot express: scaling it by the schedule makes
    the *constant offset* iteration-varying, which extraction refuses. Relaxation
    leaves it alone, so this case has to keep working.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m_eq, n))
    c = cp.Parameter(n, name="c")
    c.value = rng.standard_normal(n)
    x = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(c @ x),
        [A @ x == A @ rng.uniform(-0.2, 0.2, size=n), x <= 1.0, -1.0 <= x],
    )
    canon, _ = cvxpygen_canonicalize(problem)
    return problem, extract_qp_data(canon, param_bounds={c: (-2.0, 2.0)}), x


def _extract(spec, max_iters):
    return extract_scheduled_ir(
        spec.build,
        spec.schedules,
        config=CodegenConfig(termination=TerminationSpec(max_iters=max_iters)),
        name=spec.name,
    )


def _solve(spec, qp, max_iters):
    """Run the extracted scheduled FOM; returns the oracle result."""
    ir = _extract(spec, max_iters)
    values = {"q": qp.q, "b": qp.b, "u_ineq": qp.u_ineq}
    param_values = {f"p{i}": values[piece] for i, piece in enumerate(spec.param_copy)}
    return ReferenceOracle(ir).solve(param_values=param_values)


def _run(spec, qp, max_iters):
    """Run the extracted scheduled FOM and return its primal / dual iterates."""
    solution = _solve(spec, qp, max_iters).solution
    x = solution[spec.primal_offset : spec.primal_offset + spec.primal_dim]
    y = solution[spec.dual_offset : spec.dual_offset + spec.dual_dim]
    return x, y


# --- the method solves the problem ------------------------------------------


def test_relaxed_admm_matches_cvxpy_qp():
    problem, qp, x = _parametric_qp()
    spec = build_relaxed_admm_qp(qp)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, y_fom = _run(spec, qp, max_iters=2000)

    np.testing.assert_allclose(x_fom, x.value, atol=1e-5)
    # The dual is the canonical one, so check stationarity rather than matching
    # CVXPY's per-constraint duals directly.
    np.testing.assert_allclose(qp.P @ x_fom + qp.q + qp.A.T @ y_fom, 0.0, atol=1e-5)


def test_relaxed_admm_handles_a_constant_equality_rhs():
    """The canonical constant stays unscaled, so extraction succeeds."""
    problem, qp, x = _constant_rhs_lp()
    spec = build_relaxed_admm_qp(qp)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, _ = _run(spec, qp, max_iters=4000)

    np.testing.assert_allclose(x_fom, x.value, atol=1e-4)


def test_a_constant_relaxation_solves_it_too():
    """Any fixed relaxation in (0, 2) is valid, so the ramp is a choice not a crutch."""
    problem, qp, x = _parametric_qp()
    spec = build_relaxed_admm_qp(qp, alpha_start=1.6, alpha_final=1.6)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, _ = _run(spec, qp, max_iters=2000)

    np.testing.assert_allclose(x_fom, x.value, atol=1e-5)


# --- the table is live ------------------------------------------------------


def test_the_emitted_table_drives_the_method():
    """Freeze the relaxation and the trajectory changes: the table is read, not baked.

    Compared along the trajectory rather than at the end, because both are valid
    ADMM and settle on the same solution — on this QP within a dozen iterations,
    which would leave two visibly different methods agreeing to 1e-12.
    """
    _, qp, _ = _parametric_qp()
    ramped = build_relaxed_admm_qp(qp)
    frozen = build_relaxed_admm_qp(qp, alpha_start=1.9, alpha_final=1.9)

    traj_ramped = _solve(ramped, qp, max_iters=5).trajectory
    traj_frozen = _solve(frozen, qp, max_iters=5).trajectory

    gaps = [np.max(np.abs(a - b)) for a, b in zip(traj_ramped, traj_frozen)]
    assert max(gaps) > 1e-6, "a frozen relaxation would be the same method"


def test_freezing_the_table_at_its_first_value_is_what_the_rolled_path_would_do():
    """The failure mode the scheduled path exists to prevent, made concrete.

    Recording this method down the rolled path would bake in ``alpha_0``; that is
    a different algorithm, and here it is measurably a different trajectory.
    """
    _, qp, _ = _parametric_qp()
    ramped = build_relaxed_admm_qp(qp, alpha_start=1.0, alpha_final=1.9, ramp_iters=50)
    as_if_rolled = build_relaxed_admm_qp(qp, alpha_start=1.0, alpha_final=1.0)

    traj = _solve(ramped, qp, max_iters=5).trajectory
    traj_rolled = _solve(as_if_rolled, qp, max_iters=5).trajectory

    gaps = [np.max(np.abs(a - b)) for a, b in zip(traj, traj_rolled)]
    assert max(gaps) > 1e-9


def test_the_table_is_emitted_with_one_value_per_iteration():
    _, qp, _ = _parametric_qp()
    spec = build_relaxed_admm_qp(qp, alpha_start=1.0, alpha_final=1.9, ramp_iters=50)
    ir = _extract(spec, max_iters=200)

    table = ir.schedules["relaxation"]
    assert len(table) == 200
    assert table[0] == pytest.approx(1.0)
    assert table[50] == pytest.approx(1.9), "the ramp reaches alpha_final at ramp_iters"
    assert table[-1] == pytest.approx(1.9), "and holds there"


def test_the_solver_stays_one_block_whatever_the_horizon():
    """The point of declaring a schedule rather than generating the run."""
    _, qp, _ = _parametric_qp()
    spec = build_relaxed_admm_qp(qp)

    short = _extract(spec, max_iters=20)
    long = _extract(spec, max_iters=5000)

    assert len(short.body) == len(long.body) == 2  # KKT solve + projection
    assert short.state.slots == long.state.slots, "and the same three state slots"
    assert len(long.schedules["relaxation"]) == 5000


def test_the_spec_tells_the_agent_to_read_the_table():
    _, qp, _ = _parametric_qp()
    doc = render_markdown(_extract(build_relaxed_admm_qp(qp), max_iters=100))

    assert "fom_sched_relaxation(it)" in doc
    assert "0-based" in doc


# --- registration and guards -------------------------------------------------


def test_it_is_registered_as_a_scheduled_builder():
    assert BUILDERS["relaxed_admm"] is build_relaxed_admm_qp
    assert fom_kind("relaxed_admm") == "scheduled"
    _, qp, _ = _parametric_qp()
    assert isinstance(build_relaxed_admm_qp(qp), ScheduledFomSpec)


@pytest.mark.parametrize("bad", [{"alpha_start": 0.0}, {"alpha_final": 2.0}, {"alpha_final": -1.0}])
def test_a_relaxation_outside_the_convergent_range_is_refused(bad):
    _, qp, _ = _parametric_qp()
    with pytest.raises(ValueError, match=r"outside \(0, 2\)"):
        build_relaxed_admm_qp(qp, **bad)


def test_the_bridge_caps_max_iter_at_this_builders_table():
    """Ties the built-in to the runtime cap: past the table the accessor clamps.

    A caller raising ``cpg_set_solver_max_iter`` above the generated horizon
    would otherwise run on a frozen final relaxation rather than the method.
    """
    from fomtools.cvxpygen import interface

    _, qp, _ = _parametric_qp()
    ir = _extract(build_relaxed_admm_qp(qp), max_iters=250)

    assert interface._schedule_len(ir) == 250
    # Generated under MAX_ITERS, it exposes max_iter (capped) and no tolerance.
    assert set(interface._settings_for("scheduled", ir.termination)) == {"max_iter"}
