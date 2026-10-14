"""What the cvxpygen front end hands to code generation, and what that checks.

``generate_solver`` reaches fomtools code generation through
:meth:`FomtoolsInterface.generate_code`, which forwards the builder's
:class:`~fomtools.cvxpygen.builders.FomSpec`. These tests pin what that hand-off
carries: the recorded tape and parameter draw that turn on the stationarity
check, that recording the tape does not change the emitted solver, and that a
builder whose coefficients vary with the iteration index is now rejected instead
of being silently frozen at its first recorded value.
"""

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec  # noqa: E402
from fomtools.codegen.generate import extract_fom_ir  # noqa: E402
from fomtools.codegen.ir import NonStationaryFOMError  # noqa: E402
from fomtools.codegen.oracle import sample_param_values  # noqa: E402
from fomtools.cvxpygen import interface  # noqa: E402
from fomtools.cvxpygen.builders import (  # noqa: E402
    _GATE_ITERS,
    FomSpec,
    ScheduledFomSpec,
    build_admm_qp,
    build_pdhg_lp,
    fom_builder,
    fom_kind,
)
from fomtools.cvxpygen.qpdata import extract_qp_data  # noqa: E402
from fomtools.linearexpression import stack  # noqa: E402
from fomtools.verifier import Verifier  # noqa: E402

from .conftest import cvxpygen_canonicalize  # noqa: E402

#: The warning generation raises when nothing checked that the method is
#: stationary. Its absence is the whole point of the tape the builders record.
STATIONARITY_WARNING = "did not check that the recorded"


def _admm_qp(seed=0, n=4, m_eq=2):
    """A parametric QP with equality rows, inequality rows and a parametric q."""
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
    return extract_qp_data(canon, param_bounds={q: (-1.0, 1.0), beq: (-2.0, 2.0)})


def _pdhg_lp(seed=3, n=4, m_eq=2):
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
    return extract_qp_data(canon, param_bounds={c: (-2.0, 2.0)})


def _extract(spec, max_iters=50):
    """Extract exactly as ``FomtoolsInterface.generate_code`` does."""
    return extract_fom_ir(
        spec.verifier,
        spec.next_state,
        config=CodegenConfig(termination=TerminationSpec(max_iters=max_iters)),
        state=list(spec.state),
        params=list(spec.params),
        name=spec.name,
        iterates=list(spec.iterates) or None,
        param_values=spec.param_values,
    )


BUILDERS = {"admm": (build_admm_qp, _admm_qp), "pdhg": (build_pdhg_lp, _pdhg_lp)}


# --- the tape the built-in builders record ----------------------------------


@pytest.mark.parametrize("fom", sorted(BUILDERS))
def test_builtin_builders_record_a_tape_to_check(fom):
    build, make_qp = BUILDERS[fom]
    spec = build(make_qp())

    # The recorded iterations only: the rolled check reads every entry as an
    # iteration, so a leading initial iterate would shift the comparison by one.
    assert len(spec.iterates) == _GATE_ITERS
    assert set(spec.param_values) == set(spec.params)
    for value in spec.param_values.values():
        assert np.all(np.isfinite(value))


@pytest.mark.parametrize("fom", sorted(BUILDERS))
def test_extraction_does_not_warn_about_stationarity(fom):
    """The gate is on: generation checks rather than assumes."""
    build, make_qp = BUILDERS[fom]
    spec = build(make_qp())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ir = _extract(spec)

    assert not [w for w in caught if STATIONARITY_WARNING in str(w.message)]
    assert len(ir.body) >= 1


@pytest.mark.parametrize("fom", sorted(BUILDERS))
def test_recording_the_tape_does_not_change_the_emitted_ir(fom):
    """Extraction takes the dependency cone of one iteration, not the whole tape."""
    build, make_qp = BUILDERS[fom]
    spec = build(make_qp())
    ir = _extract(spec)

    # The verifier holds every recorded iteration; the IR holds exactly one.
    assert len(spec.verifier.steps) == _GATE_ITERS * len(ir.body)
    assert [dim for _, dim in ir.state.slots] == [expr.ndim for expr in spec.state]


def test_admm_ir_is_one_solve_plus_one_projection():
    """The emitted body is unchanged by the gate: KKT solve + inequality relu."""
    ir = _extract(build_admm_qp(_admm_qp()))
    assert len(ir.body) == 2
    assert [step.solve is not None for step in ir.body] == [True, False]


# --- a builder that is not stationary is now caught --------------------------


def _decaying_step_spec(qp, decay: bool, with_next_state: bool = True):
    """A custom builder: projected gradient whose step size decays with ``k``.

    With ``decay=False`` the identical code records a stationary method, so the
    pair pins that the check fires on the varying coefficient and nothing else.
    With ``with_next_state=False`` the same tape selects the unrolled path, where
    a varying coefficient is generated as recorded rather than rejected.
    """
    n = qp.n
    t = 0.5
    W = np.eye(n) - t * qp.P.toarray()
    lb, ub = qp.bounds["q"]

    v = Verifier(method="simulator", obbt_solver=None)
    q_e = v.add_param(n, lb=lb, ub=ub)
    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))

    groups, xk = [], x
    for k in range(_GATE_ITERS):
        step = t / (k + 1.0) if decay else t
        xk = v.step.relu(W @ xk - step * q_e)
        groups.append([xk])

    iterates = tuple(groups)
    return FomSpec(
        verifier=v,
        next_state=stack(groups[0]) if with_next_state else None,
        name="decaying",
        primal_offset=0,
        primal_dim=n,
        dual_offset=0,
        dual_dim=0,
        param_copy=("q",),
        state=(x,),
        params=(q_e,),
        iterates=iterates,
        param_values=sample_param_values(v, list(iterates), state=[x], params=[q_e], seed=0),
    )


def test_gate_catches_a_non_stationary_custom_builder():
    with pytest.raises(NonStationaryFOMError):
        _extract(_decaying_step_spec(_admm_qp(), decay=True))


def test_gate_passes_the_same_builder_held_stationary():
    ir = _extract(_decaying_step_spec(_admm_qp(), decay=False))
    assert len(ir.body) == 1


# --- the three generation paths ---------------------------------------------


@fom_builder("rolled")
def _rolled_builder(qp, **_):
    return _decaying_step_spec(qp, decay=False)


@fom_builder("unrolled")
def _unrolled_builder(qp, decay=True, **_):
    return _decaying_step_spec(qp, decay=decay, with_next_state=False)


@fom_builder("scheduled")
def _scheduled_builder(qp, **_):
    """Projected gradient whose relaxation is declared as a table, not recorded.

    ``theta`` scales the whole linear operator, so every coefficient block is
    affine in it — what :mod:`fomtools.codegen.extract_scheduled` recovers by
    probing the build function at several values.
    """
    n = qp.n
    t = 0.5
    W = np.eye(n) - t * qp.P.toarray()
    lb, ub = qp.bounds["q"]

    def build(theta):
        from fomtools.codegen.extract_scheduled import Recording

        v = Verifier(method="simulator", obbt_solver=None)
        q_e = v.add_param(n, lb=lb, ub=ub)
        x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
        x_next = v.step.relu(theta * (W @ x - t * q_e))
        return Recording(v, next_state=x_next, state=[x], params=[q_e])

    return ScheduledFomSpec(
        build=build,
        schedules={"theta": lambda k: 1.0 - 0.5 / (k + 2.0)},
        name="scheduled_pg",
        primal_offset=0,
        primal_dim=n,
        dual_offset=0,
        dual_dim=0,
        param_copy=("q",),
    )


@pytest.fixture
def captured_codegen(monkeypatch):
    """Replace the code generator with real *extraction* plus a call recorder.

    Runs the same three-way dispatch ``generate_code`` would (``extract_fom_ir``
    is what it calls) without scaffolding a directory or invoking an agent, so a
    test sees both which arguments the interface sent and what IR came back.
    """
    calls = []

    def fake_generate_code(verifier, next_state=None, out_dir=None, **kw):
        calls.append({"verifier": verifier, "next_state": next_state, **kw})
        ir = extract_fom_ir(
            verifier,
            next_state,
            config=kw.get("config"),
            state=kw.get("state"),
            params=kw.get("params"),
            name=kw.get("name", "fom"),
            schedules=kw.get("schedules"),
            iterates=kw.get("iterates"),
            param_values=kw.get("param_values"),
        )
        return SimpleNamespace(ir=ir, success=True, translate=SimpleNamespace(log=""))

    monkeypatch.setattr(interface, "fomtools_generate_code", fake_generate_code)
    return calls


def _dispatch(spec, kind, tmp_path, max_iters=20):
    return interface._run_codegen(
        spec,
        kind,
        tmp_path,
        CodegenConfig(termination=TerminationSpec(max_iters=max_iters)),
        driver=None,
    )


def test_rolled_builder_generates_one_repeated_iteration(captured_codegen, tmp_path):
    result = _dispatch(_rolled_builder(_admm_qp()), "rolled", tmp_path)

    call = captured_codegen[0]
    assert call["next_state"] is not None
    assert call["iterates"] is not None, "the tape rides along as the stationarity check"
    assert result.ir.unrolled is False
    assert len(result.ir.body) == 1


def test_unrolled_builder_generates_the_recorded_run(captured_codegen, tmp_path):
    result = _dispatch(_unrolled_builder(_admm_qp()), "unrolled", tmp_path)

    call = captured_codegen[0]
    assert call["next_state"] is None, "no next_state is what selects the unrolled path"
    assert len(call["iterates"]) == _GATE_ITERS
    assert result.ir.unrolled is True
    assert len(result.ir.body) == _GATE_ITERS


def test_a_stationary_unrolled_recording_rolls_back_up(captured_codegen, tmp_path):
    """Rolling is a size optimization generation verifies before applying."""
    result = _dispatch(_unrolled_builder(_admm_qp(), decay=False), "unrolled", tmp_path)

    assert result.ir.unrolled is False
    assert len(result.ir.body) == 1
    assert result.ir.termination.max_iters == _GATE_ITERS, "the recorded horizon is kept"


def test_scheduled_builder_generates_one_block_plus_a_table(captured_codegen, tmp_path):
    spec = _scheduled_builder(_admm_qp())
    result = _dispatch(spec, "scheduled", tmp_path, max_iters=20)

    call = captured_codegen[0]
    assert call["verifier"] is spec.build, "the build function goes in, not a verifier"
    assert call["schedules"] == spec.schedules
    assert set(result.ir.schedules) == {"theta"}
    assert len(result.ir.schedules["theta"]) == 20
    assert result.ir.unrolled is False
    assert len(result.ir.body) == 1, "one block whatever the horizon"


# --- what the declared kind must agree with ---------------------------------


def test_undecorated_builders_are_rolled():
    assert fom_kind(lambda qp: None) == "rolled"
    assert fom_kind("admm") == "rolled"
    assert fom_kind(_scheduled_builder) == "scheduled"
    assert fom_kind(_unrolled_builder) == "unrolled"


def test_fom_builder_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown FOM kind"):
        fom_builder("halfway")


def test_scheduled_kind_requires_a_scheduled_spec():
    with pytest.raises(TypeError, match="needs a ScheduledFomSpec"):
        interface._check_spec_matches_kind(_rolled_builder(_admm_qp()), "scheduled")


def test_a_scheduled_spec_must_declare_itself():
    with pytest.raises(TypeError, match="@fom_builder"):
        interface._check_spec_matches_kind(_scheduled_builder(_admm_qp()), "rolled")


def test_rolled_kind_requires_a_next_state():
    spec = _decaying_step_spec(_admm_qp(), decay=False, with_next_state=False)
    with pytest.raises(ValueError, match="no next_state"):
        interface._check_spec_matches_kind(spec, "rolled")


def test_unrolled_kind_rejects_a_next_state():
    """The two select different paths, so carrying both is ambiguous, not extra."""
    spec = _decaying_step_spec(_admm_qp(), decay=True, with_next_state=True)
    with pytest.raises(ValueError, match="also set next_state"):
        interface._check_spec_matches_kind(spec, "unrolled")


def test_unrolled_is_opt_in():
    opts = interface.FomtoolsOptions(fom=_unrolled_builder)
    with pytest.raises(ValueError, match="allow_unrolled=True"):
        interface._check_unrolled_allowed(opts, CodegenConfig())

    allowed = interface.FomtoolsOptions(fom=_unrolled_builder, allow_unrolled=True)
    interface._check_unrolled_allowed(allowed, CodegenConfig())  # no raise


def test_unrolled_rejects_tolerance_termination():
    """A single pass has no per-iteration residual for cpg_set_solver_tol to act on."""
    opts = interface.FomtoolsOptions(fom=_unrolled_builder, allow_unrolled=True)
    config = CodegenConfig(
        termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=50, tol=1e-9)
    )
    with pytest.raises(ValueError, match="cannot be combined with an unrolled FOM"):
        interface._check_unrolled_allowed(opts, config)


def test_rolled_path_still_accepts_tolerance_termination(captured_codegen, tmp_path):
    """The guard is specific to the unrolled path; a rolled loop can still early-exit."""
    config = CodegenConfig(
        termination=TerminationSpec(kind=TerminationKind.INF_NORM_CHANGE, max_iters=50, tol=1e-9)
    )
    result = interface._run_codegen(
        _rolled_builder(_admm_qp()), "rolled", tmp_path, config, driver=None
    )
    assert result.ir.termination.kind == TerminationKind.INF_NORM_CHANGE
    assert result.ir.termination.tol == 1e-9


@pytest.mark.parametrize(
    "kind,schedules,unrolled",
    [("rolled", {}, True), ("scheduled", {}, False), ("rolled", {"a": [1.0]}, False)],
)
def test_a_solver_of_the_wrong_shape_is_refused(kind, schedules, unrolled):
    """Defensive: the settings and bridge are chosen from the declared kind."""
    ir = SimpleNamespace(schedules=schedules, unrolled=unrolled)
    with pytest.raises(RuntimeError, match="generation produced a"):
        interface._check_ir_matches_kind(ir, kind)


def test_an_unrolled_declaration_may_roll_up():
    """The one mismatch that is expected, so it must not raise."""
    interface._check_ir_matches_kind(SimpleNamespace(schedules={}, unrolled=False), "unrolled")


# --- the runtime settings each kind can honestly expose ----------------------


def _canonicalize_with(fom, problem, **option_kwargs):
    """Canonicalize through the real FOMTOOLS interface, returning the instance.

    Stops short of code generation: this is what cvxpygen does *before* it ever
    reaches ``generate_code``, which is exactly when the settings struct is
    fixed — and therefore why the settings surface cannot be inferred from the
    IR that generation later produces.
    """
    from cvxpygen.canonicalizer import Canonicalizer

    interface.register()
    options = interface.FomtoolsOptions(fom=fom, **option_kwargs)
    with interface.pending_options(options):
        _, iface = Canonicalizer(solver="FOMTOOLS").canonicalize(problem)
    return iface


def _qp_problem():
    rng = np.random.default_rng(0)
    n, m_eq = 4, 2
    A = rng.standard_normal((m_eq, n))
    q = cp.Parameter(n, name="q")
    q.value = 0.1 * rng.standard_normal(n)
    x = cp.Variable(n, name="x")
    return cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
        [A @ x == A @ rng.uniform(-0.3, 0.3, size=n), x <= 1.0, -1.0 <= x],
    )


@pytest.mark.parametrize("kind", ["rolled", "scheduled"])
def test_looping_solvers_with_a_tolerance_expose_all_three_settings(kind):
    term = CodegenConfig(
        termination=TerminationSpec(
            kind=TerminationKind.INF_NORM_CHANGE, max_iters=250, tol=1e-5, check_termination=5
        )
    ).termination
    stgs = interface._settings_for(kind, term)

    assert set(stgs) == {"max_iter", "tol", "check_termination"}
    assert stgs["max_iter"].default == "250"
    assert float(stgs["tol"].default) == 1e-5
    assert stgs["check_termination"].default == "5"


@pytest.mark.parametrize("kind", ["rolled", "scheduled"])
def test_max_iters_solvers_expose_only_max_iter(kind):
    """Regression: under MAX_ITERS the tolerance check is compiled out
    (FOM_USE_TOL 0), so a runtime ``tol`` was accepted and silently ignored."""
    term = CodegenConfig(termination=TerminationSpec(max_iters=250)).termination
    stgs = interface._settings_for(kind, term)

    assert set(stgs) == {"max_iter"}
    assert stgs["max_iter"].default == "250"


def test_an_unrolled_solver_exposes_no_settings():
    """None of the three mean anything for a body that is the whole run."""
    assert interface._settings_for("unrolled", CodegenConfig().termination) == {}


def test_the_interface_drops_the_settings_struct_when_there_are_none():
    """An empty Canon_Settings_t would be an empty struct — not ISO C.

    cvxpygen guards the struct's typedef, instance and prototype behind
    ``stgs_requires_extra_struct_type`` and derives every ``cpg_set_solver_*``
    from the settings themselves, so turning the flag off removes all of them.
    """
    iface = _canonicalize_with(_unrolled_builder, _qp_problem(), allow_unrolled=True)

    assert iface.stgs == {}
    assert iface.stgs_requires_extra_struct_type is False


def test_the_interface_keeps_the_settings_struct_for_a_looping_solver():
    iface = _canonicalize_with("admm", _qp_problem())

    assert set(iface.stgs) == {"max_iter"}
    assert iface.stgs_requires_extra_struct_type is True


def test_the_settings_flag_does_not_leak_between_runs():
    """It is set per instance, so an unrolled run must not disarm the next one."""
    _canonicalize_with(_unrolled_builder, _qp_problem(), allow_unrolled=True)
    iface = _canonicalize_with("admm", _qp_problem())

    assert iface.stgs_requires_extra_struct_type is True


def test_schedule_length_is_the_shortest_table():
    """The binding table is the first one whose accessor starts clamping."""
    ir = SimpleNamespace(schedules={"a": np.zeros(64), "b": np.zeros(40)})
    assert interface._schedule_len(ir) == 40
    assert interface._schedule_len(SimpleNamespace(schedules={})) is None


def test_unrolled_iteration_count_is_the_recorded_iterations():
    """The bridge reports what the emitted run performs, not the tape length."""
    spec = _decaying_step_spec(_admm_qp(), decay=True, with_next_state=False)
    assert interface._unrolled_iter_count(spec) == _GATE_ITERS


def test_a_leading_initial_iterate_is_not_counted_as_an_iteration():
    """An IterateTrajectory keeps the starting point in slot 0; it ran nothing.

    Extraction drops that entry before generating, so counting the tape verbatim
    would bake ``_GATE_ITERS + 1`` into the emitted ``fom_iter`` for a builder
    that recorded where it started.
    """
    spec = _decaying_step_spec(_admm_qp(), decay=True, with_next_state=False)
    with_initial = replace(spec, iterates=((spec.state[0],), *spec.iterates))

    assert len(with_initial.iterates) == _GATE_ITERS + 1
    assert interface._unrolled_iter_count(with_initial) == _GATE_ITERS


def test_a_table_shorter_than_the_horizon_is_refused(captured_codegen, tmp_path):
    """Generating past the tables would freeze the coefficients mid-schedule."""
    spec = _scheduled_builder(_admm_qp())
    short = ScheduledFomSpec(
        build=spec.build,
        schedules={"theta": [0.9] * 5},
        name=spec.name,
        primal_offset=0,
        primal_dim=spec.primal_dim,
        dual_offset=0,
        dual_dim=0,
        param_copy=spec.param_copy,
    )
    with pytest.raises(ValueError, match="entries but the generated loop"):
        _dispatch(short, "scheduled", tmp_path, max_iters=20)


# --- through the real front end ---------------------------------------------


@pytest.mark.requires_compiler
def test_generate_solver_does_not_warn_about_stationarity(requires_compiler, tmp_path, monkeypatch):
    """End to end: the warning is gone and the checked-in golden still verifies.

    Reuses the e2e problem and golden solver body, so this also pins that the
    extra recorded iterations left the emitted solver byte-identical — a changed
    IR would fail ``make verify`` against that golden.
    """
    from fomtools.codegen.agent.stub import StubAgentDriver
    from fomtools.cvxpygen import generate_solver
    from tests.codegen.test_cvxpygen_e2e import GOLDEN, _build_problem

    monkeypatch.chdir(tmp_path)
    problem, q, _ = _build_problem()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = generate_solver(
            problem,
            "cpg_gate",
            fom="admm",
            param_bounds={q: (-1.0, 1.0)},
            config=CodegenConfig(termination=TerminationSpec(max_iters=2000)),
            driver=StubAgentDriver(GOLDEN),
            wrapper=False,
        )

    assert result.success, result.translate.log
    assert not [w for w in caught if STATIONARITY_WARNING in str(w.message)]


@pytest.mark.parametrize("code_optimization", [False, True])
def test_code_optimization_reaches_the_generator(captured_codegen, tmp_path, code_optimization):
    interface._run_codegen(
        _rolled_builder(_admm_qp()),
        "rolled",
        tmp_path,
        CodegenConfig(termination=TerminationSpec(max_iters=20)),
        driver=None,
        code_optimization=code_optimization,
    )
    call = captured_codegen[0]
    assert call["code_optimization"] is code_optimization
    # Profiling turns on the head-to-head benchmark that decides whether the
    # optimized solver is kept.
    assert call["profile"] is code_optimization


@pytest.mark.parametrize("code_optimization", [False, True])
def test_scheduled_path_forwards_code_optimization(captured_codegen, tmp_path, code_optimization):
    interface._run_codegen(
        _scheduled_builder(_admm_qp()),
        "scheduled",
        tmp_path,
        CodegenConfig(termination=TerminationSpec(max_iters=20)),
        driver=None,
        code_optimization=code_optimization,
    )
    assert captured_codegen[0]["code_optimization"] is code_optimization
