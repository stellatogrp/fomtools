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

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.codegen.config import CodegenConfig, TerminationSpec  # noqa: E402
from fomtools.codegen.generate import extract_fom_ir  # noqa: E402
from fomtools.codegen.ir import NonStationaryFOMError  # noqa: E402
from fomtools.codegen.oracle import sample_param_values  # noqa: E402
from fomtools.cvxpygen.builders import (  # noqa: E402
    _GATE_ITERS,
    FomSpec,
    build_admm_qp,
    build_pdhg_lp,
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


def _decaying_step_spec(qp, decay: bool):
    """A custom builder: projected gradient whose step size decays with ``k``.

    With ``decay=False`` the identical code records a stationary method, so the
    pair pins that the check fires on the varying coefficient and nothing else.
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
        next_state=stack(groups[0]),
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
