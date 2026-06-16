"""Math gate for the cvxpygen FOM builders.

Extract the IR from each builder and run the pure-NumPy reference oracle for
many iterations; the fixed point must match the QP/LP solution that CVXPY
computes for the same problem. No C compiler or code generation involved.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.codegen.config import CodegenConfig, TerminationSpec  # noqa: E402
from fomtools.codegen.extract import extract_ir  # noqa: E402
from fomtools.codegen.oracle import ReferenceOracle  # noqa: E402
from fomtools.cvxpygen.builders import build_admm_qp, build_pdhg_lp, resolve_builder  # noqa: E402
from fomtools.cvxpygen.qpdata import extract_qp_data  # noqa: E402


def _canonicalize(problem):
    from cvxpygen.canonicalizer import Canonicalizer

    return Canonicalizer(solver="OSQP").canonicalize(problem)


def _run_fom(spec, qp, max_iters):
    """Run the extracted FOM ``max_iters`` iterations from the canonical data."""
    config = CodegenConfig(termination=TerminationSpec(max_iters=max_iters))
    ir = extract_ir(
        spec.verifier,
        spec.next_state,
        state=list(spec.state),
        params=list(spec.params),
        config=config,
        name=spec.name,
    )
    values = {"q": qp.q, "b": qp.b, "u_ineq": qp.u_ineq}
    param_values = {f"p{i}": values[piece] for i, piece in enumerate(spec.param_copy)}
    res = ReferenceOracle(ir).solve(param_values=param_values)
    x = res.solution[spec.primal_offset : spec.primal_offset + spec.primal_dim]
    y = res.solution[spec.dual_offset : spec.dual_offset + spec.dual_dim]
    return x, y


def _parametric_qp(seed=0, n=4, m_eq=2):
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
    return problem, q, beq, x


def test_admm_matches_cvxpy_qp():
    problem, q, beq, x = _parametric_qp()
    canon, _ = _canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={q: (-1.0, 1.0), beq: (-2.0, 2.0)})
    spec = build_admm_qp(qp)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, y_fom = _run_fom(spec, qp, max_iters=4000)

    np.testing.assert_allclose(x_fom, x.value, atol=1e-5)
    # The FOM dual y is the canonical dual; check stationarity instead of
    # matching CVXPY's per-constraint duals directly.
    res = qp.P @ x_fom + qp.q + qp.A.T @ y_fom
    np.testing.assert_allclose(res, 0.0, atol=1e-5)


def test_admm_constant_data_qp():
    # No parametric q: only the equality rhs is a parameter.
    rng = np.random.default_rng(1)
    n, m_eq = 3, 2
    A = rng.standard_normal((m_eq, n))
    beq = cp.Parameter(m_eq, name="beq")
    beq.value = 0.2 * np.ones(m_eq)
    x = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + np.ones(n) @ x),
        [A @ x == beq, x <= 2.0, -2.0 <= x],
    )
    canon, _ = _canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={beq: (-1.0, 1.0)})
    assert qp.param_dependent == {"b"}
    spec = build_admm_qp(qp)
    assert spec.param_copy == ("b",)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, _ = _run_fom(spec, qp, max_iters=4000)
    np.testing.assert_allclose(x_fom, x.value, atol=1e-5)


def test_admm_parametric_inequality_rhs():
    rng = np.random.default_rng(2)
    n = 3
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = cp.Parameter(2 * n, name="h")
    h.value = np.full(2 * n, 0.8)
    q_const = rng.standard_normal(n)
    x = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + q_const @ x), [G @ x <= h])
    canon, _ = _canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={h: (0.5, 1.0)})
    assert "u_ineq" in qp.param_dependent
    spec = build_admm_qp(qp)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, _ = _run_fom(spec, qp, max_iters=4000)
    np.testing.assert_allclose(x_fom, x.value, atol=1e-5)


def test_pdhg_matches_cvxpy_lp():
    rng = np.random.default_rng(3)
    n, m_eq = 4, 2
    A = rng.standard_normal((m_eq, n))
    c = cp.Parameter(n, name="c")
    c.value = rng.standard_normal(n)
    x = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(c @ x),
        [A @ x == A @ rng.uniform(-0.2, 0.2, size=n), x <= 1.0, -1.0 <= x],
    )
    canon, _ = _canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={c: (-2.0, 2.0)})
    spec = build_pdhg_lp(qp)

    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    x_fom, _ = _run_fom(spec, qp, max_iters=20000)
    np.testing.assert_allclose(x_fom, x.value, atol=1e-4)


def test_pdhg_rejects_qp():
    problem, q, beq, _ = _parametric_qp()
    canon, _ = _canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={q: (-1, 1), beq: (-1, 1)})
    with pytest.raises(ValueError, match="zero quadratic term"):
        build_pdhg_lp(qp)


def test_resolve_builder():
    assert resolve_builder("admm") is build_admm_qp
    assert resolve_builder("pdhg") is build_pdhg_lp
    assert resolve_builder(build_admm_qp) is build_admm_qp
    with pytest.raises(ValueError, match="unknown FOM"):
        resolve_builder("simplex")
