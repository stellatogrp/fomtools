"""Ruiz equilibration of the canonical QP (``fomtools.cvxpygen.scaling``).

The scaling itself, the exactness of the scaled parameter bounds, and the math
gate: a builder run on the equilibrated QP, unscaled as the bridge unscales it,
must reach the *original* problem's solution, and converge faster on a badly
scaled problem. No C compiler or code generation involved.
"""

import numpy as np
import pytest
import scipy.sparse as spa

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.cvxpygen.builders import build_admm_qp, build_pdhg_lp  # noqa: E402
from fomtools.cvxpygen.qpdata import CVXPYGEN_INF, QPData, extract_qp_data  # noqa: E402
from fomtools.cvxpygen.scaling import ruiz_for, ruiz_kkt, scale_qp  # noqa: E402

from .conftest import cvxpygen_canonicalize  # noqa: E402
from .test_cvxpygen_builders import _run_fom  # noqa: E402


def _badly_scaled_qp(seed=0, n=5, m_eq=2):
    """A parametric QP whose variables and rows span several orders of magnitude."""
    rng = np.random.default_rng(seed)
    col = np.logspace(-2, 2, n)
    A = rng.standard_normal((m_eq, n)) * col
    q = cp.Parameter(n, name="q")
    x = cp.Variable(n, name="x")
    P = np.diag(np.logspace(-1, 3, n))
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
        [A @ x == A @ (0.1 / col), 100.0 * x <= 1.0, -1.0 <= x],
    )
    q.value = rng.standard_normal(n)
    return problem, q, x


def _qp_data(problem, q):
    canon, _ = cvxpygen_canonicalize(problem)
    return extract_qp_data(canon, param_bounds={q: (q.value - 1.0, q.value + 1.0)})


def _kkt(qp: QPData) -> np.ndarray:
    P, A = qp.P.toarray(), qp.A.toarray()
    return np.block([[P, A.T], [A, np.zeros((qp.m, qp.m))]])


def test_ruiz_equilibrates_the_kkt_matrix():
    problem, q, _ = _badly_scaled_qp()
    qp = _qp_data(problem, q)
    scaled = scale_qp(qp, ruiz_for(qp))

    def spread(K):
        norms = np.abs(K).max(axis=0)
        norms = norms[norms > 0]
        return norms.max() / norms.min()

    # The cost scaling c multiplies the P block only, so compare D and E's work
    # on the KKT matrix without it.
    s = ruiz_for(qp)
    K = np.diag(np.r_[s.D, s.E]) @ _kkt(qp) @ np.diag(np.r_[s.D, s.E])
    assert spread(K) < 2.0
    assert spread(_kkt(qp)) >= 1e3
    assert np.all(np.isfinite(scaled.A.data))


def test_lp_scaling_ignores_the_empty_quadratic_term():
    A = spa.csc_matrix(np.array([[1e3, 1.0], [0.0, 1e-2]]))
    s = ruiz_kkt(spa.csc_matrix((2, 2)), A, q_abs=np.array([1.0, 1.0]))
    scaled = np.abs(np.diag(s.E) @ A.toarray() @ np.diag(s.D))
    assert np.allclose(scaled.max(axis=0), 1.0, atol=0.1)
    assert np.allclose(scaled.max(axis=1), 1.0, atol=0.1)
    assert s.c > 0


def test_scaled_bounds_are_exact():
    problem, q, _ = _badly_scaled_qp()
    qp = _qp_data(problem, q)
    s = ruiz_for(qp)
    scaled = scale_qp(qp, s)
    lb, ub = qp.bounds["q"]
    slb, sub = scaled.bounds["q"]
    np.testing.assert_allclose(slb, s.c * s.D * lb)
    np.testing.assert_allclose(sub, s.c * s.D * ub)
    assert np.all(slb <= sub)
    # The data at the current value lies inside the scaled box.
    assert np.all(slb <= scaled.q + 1e-12) and np.all(scaled.q <= sub + 1e-12)


def test_infinite_bounds_stay_infinite():
    problem, q, _ = _badly_scaled_qp()
    qp = _qp_data(problem, q)
    u = qp.u_ineq.copy()
    u[0] = CVXPYGEN_INF
    qp_inf = QPData(**{**qp.__dict__, "u_ineq": u})
    scaled = scale_qp(qp_inf, ruiz_for(qp_inf))
    assert scaled.u_ineq[0] == CVXPYGEN_INF


@pytest.mark.parametrize("seed", [0, 1])
def test_admm_on_the_equilibrated_qp_solves_the_original(seed):
    problem, q, x = _badly_scaled_qp(seed)
    qp = _qp_data(problem, q)
    s = ruiz_for(qp)
    scaled = scale_qp(qp, s)

    xs, ys = _run_fom(build_admm_qp(scaled), scaled, max_iters=4000)
    x_fom, y_fom = s.D * xs, s.E * ys / s.c  # what the bridge publishes

    problem.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(x_fom, x.value, atol=1e-5)
    # The original problem's KKT conditions, in the original (unscaled) space.
    stationarity = qp.P @ x_fom + qp.q + qp.A.T @ y_fom
    assert np.max(np.abs(stationarity)) < 1e-5


def test_equilibration_speeds_up_admm_on_a_badly_scaled_qp():
    problem, q, x = _badly_scaled_qp()
    qp = _qp_data(problem, q)
    s = ruiz_for(qp)
    scaled = scale_qp(qp, s)
    problem.solve(solver=cp.CLARABEL)

    iters = 200
    x_plain, _ = _run_fom(build_admm_qp(qp), qp, max_iters=iters)
    xs, _ = _run_fom(build_admm_qp(scaled), scaled, max_iters=iters)
    err_plain = np.max(np.abs(x_plain - x.value))
    err_scaled = np.max(np.abs(s.D * xs - x.value))
    assert err_scaled < 0.1 * err_plain


def test_pdhg_on_the_equilibrated_lp_solves_the_original():
    rng = np.random.default_rng(3)
    n, m = 4, 2
    A = rng.standard_normal((m, n)) * np.logspace(-1, 1, n)
    c = cp.Parameter(n, name="c")
    x = cp.Variable(n, name="x")
    problem = cp.Problem(cp.Minimize(c @ x), [A @ x == A @ np.full(n, 0.5), x >= 0, x <= 1])
    c.value = rng.uniform(0.5, 1.5, n)
    canon, _ = cvxpygen_canonicalize(problem)
    qp = extract_qp_data(canon, param_bounds={c: (0.4, 1.6)})
    s = ruiz_for(qp)
    scaled = scale_qp(qp, s)

    xs, _ = _run_fom(build_pdhg_lp(scaled), scaled, max_iters=20000)
    problem.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(s.D * xs, x.value, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1])
def test_ruiz_matches_osqps_own_scaling(seed):
    """ADMM on our equilibrated QP tracks OSQP's (scaling=10) iterates, unscaled.

    OSQP scales the cost by the current ``q``; ``ruiz_for`` bounds ``|q|`` over its
    parameter box instead, so the comparison passes the current ``|q|``.
    """
    osqp = pytest.importorskip("osqp")
    problem, q, _ = _badly_scaled_qp(seed)
    qp = _qp_data(problem, q)
    s = ruiz_kkt(qp.P, qp.A, q_abs=np.abs(qp.q))
    scaled = scale_qp(qp, s)
    l = np.r_[qp.b, -np.inf * np.ones(qp.n_ineq)]
    u = np.r_[qp.b, qp.u_ineq]
    for iters in (25, 100):
        solver = osqp.OSQP()
        solver.setup(
            P=spa.triu(qp.P, format="csc"), q=qp.q, A=qp.A, l=l, u=u, verbose=False,
            scaling=10, adaptive_rho=False, max_iter=iters, check_termination=0,
            warm_starting=False, polishing=False,
        )  # fmt: skip
        x_osqp = solver.solve().x
        xs, _ = _run_fom(build_admm_qp(scaled), scaled, max_iters=iters)
        np.testing.assert_allclose(s.D * xs, x_osqp, rtol=1e-6, atol=1e-8)
