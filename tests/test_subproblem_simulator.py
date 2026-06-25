"""Correctness tests for the Simulator's per-sample subproblem evaluators.

The complementarity proxes/steps (simplex, ℓ1-ball, ℓ∞-norm, total variation,
polyhedral projection, and inequality QP/LP) are evaluated either by batched
solver-free projections or by per-sample cvxpy / Condat. Each test builds the
prox/step on the simulator, registers the per-sample reference (the proxable
function's own ``.prox`` / the step's ``.solve()``) as a parameter, and uses the
inf-norm objective as an oracle: the recorded residual per sample is
``||x_sim - x_ref||_inf``, which must be ~0 when the fast path is correct.
"""

import numpy as np
import pytest

from fomtools.canonicalizers.simulator_subproblems import (
    project_l1_ball_batch,
    project_simplex_batch,
)
from fomtools.proxable_functions import (
    IndBallL1,
    IndPolyhedral,
    IndSimplex,
    NormInf,
    TotalVariation1D,
)
from fomtools.steps import LinearProgram, QuadraticProgram, StandardFormLinearProgram
from fomtools.verifier import Verifier

TOL_CLOSED = 1e-5  # batched closed-form projections
TOL_SOLVER = 1e-4  # cvxpy-referenced cases


def _residual_max(vp, x, x_ref, n):
    """Wire an inf-norm oracle on (x - x_ref) and return the max final residual."""
    xref = vp.add_param(ndim=n, samples=x_ref)
    vp.set_infinity_norm_objective(x - xref)
    residuals = vp.solve()
    return residuals[:, -1].max()


# --- batched projection unit tests (independent of the verifier) ------------


def test_project_simplex_batch_matches_reference():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(30, 8))
    a = 1.7
    out = project_simplex_batch(V, a)
    assert np.allclose(out.sum(axis=1), a)
    assert (out >= -1e-12).all()
    # KKT: nonzero coords share a common threshold v - theta.
    for i in range(V.shape[0]):
        active = out[i] > 1e-9
        thr = V[i][active] - out[i][active]
        assert thr.max() - thr.min() < 1e-6


def test_project_l1_ball_batch_branches():
    rng = np.random.default_rng(1)
    inside = rng.normal(size=(5, 6)) * 0.05  # ||.||_1 < 1
    outside = rng.normal(size=(5, 6)) * 3.0  # ||.||_1 > 1
    np.testing.assert_allclose(project_l1_ball_batch(inside, 1.0), inside, atol=1e-12)
    proj = project_l1_ball_batch(outside, 1.0)
    assert np.all(np.abs(proj).sum(axis=1) <= 1.0 + 1e-6)
    assert np.allclose(project_l1_ball_batch(outside, 0.0), 0.0)


# --- prox correctness vs the cvxpy reference --------------------------------


def test_simplex_prox_simulator():
    rng = np.random.default_rng(2)
    n, N = 5, 6
    v_samples = rng.normal(size=(N, n))
    x_ref = np.stack([IndSimplex(a=1.0).prox(v) for v in v_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(IndSimplex(a=1.0), v)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_l1ball_prox_simulator():
    rng = np.random.default_rng(3)
    n = 5
    # Mix of inside/outside the ball so both branches are exercised.
    v_samples = np.vstack([rng.normal(size=(3, n)) * 0.1, rng.normal(size=(3, n)) * 2.0])
    x_ref = np.stack([IndBallL1(r=1.0).prox(v) for v in v_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(IndBallL1(r=1.0), v)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_norminf_prox_simulator():
    rng = np.random.default_rng(4)
    n, N, gamma, lambd = 5, 6, 0.8, 0.5
    v_samples = rng.normal(size=(N, n))
    x_ref = np.stack([NormInf(lambd).prox(v, gamma=gamma) for v in v_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(NormInf(lambd), v, gamma=gamma)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_tv1d_prox_simulator():
    rng = np.random.default_rng(5)
    n, N, gamma, lambd = 8, 6, 1.0, 0.4
    # Include a piecewise-constant signal (row 0) and random rows.
    v_samples = rng.normal(size=(N, n))
    v_samples[0] = np.array([0, 0, 0, 5, 5, 5, 1, 1], dtype=float)
    x_ref = np.stack([TotalVariation1D(lambd).prox(v, gamma=gamma) for v in v_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(TotalVariation1D(lambd), v, gamma=gamma)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_polyhedral_prox_simulator():
    rng = np.random.default_rng(6)
    n, N = 3, 6
    # Box [-1, 1]^n written as A u <= b.
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.ones(2 * n)
    v_samples = rng.normal(size=(N, n)) * 2.0
    x_ref = np.stack([IndPolyhedral(A, b).prox(v) for v in v_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(IndPolyhedral(A, b), v)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


# --- QP / LP steps vs the step's cvxpy .solve() -----------------------------


def test_inequality_qp_simulator():
    rng = np.random.default_rng(7)
    n, N = 3, 6
    Y = rng.normal(size=(n, n))
    P = Y.T @ Y + np.eye(n)
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = np.ones(2 * n)
    q_samples = rng.normal(size=(N, n))
    x_ref = np.stack([QuadraticProgram(P=P, q=q, C=C, d=d).solve() for q in q_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    q_param = vp.add_param(ndim=n, samples=q_samples)
    x = vp.quadratic_program(P=P, q=q_param, C=C, d=d)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_lp_simulator():
    rng = np.random.default_rng(8)
    n, N = 3, 6
    C = np.vstack([np.eye(n), -np.eye(n)])  # box [-1, 1]^n
    d = np.ones(2 * n)
    q_samples = rng.normal(size=(N, n))
    x_ref = np.stack([LinearProgram(q=q, C=C, d=d).solve() for q in q_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    q_param = vp.add_param(ndim=n, samples=q_samples)
    x = vp.linear_program(q=q_param, C=C, d=d)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_standardform_lp_simulator():
    rng = np.random.default_rng(9)
    n, N = 3, 6
    A = np.ones((1, n))
    b = np.array([1.0])  # simplex-like feasible set x >= 0, 1^T x = 1
    c_samples = rng.normal(size=(N, n))
    x_ref = np.stack([StandardFormLinearProgram(c=c, A=A, b=b).solve() for c in c_samples])

    vp = Verifier(method="simulator", obbt_solver=None)
    c_param = vp.add_param(ndim=n, samples=c_samples)
    x = vp.standard_form_linear_program(c=c_param, A=A, b=b)
    assert _residual_max(vp, x, x_ref, n) < TOL_SOLVER


def test_per_sample_scalar_param_rejected():
    """A per-sample varying simplex sum is rejected with a clear message."""
    rng = np.random.default_rng(10)
    n, N = 3, 4
    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=rng.normal(size=(N, n)))
    a_param = vp.add_param(ndim=1, samples=rng.uniform(0.5, 1.5, size=(N, 1)))
    with pytest.raises(NotImplementedError):
        vp.prox(IndSimplex(a=a_param), v)
