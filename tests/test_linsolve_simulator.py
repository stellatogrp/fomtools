"""Correctness tests for the Simulator's batched linear-system solves.

The implicit-iterate proxes/steps whose output is defined by a *pure linear
system* (``Quadratic``/``LeastSquares``, ``IndAffine``, and the equality-only
``QuadraticProgram``) are lowered to an ``Affine`` step and evaluated by the
simulator with a single batched multi-RHS solve. Each test computes a per-sample
reference in numpy, registers it as a parameter, and uses the public infinity-norm
objective as an oracle: the recorded residual per sample is ``||x_sim - x_ref||_inf``,
which must be ~0 if the batched solve is correct.

The complementarity proxes/steps (simplex, ℓ1-ball, ℓ∞-norm, total variation,
polyhedral projection, inequality QP/LP) are covered in
``tests/test_subproblem_simulator.py``.
"""

import numpy as np

from fomtools.proxable_functions import IndAffine, LeastSquares, Quadratic
from fomtools.verifier import Verifier

TOL = 1e-5


def test_quadratic_prox_simulator():
    """prox solves (I + gamma*Q) x = v - gamma*q; compare to numpy solve."""
    rng = np.random.default_rng(0)
    n, N, gamma = 4, 6, 0.7

    X = rng.normal(size=(n, n))
    Q = X.T @ X  # PSD
    q = rng.normal(size=n)
    v_samples = rng.uniform(-1.0, 1.0, size=(N, n))

    M = np.eye(n) + gamma * Q
    x_ref = np.linalg.solve(M, (v_samples - gamma * q).T).T  # (N, n)

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(Quadratic(Q=Q, q=q), v, gamma=gamma)
    xref = vp.add_param(ndim=n, samples=x_ref)
    vp.set_infinity_norm_objective(x - xref)

    residuals = vp.solve()
    assert residuals[:, -1].max() < TOL


def test_leastsquares_prox_simulator():
    """LeastSquares reuses the Quadratic lowering (Q = A^T A, q = -A^T b)."""
    rng = np.random.default_rng(1)
    n, m, N, gamma = 4, 5, 6, 1.0

    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    v_samples = rng.uniform(-1.0, 1.0, size=(N, n))

    Q = A.T @ A
    q = -A.T @ b
    M = np.eye(n) + gamma * Q
    x_ref = np.linalg.solve(M, (v_samples - gamma * q).T).T

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(LeastSquares(A=A, b=b), v, gamma=gamma)
    xref = vp.add_param(ndim=n, samples=x_ref)
    vp.set_infinity_norm_objective(x - xref)

    residuals = vp.solve()
    assert residuals[:, -1].max() < TOL


def test_indaffine_prox_simulator():
    """Projection onto {Ax = b}: x = v - A^T (A A^T)^{-1} (A v - b)."""
    rng = np.random.default_rng(2)
    n, m, N = 5, 2, 6  # m < n, A full row rank

    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    v_samples = rng.uniform(-1.0, 1.0, size=(N, n))

    AAt = A @ A.T
    nu = np.linalg.solve(AAt, (v_samples @ A.T - b).T).T  # (N, m)
    x_ref = v_samples - nu @ A  # (N, n)

    vp = Verifier(method="simulator", obbt_solver=None)
    v = vp.add_param(ndim=n, samples=v_samples)
    x = vp.prox(IndAffine(A=A, b=b), v)
    xref = vp.add_param(ndim=n, samples=x_ref)
    vp.set_infinity_norm_objective(x - xref)

    residuals = vp.solve()
    # Feasibility is the defining property: A x_ref == b for every sample.
    assert np.allclose(v_samples @ A.T - nu @ AAt, b)
    assert residuals[:, -1].max() < TOL


def test_equality_qp_simulator():
    """Equality-constrained QP via the block KKT linear solve; q is per-sample."""
    rng = np.random.default_rng(3)
    n, p, N = 4, 2, 6

    Y = rng.normal(size=(n, n))
    P = Y.T @ Y + np.eye(n)  # PD
    A = rng.normal(size=(p, n))
    b = rng.normal(size=p)
    q_samples = rng.uniform(-1.0, 1.0, size=(N, n))

    M_block = np.block([[P, A.T], [A, np.zeros((p, p))]])
    rhs = np.concatenate([-q_samples, np.tile(b, (N, 1))], axis=1)  # (N, n+p)
    x_ref = np.linalg.solve(M_block, rhs.T).T[:, :n]

    vp = Verifier(method="simulator", obbt_solver=None)
    q_param = vp.add_param(ndim=n, samples=q_samples)
    x = vp.quadratic_program(P=P, q=q_param, A=A, b=b)
    xref = vp.add_param(ndim=n, samples=x_ref)
    vp.set_infinity_norm_objective(x - xref)

    residuals = vp.solve()
    assert residuals[:, -1].max() < TOL


def test_unconstrained_qp_simulator():
    """Unconstrained QP reduces to P x = -q (per-sample q)."""
    rng = np.random.default_rng(4)
    n, N = 4, 6

    Y = rng.normal(size=(n, n))
    P = Y.T @ Y + np.eye(n)  # PD
    q_samples = rng.uniform(-1.0, 1.0, size=(N, n))
    x_ref = np.linalg.solve(P, (-q_samples).T).T

    vp = Verifier(method="simulator", obbt_solver=None)
    q_param = vp.add_param(ndim=n, samples=q_samples)
    x = vp.quadratic_program(P=P, q=q_param)
    xref = vp.add_param(ndim=n, samples=x_ref)
    vp.set_infinity_norm_objective(x - xref)

    residuals = vp.solve()
    assert residuals[:, -1].max() < TOL
