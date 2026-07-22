"""Shared problem and algorithm helpers for the optimal-control tutorials."""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa
from scipy.sparse import csr_matrix


def make_control_dynamics():
    """Return the deterministic dynamics matrices used in the tutorials."""
    A_dyn = np.array([
        [1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.],
        [0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0.],
        [0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0.],
        [0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846],
    ])
    B_dyn = np.array([
        [0., -0.0726, 0., 0.0726],
        [-0.0726, 0., 0.0726, 0.],
        [-0.0152, 0.0152, -0.0152, 0.0152],
        [-0., -0.0006, -0., 0.0006],
        [0.0006, 0., -0.0006, 0.0000],
        [0.0106, 0.0106, 0.0106, 0.0106],
        [0., -1.4512, 0., 1.4512],
        [-1.4512, 0., 1.4512, 0.],
        [-0.3049, 0.3049, -0.3049, 0.3049],
        [-0., -0.0236, 0., 0.0236],
        [0.0236, 0., -0.0236, 0.],
        [0.2107, 0.2107, 0.2107, 0.2107],
    ])
    return A_dyn, B_dyn


def build_control_qp(A_dyn, B_dyn, Q, R, horizon):
    """Build the stacked quadratic program used by the control tutorials."""
    n_state, n_input = B_dyn.shape
    n_control = (horizon - 1) * n_input
    n_total = n_state + n_control

    Sx = spa.vstack([
        spa.csr_matrix(np.linalg.matrix_power(A_dyn, i)) for i in range(horizon)
    ])
    Sz = spa.csc_matrix(((horizon - 1) * n_state, n_control))
    for i in range(horizon - 1):
        AiB = np.linalg.matrix_power(A_dyn, i) @ B_dyn
        Sz += spa.kron(spa.eye(horizon - 1, k=-i), AiB)
    Sz = spa.vstack([spa.csc_matrix((n_state, n_control)), Sz])

    Q_block = spa.kron(spa.eye(horizon), Q)
    R_block = spa.kron(spa.eye(horizon - 1), R)
    P = spa.bmat([
        [Sx.T @ Q_block @ Sx, Sx.T @ Q_block @ Sz],
        [Sz.T @ Q_block @ Sx, Sz.T @ Q_block @ Sz + R_block],
    ]).tocsr()
    M = spa.eye(n_total, format="csr")
    q = np.zeros(n_total)
    return M, P, q, n_total


def build_admm_lhs(M, P, rho, sigma):
    """Return the sparse OSQP linear-solve matrix."""
    return csr_matrix(P + sigma * spa.eye(P.shape[0]) + rho * (M.T @ M))


def admm_step(VP, M, q, rho, sigma, z_prev, v_prev, project, lhs_mat):
    """Apply one OSQP-style ADMM step for the control problem."""
    w_next = project(VP, v_prev)
    rhs = sigma * z_prev - q + rho * M.T @ (2 * w_next - v_prev)
    z_next = VP.step.implicit_linear(rhs, output_mat=lhs_mat)
    v_next = v_prev + M @ z_next - w_next
    return z_next, v_next, [z_next - z_prev, v_next - v_prev]


def solve_admm_numpy(M, q, rho, sigma, lhs_mat, lower, upper, z0, v0, max_iters):
    """Run the control ADMM recurrence with NumPy."""
    matrix = M.toarray() if hasattr(M, "toarray") else M
    lhs = lhs_mat.toarray() if hasattr(lhs_mat, "toarray") else lhs_mat
    z = np.array(z0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    for _ in range(max_iters):
        w = np.clip(v, lower, upper)
        rhs = sigma * z - q + rho * matrix.T @ (2 * w - v)
        z = np.linalg.solve(lhs, rhs)
        v = v + matrix @ z - w
    return np.concatenate([z, v])
