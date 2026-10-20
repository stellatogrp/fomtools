"""Shared problem and algorithm helpers for the optimal-control tutorials."""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa
from scipy.sparse import csr_matrix

from fomtools import IterateTrajectory


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


def admm_step(VP, M, q, rho, sigma, z_prev, v_prev, project, lhs_mat, k=None):
    """Apply one OSQP-style ADMM step for the control problem.

    ``z_prev`` and ``v_prev`` may be current expressions or backend-aware
    trajectories. Trajectories read slot ``k - 1`` and store the next state in
    slot ``k``.
    """
    z_trajectory = z_prev if isinstance(z_prev, IterateTrajectory) else None
    v_trajectory = v_prev if isinstance(v_prev, IterateTrajectory) else None
    if (z_trajectory is not None or v_trajectory is not None) and k is None:
        raise ValueError("k is required when admm_step receives a trajectory.")

    z_current = z_trajectory[k - 1] if z_trajectory is not None else z_prev
    v_current = v_trajectory[k - 1] if v_trajectory is not None else v_prev

    w_next = project(VP, v_current)
    rhs = sigma * z_current - q + rho * M.T @ (2 * w_next - v_current)
    z_next = VP.step.implicit_linear(rhs, output_mat=lhs_mat)
    if z_trajectory is not None:
        z_trajectory[k] = z_next
        z_next = z_trajectory[k]

    v_next = v_current + M @ z_next - w_next
    if v_trajectory is not None:
        v_trajectory[k] = v_next
        v_next = v_trajectory[k]

    return (
        z_trajectory if z_trajectory is not None else z_next,
        v_trajectory if v_trajectory is not None else v_next,
        [z_next - z_current, v_next - v_current],
    )


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


def estimate_control_pepit_radius(
    M,
    q,
    rho,
    sigma,
    lhs_mat,
    input_lb,
    input_ub,
    z0_samples,
    v0_samples,
    x_samples,
    max_iters=2000,
    tol=1e-10,
):
    """Estimate the sampled OSQP-state radius used by the control PEP.

    Runs the vectorized ADMM recurrence to a fixed point for every sampled
    initial state and parameter, then returns the largest distance between
    ``(z0, v0)`` and the corresponding fixed point. The result is empirical:
    it covers the supplied initial samples, not their entire boxes.
    """
    lhs_mat = lhs_mat if spa.issparse(lhs_mat) else spa.csc_matrix(lhs_mat)
    lhs_lu = spa.linalg.splu(lhs_mat.tocsc())
    z_star_samples = np.array(z0_samples, dtype=float, copy=True)
    v_star_samples = np.array(v0_samples, dtype=float, copy=True)
    n_samples = x_samples.shape[0]
    lower_samples = np.hstack([
        x_samples,
        np.broadcast_to(input_lb, (n_samples, np.size(input_lb))),
    ])
    upper_samples = np.hstack([
        x_samples,
        np.broadcast_to(input_ub, (n_samples, np.size(input_ub))),
    ])

    for _ in range(max_iters):
        w_star_samples = np.clip(
            v_star_samples,
            lower_samples,
            upper_samples,
        )
        rhs_samples = sigma * z_star_samples - q[np.newaxis, :] + rho * (
            2 * w_star_samples - v_star_samples
        ) @ M
        z_next_samples = lhs_lu.solve(rhs_samples.T).T
        v_next_samples = (
            v_star_samples + z_next_samples @ M - w_star_samples
        )
        step_norm = np.max(np.linalg.norm(
            np.hstack([
                z_next_samples - z_star_samples,
                v_next_samples - v_star_samples,
            ]),
            axis=1,
        ))
        z_star_samples = z_next_samples
        v_star_samples = v_next_samples
        if step_norm <= tol:
            break

    return float(np.max(np.linalg.norm(
        np.hstack([
            z0_samples - z_star_samples,
            v0_samples - v_star_samples,
        ]),
        axis=1,
    )))
