"""Shared problem and algorithm helpers for the portfolio tutorials."""

from __future__ import annotations

import numpy as np

from fomtools import IterateTrajectory


def make_portfolio_data(n: int = 4, seed: int = 0, position_limit: float = 0.6):
    """Return deterministic mean-variance data and the portfolio constraint box."""
    rng = np.random.RandomState(seed)
    factor = rng.normal(size=(n, n))
    covariance = (factor @ factor.T) / n + 0.1 * np.eye(n)
    expected_return = rng.uniform(0.05, 0.15, size=n)
    upper_weight = position_limit * np.ones(n)
    constraint_matrix = np.vstack([np.eye(n), np.ones((1, n))])
    lower = np.hstack([np.zeros(n), [1.0]])
    upper = np.hstack([upper_weight, [1.0]])
    return covariance, expected_return, constraint_matrix, lower, upper


def admm_step(
    VP,
    constraint_matrix,
    lower,
    upper,
    expected_return,
    rho,
    sigma,
    x_prev,
    v_prev,
    lhs_mat,
    k=None,
):
    """Apply one OSQP-style ADMM step for the portfolio problem.

    ``x_prev`` and ``v_prev`` may be current expressions or backend-aware
    trajectories. Trajectories read slot ``k - 1`` and store the next state in
    slot ``k``.
    """
    x_trajectory = x_prev if isinstance(x_prev, IterateTrajectory) else None
    v_trajectory = v_prev if isinstance(v_prev, IterateTrajectory) else None
    if (x_trajectory is not None or v_trajectory is not None) and k is None:
        raise ValueError("k is required when admm_step receives a trajectory.")

    x_current = x_trajectory[k - 1] if x_trajectory is not None else x_prev
    v_current = v_trajectory[k - 1] if v_trajectory is not None else v_prev

    w_next = VP.step.satlin(v_current, l=lower, u=upper)
    rhs = (
        sigma * x_current
        + expected_return
        + rho * constraint_matrix.T @ (2 * w_next - v_current)
    )
    x_next = VP.step.implicit_linear(rhs, output_mat=lhs_mat)
    if x_trajectory is not None:
        x_trajectory[k] = x_next
        x_next = x_trajectory[k]

    v_next = v_current + constraint_matrix @ x_next - w_next
    if v_trajectory is not None:
        v_trajectory[k] = v_next
        v_next = v_trajectory[k]

    return (
        x_trajectory if x_trajectory is not None else x_next,
        v_trajectory if v_trajectory is not None else v_next,
        [x_next - x_current, v_next - v_current],
    )


def solve_admm_numpy(
    constraint_matrix,
    lower,
    upper,
    expected_return,
    rho,
    sigma,
    lhs_mat,
    x0,
    v0,
    max_iters,
):
    """Run the portfolio ADMM recurrence with NumPy."""
    x = np.array(x0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    lhs = lhs_mat.toarray() if hasattr(lhs_mat, "toarray") else lhs_mat
    for _ in range(max_iters):
        w = np.clip(v, lower, upper)
        rhs = sigma * x + expected_return + rho * constraint_matrix.T @ (2 * w - v)
        x = np.linalg.solve(lhs, rhs)
        v = v + constraint_matrix @ x - w
    return np.concatenate([x, v])
