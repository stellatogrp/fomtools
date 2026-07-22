"""Shared problem and algorithm helpers for the portfolio tutorials."""

from __future__ import annotations

import numpy as np


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
):
    """Apply one OSQP-style ADMM step for the portfolio problem."""
    w_next = VP.step.satlin(v_prev, l=lower, u=upper)
    rhs = (
        sigma * x_prev
        + expected_return
        + rho * constraint_matrix.T @ (2 * w_next - v_prev)
    )
    x_next = VP.step.implicit_linear(rhs, output_mat=lhs_mat)
    v_next = v_prev + constraint_matrix @ x_next - w_next
    return x_next, v_next, [x_next - x_prev, v_next - v_prev]


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
