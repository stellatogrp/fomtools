"""Shared problem and algorithm helpers for the NNLS tutorials."""

from __future__ import annotations

import numpy as np


def make_nnls_matrix(m: int = 5, n: int = 3, seed: int = 0) -> np.ndarray:
    """Return the deterministic dense matrix used by the NNLS tutorials."""
    rng = np.random.RandomState(seed)
    return rng.normal(size=(m, n))


def pgd_step(VP, A, eta, z_prev, w_prev, x_param, k, momentum=False):
    """Apply one projected-gradient step, optionally with FISTA momentum."""
    w_next = VP.step.relu(z_prev - eta * A.T @ (A @ z_prev - x_param))
    if momentum:
        beta_k = (k - 1) / (k + 2)
        z_next = w_next + beta_k * (w_next - w_prev)
    else:
        z_next = w_next
    return z_next, w_next, z_next - z_prev


def solve_pgd_numpy(A, eta, z0, x, max_iters):
    """Run the non-momentum PGD recurrence with NumPy."""
    z = np.array(z0, dtype=float, copy=True)
    for _ in range(max_iters):
        z = np.maximum(z - eta * A.T @ (A @ z - x), 0.0)
    return z
