"""Shared problem and algorithm helpers for the NNLS tutorials."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from fomtools import IterateTrajectory, LinearExpression


def make_nnls_matrix(m: int = 5, n: int = 3, seed: int = 0) -> np.ndarray:
    """Return the deterministic dense matrix used by the NNLS tutorials."""
    rng = np.random.RandomState(seed)
    return rng.normal(size=(m, n))


def pgd_step(vp, A, eta, z_prev, w_prev, x_param, k, momentum=False):
    """Apply one projected-gradient step, optionally with FISTA momentum.

    ``z_prev`` may be either the current iterate expression or an
    :class:`~fomtools.iteratetrajectory.IterateTrajectory`.  In the latter
    case, slot ``k - 1`` is used as the current iterate and the computed next
    iterate is assigned to slot ``k``.  Returning the same trajectory keeps
    the usual ``z, w, residual = pgd_step(...)`` loop convention intact.
    """
    if isinstance(z_prev, IterateTrajectory):
        trajectory = z_prev
        current = trajectory[k - 1]
    elif isinstance(z_prev, LinearExpression):
        trajectory = None
        current = z_prev
    else:
        raise TypeError(
            "z_prev must be a LinearExpression or IterateTrajectory, got "
            f"{type(z_prev).__name__}."
        )

    w_next = vp.relu(current - eta * A.T @ (A @ current - x_param))
    if momentum:
        beta_k = (k - 1) / (k + 2)
        z_next = w_next + beta_k * (w_next - w_prev)
    else:
        z_next = w_next

    if trajectory is not None:
        trajectory[k] = z_next
        z_next = trajectory[k]
        z_out = trajectory
    else:
        z_out = z_next

    return z_out, w_next, z_next - current


def solve_pgd_numpy(A, eta, z0, x, max_iters):
    """Run the non-momentum PGD recurrence with NumPy."""
    z = np.array(z0, dtype=float, copy=True)
    for _ in range(max_iters):
        z = np.maximum(z - eta * A.T @ (A @ z - x), 0.0)
    return z


def certify_nnls_pepit_radius(A, x_lb, x_ub, z0_lb, z0_ub):
    """Return a certified initial-condition radius for the NNLS PEP.

    The NNLS solution map is ``1 / sigma_min(A)``-Lipschitz. The returned
    radius combines the farthest initial-box point from the solution at the
    center of the parameter box with this Lipschitz constant times the
    parameter box's Euclidean circumradius.
    """
    m, n = A.shape
    x_lb = np.broadcast_to(np.asarray(x_lb, dtype=float), (m,))
    x_ub = np.broadcast_to(np.asarray(x_ub, dtype=float), (m,))
    z0_lb = np.broadcast_to(np.asarray(z0_lb, dtype=float), (n,))
    z0_ub = np.broadcast_to(np.asarray(z0_ub, dtype=float), (n,))

    sigma_min = np.sqrt(np.min(np.linalg.eigvalsh(A.T @ A)))
    x_center = 0.5 * (x_lb + x_ub)
    z_star_center = nnls(A, x_center)[0]
    z0_far = np.maximum(
        np.abs(z0_ub - z_star_center),
        np.abs(z0_lb - z_star_center),
    )
    x_circumradius = 0.5 * np.linalg.norm(x_ub - x_lb)
    return float(np.linalg.norm(z0_far) + x_circumradius / sigma_min)
