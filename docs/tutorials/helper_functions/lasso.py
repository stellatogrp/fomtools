"""Shared problem and algorithm helpers for the Lasso tutorials."""

from __future__ import annotations

import numpy as np

from fomtools import IterateTrajectory, LinearExpression


def make_lasso_data(
    p: int = 3,
    n: int = 4,
    lambd: float = 1e-1,
    num_signals: int = 20,
    seed: int = 0,
):
    """Return the deterministic dictionary and measurement box used in the tutorials."""
    rng = np.random.RandomState(seed)
    dictionary = rng.normal(size=(p, n)) * rng.binomial(n=1, p=0.2, size=(p, n))
    signals = rng.normal(size=(n, num_signals)) * rng.binomial(
        n=1, p=0.1, size=(n, num_signals)
    )
    measurements = dictionary @ signals + rng.normal(
        scale=1e-2, size=(p, num_signals)
    )
    x_lb = np.min(measurements, axis=1)
    x_ub = np.max(measurements, axis=1)
    L = float(np.linalg.eigvalsh(dictionary.T @ dictionary)[-1])
    eta = 1.0 / L
    return dictionary, x_lb, x_ub, L, eta, lambd


def ista_step(
    VP,
    dictionary,
    eta,
    lambd,
    z_prev,
    y_prev,
    x_param,
    t_prev,
    k,
    momentum=False,
):
    """Apply one ISTA step, optionally with FISTA momentum.

    ``z_prev`` may be either the current iterate expression or an
    :class:`~fomtools.iteratetrajectory.IterateTrajectory`. In the latter
    case, slot ``k - 1`` is used as the current iterate and the computed next
    iterate is assigned to slot ``k``.
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

    y_next = VP.step.soft_threshold(
        current - eta * dictionary.T @ (dictionary @ current - x_param),
        lambd=lambd * eta,
    )
    if momentum:
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        z_next = y_next + ((t_prev - 1.0) / t_next) * (y_next - y_prev)
        residual = y_next - y_prev
    else:
        t_next, z_next, residual = t_prev, y_next, y_next - current

    if trajectory is not None:
        trajectory[k] = z_next
        z_next = trajectory[k]
        z_out = trajectory
    else:
        z_out = z_next

    return z_out, y_next, t_next, residual


def solve_ista_numpy(dictionary, eta, lambd, s0, x, max_iters):
    """Run the non-momentum ISTA recurrence with NumPy."""
    s = np.array(s0, dtype=float, copy=True)
    for _ in range(max_iters):
        argument = s - eta * dictionary.T @ (dictionary @ s - x)
        s = np.sign(argument) * np.maximum(np.abs(argument) - lambd * eta, 0.0)
    return s


def certify_lasso_pepit_radius(lambd, s0_lb, s0_ub, x_lb, x_ub, n):
    """Return a certified initial-condition radius for the Lasso PEP.

    Unlike :func:`estimate_lasso_pepit_radius`, this covers every point in the
    boxes rather than only the drawn samples, so the resulting PEPit curve is a
    valid upper bound for the set the verification problem searches.

    Comparing the Lasso objective at any solution ``s*(x)`` against its value at
    the origin gives ``0.5||D s* - x||^2 + lambd ||s*||_1 <= 0.5||x||^2``, hence
    ``||s*||_2 <= ||s*||_1 <= ||x||^2 / (2 lambd)``. Maximizing ``||x||^2`` and
    ``||s0||_2`` over their boxes and applying the triangle inequality bounds
    ``||s0 - s*(x)||_2`` uniformly. No strong convexity is needed, which matters
    here because ``D`` has more columns than rows.
    """
    x_lb = np.broadcast_to(np.asarray(x_lb, dtype=float), (len(np.atleast_1d(x_lb)),))
    x_ub = np.broadcast_to(np.asarray(x_ub, dtype=float), x_lb.shape)
    s0_lb = np.broadcast_to(np.asarray(s0_lb, dtype=float), (n,))
    s0_ub = np.broadcast_to(np.asarray(s0_ub, dtype=float), (n,))

    max_x_squared = float(np.sum(np.maximum(x_lb**2, x_ub**2)))
    s_star_bound = max_x_squared / (2.0 * lambd)
    s0_bound = float(np.linalg.norm(np.maximum(np.abs(s0_lb), np.abs(s0_ub))))
    return s0_bound + s_star_bound
