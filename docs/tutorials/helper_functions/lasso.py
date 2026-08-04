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


def estimate_lasso_pepit_radius(
    dictionary,
    eta,
    lambd,
    s0_samples,
    x_samples,
    max_iters=2000,
    tol=1e-10,
):
    """Estimate the sampled initial-condition radius used by the Lasso PEP.

    Runs vectorized ISTA from zero to approximate the optimizer associated
    with each sampled measurement, then returns the largest distance from a
    sampled initial iterate to its corresponding optimizer. The result is
    empirical: it covers the supplied samples, not every point in their boxes.
    """
    s_star_samples = np.zeros(
        (x_samples.shape[0], dictionary.shape[1]), dtype=float
    )
    for _ in range(max_iters):
        grad = (s_star_samples @ dictionary.T - x_samples) @ dictionary
        argument = s_star_samples - eta * grad
        s_next_samples = np.sign(argument) * np.maximum(
            np.abs(argument) - lambd * eta,
            0.0,
        )
        step_norm = np.max(
            np.linalg.norm(s_next_samples - s_star_samples, axis=1)
        )
        s_star_samples = s_next_samples
        if step_norm <= tol:
            break

    return float(np.max(np.linalg.norm(s0_samples - s_star_samples, axis=1)))
