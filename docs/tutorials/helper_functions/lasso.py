"""Shared problem and algorithm helpers for the Lasso tutorials."""

from __future__ import annotations

import numpy as np


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
    """Apply one ISTA step, optionally with FISTA momentum."""
    del k  # The standard FISTA coefficient is carried by ``t_prev``.
    y_next = VP.step.soft_threshold(
        z_prev - eta * dictionary.T @ (dictionary @ z_prev - x_param),
        lambd=lambd * eta,
    )
    if momentum:
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        z_next = y_next + ((t_prev - 1.0) / t_next) * (y_next - y_prev)
        residual = y_next - y_prev
    else:
        t_next, z_next, residual = t_prev, y_next, y_next - z_prev
    return z_next, y_next, t_next, residual


def solve_ista_numpy(dictionary, eta, lambd, s0, x, max_iters):
    """Run the non-momentum ISTA recurrence with NumPy."""
    s = np.array(s0, dtype=float, copy=True)
    for _ in range(max_iters):
        argument = s - eta * dictionary.T @ (dictionary @ s - x)
        s = np.sign(argument) * np.maximum(np.abs(argument) - lambd * eta, 0.0)
    return s
