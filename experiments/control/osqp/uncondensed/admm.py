"""The ADMM iteration of the paper on the uncondensed program.

Equation (16), with a general constraint matrix rather than the identity, and
with a per-row penalty rather than a scalar:

    w    = sat_[l(x), u(x)](v),
    zeta = solve(P + sigma I + M' R M,  sigma zeta - q + M' R (2 w - v)),
    v    = v + M zeta - w.

The same three steps as the condensed version: one saturated-linear step and
two affine steps.  What differs is that ``M`` is no longer the identity, so each
iteration performs two sparse matrix-vector products in addition to the linear
solve, and the solve applies a sparse factorization rather than a dense one.

The bounds are those of the current sampling instant, because the measured state
enters the program through them.  This is the part that has no counterpart in
the condensed formulation, where the bounds are fixed once and for all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncondensed.mpc import UncondensedProblem


@dataclass
class AdmmResult:
    """Outcome of a fixed number of ADMM iterations."""

    zeta: np.ndarray
    v: np.ndarray
    iterations: int
    residual: float | None = None
    residuals: np.ndarray | None = None


def solve(
    problem: UncondensedProblem,
    q: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    iterations: int,
    zeta0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    trace: bool = False,
) -> AdmmResult:
    """Run ``iterations`` ADMM steps on the program of ``problem``.

    ``lo`` and ``hi`` come from :meth:`UncondensedProblem.bounds`, which places
    the measured state on the initial-state rows.
    """
    M, rho, sigma = problem.M, problem.rho, problem.sigma
    Mt = M.T

    zeta = np.zeros(problem.n) if zeta0 is None else np.asarray(zeta0, dtype=float).copy()
    v = np.zeros(problem.m) if v0 is None else np.asarray(v0, dtype=float).copy()

    residuals = np.empty(iterations) if trace else None
    residual = None

    for k in range(iterations):
        zeta_prev, v_prev = zeta, v
        w = np.minimum(np.maximum(v, lo), hi)
        zeta = problem.solve_system(sigma * zeta - q + Mt @ (rho * (2.0 * w - v)))
        v = v + M @ zeta - w
        if trace or k == iterations - 1:
            residual = scaled_residual(problem, zeta, v, zeta_prev, v_prev)
            if trace:
                residuals[k] = residual

    return AdmmResult(zeta=zeta, v=v, iterations=iterations, residual=residual, residuals=residuals)


def scaled_residual(
    problem: UncondensedProblem,
    zeta: np.ndarray,
    v: np.ndarray,
    zeta_prev: np.ndarray,
    v_prev: np.ndarray,
) -> float:
    """Infinity norm of the fixed-point residual mapped to optimality residuals.

    Equation (17) with a general constraint matrix and a per-row penalty:
    ``H = -[[sigma I, M' R], [0, I]]``.
    """
    dzeta = zeta - zeta_prev
    dv = v - v_prev
    upper = problem.sigma * dzeta + problem.M.T @ (problem.rho * dv)
    return float(max(np.max(np.abs(upper)), np.max(np.abs(dv))))


def solve_exact(problem: UncondensedProblem, q: np.ndarray, lo, hi) -> np.ndarray:
    """Solve the same quadratic program to high accuracy with OSQP."""
    import osqp

    prob = osqp.OSQP()
    prob.setup(
        P=problem.P,
        q=q,
        A=problem.M,
        l=lo,
        u=hi,
        verbose=False,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=200_000,
    )
    result = prob.solve()
    if result.x is None or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"OSQP failed on the reference solve: {result.info.status}")
    return result.x
