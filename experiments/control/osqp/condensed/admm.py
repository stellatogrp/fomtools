"""The ADMM iteration of the OSQP solver, in NumPy.

This is the algorithm of equation (16) of the paper, specialized to a constraint
matrix equal to the identity, which is what the condensed formulation of
``mpc.py`` produces.  Writing the state as ``s = (z, v)``, one iteration is

    w  = sat_[l, u](v),
    z  = solve(P + sigma I + rho I,  sigma z - q + rho (2 w - v)),
    v  = v + z - w.

One saturated-linear step and two affine steps, the second of which is a linear
system solve.  The matrix of that system is fixed by the problem data, so it is
factorized once when the problem is built and only applied here.  That is the
honest counterpart of the generated code, which applies a factorization computed
at generation time; refactorizing inside the loop would overstate what code
generation contributes.

The iteration is deliberately written as the paper writes it rather than in a
fused or preallocated form.  It is the object the rest of the experiment
measures, and stage two of the experiment generates C for exactly this
specification, so the two must be recognizably the same algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from condensed.mpc import MpcProblem


@dataclass
class AdmmResult:
    """Outcome of a fixed number of ADMM iterations."""

    z: np.ndarray
    v: np.ndarray
    iterations: int
    #: Infinity norm of the scaled fixed-point residual at the last iteration,
    #: or ``None`` when fewer than one iteration was run.
    residual: float | None = None
    #: Per-iteration residuals, present only when ``trace`` was requested.
    residuals: np.ndarray | None = None


def solve(
    problem: MpcProblem,
    q: np.ndarray,
    iterations: int,
    z0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    trace: bool = False,
) -> AdmmResult:
    """Run ``iterations`` ADMM steps on the program of ``problem`` at cost ``q``.

    The initial iterate defaults to the origin, matching the cold start that the
    generated solver performs on every parameter update.
    """
    n = problem.n
    lo, hi = problem.lo, problem.hi
    rho, sigma = problem.rho, problem.sigma

    z = np.zeros(n) if z0 is None else np.asarray(z0, dtype=float).copy()
    v = np.zeros(n) if v0 is None else np.asarray(v0, dtype=float).copy()

    residuals = np.empty(iterations) if trace else None
    residual = None

    for k in range(iterations):
        z_prev, v_prev = z, v
        w = np.minimum(np.maximum(v, lo), hi)
        z = problem.solve_system(sigma * z - q + rho * (2.0 * w - v))
        v = v + z - w
        if trace or k == iterations - 1:
            residual = scaled_residual(problem, z, v, z_prev, v_prev)
            if trace:
                residuals[k] = residual

    return AdmmResult(z=z, v=v, iterations=iterations, residual=residual, residuals=residuals)


def scaled_residual(
    problem: MpcProblem,
    z: np.ndarray,
    v: np.ndarray,
    z_prev: np.ndarray,
    v_prev: np.ndarray,
) -> float:
    """Infinity norm of the fixed-point residual mapped to optimality residuals.

    Equation (17) of the paper: the primal and dual residuals of the quadratic
    program are the linear image ``H (s - s_prev)`` of the fixed-point residual,
    with ``H = -[[sigma I, rho M'], [0, I]]``.  With ``M = I`` this is

        primal-dual block:  -(sigma (z - z_prev) + rho (v - v_prev)),
        primal block:       -(v - v_prev),

    so the residual the user cares about is available without evaluating the
    optimality conditions directly.
    """
    dz = z - z_prev
    dv = v - v_prev
    upper = problem.sigma * dz + problem.rho * dv
    return float(max(np.max(np.abs(upper)), np.max(np.abs(dv))))


def solve_exact(problem: MpcProblem, q: np.ndarray) -> np.ndarray:
    """Solve the same quadratic program to high accuracy with OSQP.

    Used only to check the iteration, never inside the closed loop.
    """
    import scipy.sparse as sp

    import osqp

    n = problem.n
    prob = osqp.OSQP()
    prob.setup(
        P=sp.csc_matrix(problem.P),
        q=q,
        A=sp.eye(n, format="csc"),
        l=problem.lo,
        u=problem.hi,
        verbose=False,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=100_000,
    )
    result = prob.solve()
    if result.x is None or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"OSQP failed on the reference solve: {result.info.status}")
    return result.x
