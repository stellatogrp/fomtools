"""Built-in example FOMs for the code-generation CLI and demos.

Each builder returns a dict with at least ``verifier`` and ``next_state`` (the
iterate after one iteration), plus a ``name``. These mirror the ISTA and
Douglas-Rachford problems used throughout the test suite and use the
``simulator`` backend so no MIP/SDP solver is required.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import stack
from fomtools.verifier import Verifier


def ista_example(*, n: int = 2, m: int = 4, seed: int = 0, K: int = 3) -> dict:
    """Lasso ISTA: ``z <- soft_threshold(At z + Bt b, lambda t)``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m,))
    lambd = 1.0
    t = 1.0 / np.max(np.real(np.linalg.eigvals(A.T @ A)))
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T

    v = Verifier(method="simulator", obbt_solver=None)
    b_param = v.add_param(m, lb=b - 1.0, ub=b + 1.0)
    z = v.add_iterate(n, lb=2, ub=2)
    iterates = [z]
    for _ in range(K):
        z = v.step.soft_threshold(At @ z + Bt @ b_param, lambd=lambd * t)
        iterates.append(z)
    return {"verifier": v, "next_state": iterates[1], "name": "ista"}


def dr_example(*, n: int = 4, m: int = 3, seed: int = 5, K: int = 3) -> dict:
    """Douglas-Rachford splitting for a small QP (affine solve + projection)."""
    rng = np.random.default_rng(seed)
    Phalf = rng.normal(size=(n, n))
    P = Phalf @ Phalf.T + 0.1 * np.eye(n)
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    b = A @ (rng.normal(size=n) / n)
    M = np.block([[P, A.T], [-A, np.zeros((m, m))]])
    lhs = spa.csc_matrix(M) + spa.eye(m + n)
    q_l = np.hstack([c - 0.1, b])
    q_u = np.hstack([c + 0.1, b])

    v = Verifier(method="simulator", obbt_solver=None)
    q_param = v.add_param(n + m, lb=q_l, ub=q_u)
    z0 = v.add_iterate(m + n, lb=np.zeros(n + m), ub=np.zeros(n + m))
    z = [z0]
    for _ in range(K):
        u = v.step.implicit_linear(z[-1] - q_param, lhs_mat=lhs.todense())
        utilde = v.step.relu(2 * u - z[-1], proj_ranges=(n, n + m))
        z.append(z[-1] + utilde - u)
    return {"verifier": v, "next_state": z[1], "name": "dr"}


def _mpc_qp_data(nx: int, nu: int, N: int, seed: int):
    """Build a model-predictive-control QP in OSQP standard form.

    Returns ``P, M, l, u, KKT`` plus sizes and ADMM parameters for the QP
    ``min 1/2 z'Pz + q'z  s.t.  l <= M z <= u`` over the stacked variable
    ``z = [x_0..x_N, u_0..u_{N-1}]``. The initial state is baked into ``l``/``u``
    (so they are constant); the linear cost ``q`` is left as the parameter.
    """
    rng = np.random.default_rng(seed)
    # Stable-ish discrete dynamics.
    Ad = rng.normal(size=(nx, nx))
    Ad = 0.9 * Ad / np.max(np.abs(np.linalg.eigvals(Ad)))
    Bd = 0.5 * rng.normal(size=(nx, nu))

    Q = np.eye(nx)
    QN = 10.0 * np.eye(nx)
    R = 0.1 * np.eye(nu)
    x_init = rng.uniform(-0.5, 0.5, size=nx)

    nv = nx * (N + 1) + nu * N
    uoff = nx * (N + 1)

    P = np.zeros((nv, nv))
    for t in range(N + 1):
        P[t * nx : (t + 1) * nx, t * nx : (t + 1) * nx] = QN if t == N else Q
    for t in range(N):
        P[uoff + t * nu : uoff + (t + 1) * nu, uoff + t * nu : uoff + (t + 1) * nu] = R

    # Equality rows: x_0 = x_init, and x_{t+1} - Ad x_t - Bd u_t = 0.
    neq = nx * (N + 1)
    Aeq = np.zeros((neq, nv))
    beq = np.zeros(neq)
    Aeq[0:nx, 0:nx] = np.eye(nx)
    beq[0:nx] = x_init
    for t in range(N):
        r = nx * (t + 1)
        Aeq[r : r + nx, (t + 1) * nx : (t + 2) * nx] = np.eye(nx)
        Aeq[r : r + nx, t * nx : (t + 1) * nx] = -Ad
        Aeq[r : r + nx, uoff + t * nu : uoff + (t + 1) * nu] = -Bd

    # Box rows on every variable.
    x_min, x_max, u_min, u_max = -5.0, 5.0, -1.0, 1.0
    lb = np.empty(nv)
    ub = np.empty(nv)
    lb[:uoff] = x_min
    ub[:uoff] = x_max
    lb[uoff:] = u_min
    ub[uoff:] = u_max

    M = np.vstack([Aeq, np.eye(nv)])
    lvec = np.hstack([beq, lb])  # equality rows: l == u == beq
    uvec = np.hstack([beq, ub])
    return P, M, lvec, uvec, nv, M.shape[0], x_init


def mpc_osqp_example(
    *,
    nx: int = 4,
    nu: int = 2,
    N: int = 8,
    seed: int = 1,
    rho: float = 0.1,
    sigma: float = 1e-6,
    alpha: float = 1.6,
) -> dict:
    """One iteration of OSQP's ADMM for an MPC QP.

    State is ``(x, z, y)`` (primal, auxiliary, dual); the parameter is the linear
    cost ``q``. The KKT matrix is constant (fixed problem data), so its inverse
    is folded into the generated coefficients at codegen time. The box projection
    is a saturated-linear step; everything else is affine/linear.
    """
    P, M, lvec, uvec, nv, nc, x_init = _mpc_qp_data(nx, nu, N, seed)
    KKT = np.block([[P + sigma * np.eye(nv), M.T], [M, -(1.0 / rho) * np.eye(nc)]])
    rng = np.random.default_rng(seed + 100)
    q_nom = 0.1 * rng.normal(size=nv)

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(nv, lb=q_nom - 0.05, ub=q_nom + 0.05)
    x = v.add_iterate(nv, lb=np.zeros(nv), ub=np.zeros(nv))
    z = v.add_iterate(nc, lb=np.zeros(nc), ub=np.zeros(nc))
    y = v.add_iterate(nc, lb=np.zeros(nc), ub=np.zeros(nc))

    # Solve the KKT system for (x_tilde, nu_dual).
    rhs = stack([sigma * x - q, z - (1.0 / rho) * y])
    sol = v.step.implicit_linear(rhs, lhs_mat=KKT)
    x_tilde = sol[:nv]
    nu_dual = sol[nv : nv + nc]

    z_tilde = z + (1.0 / rho) * (nu_dual - y)
    proj_arg = alpha * z_tilde + (1.0 - alpha) * z + (1.0 / rho) * y
    z_next = v.step.satlin(proj_arg, l=lvec, u=uvec)
    x_next = alpha * x_tilde + (1.0 - alpha) * x
    y_next = y + rho * (alpha * z_tilde + (1.0 - alpha) * z - z_next)

    next_state = stack([x_next, z_next, y_next])
    return {
        "verifier": v,
        "next_state": next_state,
        "name": "mpc_osqp",
        "data": {
            "P": P,
            "M": M,
            "l": lvec,
            "u": uvec,
            "KKT": KKT,
            "q_nom": q_nom,
            "nv": nv,
            "nc": nc,
            "rho": rho,
            "sigma": sigma,
            "alpha": alpha,
            "x_init": x_init,
        },
    }


def cg_qp_example(*, n: int = 6, seed: int = 3) -> dict:
    """A symmetric positive-definite linear-solve FOM: ``x <- solve(M, x + q)``.

    The affine system ``M`` is SPD, so its solve can use the *indirect* CG
    backend (``solve_method=CG``) on the large-scale targets, or a sparse direct
    factorization on the embedded target — the generated calling convention is
    identical. ``eig(M) >= 2`` makes the outer map a contraction (stable).
    """
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(n, n))
    M = half @ half.T + 2.0 * np.eye(n)

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(n, lb=-np.ones(n), ub=np.ones(n))
    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    iterates = [x]
    for _ in range(3):
        x = v.step.implicit_linear(x + q, lhs_mat=M)
        iterates.append(x)
    return {"verifier": v, "next_state": iterates[1], "name": "cgqp"}


def pdhg_lp_example(*, n: int = 4, m: int = 3, seed: int = 7) -> dict:
    r"""Plain PDHG (Chambolle-Pock) for an LP — no restarts.

    Solves ``min c'x  s.t.  A x >= b,  l <= x <= u`` via the saddle problem
    ``min_{l<=x<=u} max_{y>=0} c'x + y'(b - A x)``. One PDHG iteration (primal
    box projection, then dual nonnegative projection with primal extrapolation):

        x+   = clip(x - tau (c - A' y), l, u)
        y+   = relu(y + sigma (b - A (2 x+ - x)))

    Pure first-order: matrix-vector products + projections, no linear solve. The
    objective ``c`` is the solve-time parameter; ``A, b, l, u`` are fixed. This
    is the LP base of PDLP-style solvers, in its plain (restart-free) form.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    x_feas = rng.uniform(-0.5, 0.5, size=n)
    b = A @ x_feas - 0.5  # feasible: A x_feas >= b
    c_nom = rng.normal(size=n)
    lo, hi = -np.ones(n), np.ones(n)
    normA = float(np.sqrt(np.max(np.real(np.linalg.eigvals(A.T @ A)))))
    tau = sigma = 0.9 / normA

    v = Verifier(method="simulator", obbt_solver=None)
    c = v.add_param(n, lb=c_nom - 0.1, ub=c_nom + 0.1)
    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
    iters = [stack([x, y])]
    for _ in range(3):
        x_next = v.step.satlin(x - tau * c + tau * (A.T @ y), l=lo, u=hi)
        y_next = v.step.relu(y + sigma * b - sigma * (A @ (2 * x_next - x)))
        x, y = x_next, y_next
        iters.append(stack([x, y]))
    return {"verifier": v, "next_state": iters[1], "name": "pdhg_lp"}


def indefinite_kkt_example(*, n: int = 4, seed: int = 4) -> dict:
    r"""An affine solve on a symmetric-INDEFINITE, non-quasidefinite KKT system.

    ``K = [[0, A^T], [A, 0]]`` (zero (1,1) block, ``A`` square nonsingular) is
    symmetric but has an all-zero diagonal, so no-pivot LDL^T breaks down: code
    generation falls back to a pivoted **LU** factorization (never an inverse).
    Singular values of ``A`` are kept in ``[1.5, 3]`` so ``x <- solve(K, x + q)``
    is a contraction. This is the example for "the system is not quasidefinite".
    """
    rng = np.random.default_rng(seed)
    U, _, Vt = np.linalg.svd(rng.normal(size=(n, n)))
    A = U @ np.diag(rng.uniform(1.5, 3.0, size=n)) @ Vt
    K = np.block([[np.zeros((n, n)), A.T], [A, np.zeros((n, n))]])

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(2 * n, lb=-np.ones(2 * n), ub=np.ones(2 * n))
    x = v.add_iterate(2 * n, lb=np.zeros(2 * n), ub=np.zeros(2 * n))
    iters = [x]
    for _ in range(3):
        x = v.step.implicit_linear(x + q, lhs_mat=K)
        iters.append(x)
    return {"verifier": v, "next_state": iters[1], "name": "indef_kkt"}


EXAMPLES = {
    "ista": ista_example,
    "dr": dr_example,
    "mpc": mpc_osqp_example,
    "cgqp": cg_qp_example,
    "pdhg_lp": pdhg_lp_example,
    "indef_kkt": indefinite_kkt_example,
}


def get_example(name: str) -> dict:
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; choices: {sorted(EXAMPLES)}")
    return EXAMPLES[name]()
