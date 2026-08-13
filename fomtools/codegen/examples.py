"""Built-in example FOMs for the code-generation CLI and demos.

Most builders return a dict with ``verifier`` and ``next_state`` (the iterate
after one iteration) plus a ``name``, which generates a rolled solver. These
mirror the ISTA and Douglas-Rachford problems used throughout the test suite and
use the ``simulator`` backend so no MIP/SDP solver is required.

Two builders show the other generation paths. :func:`fista_example` returns
``iterates`` instead: a recorded *run*, generated as written, which is how a
method whose coefficients change every iteration is handled without declaring
anything. :func:`scheduled_example` returns a ``build`` function and its
``schedules``, which trades that generality for a compact solver at a long
horizon — one block plus a table rather than one block per iteration. See
:mod:`fomtools.codegen.unrolled` and :mod:`fomtools.codegen.scheduled`.

Algorithm coefficients are written out here rather than imported: they belong to
the method being demonstrated, not to the code generator.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import stack
from fomtools.verifier import Verifier


def _banded_spd(n: int, bandwidth: int, seed: int, scale: float = 0.3) -> np.ndarray:
    """A sparse, symmetric, strictly diagonally-dominant (hence SPD) banded matrix.

    Off-diagonals live within ``bandwidth`` of the diagonal; the diagonal is set to
    its row's off-diagonal absolute sum plus ``2`` — so every eigenvalue is ``>= 2``
    (Gershgorin), making ``x <- solve(M, x + q)`` a contraction, and the matrix is
    SPD without computing a single eigenvalue. Structurally sparse (``~n(2b+1)``
    nonzeros), so the generated code emits a CSR matrix-vector product rather than a
    dense ``n^2`` block — the point of a GPU-scale CG example.
    """
    rng = np.random.default_rng(seed)
    M = np.zeros((n, n))
    for k in range(1, bandwidth + 1):
        band = scale * rng.uniform(-1.0, 1.0, size=n - k)
        idx = np.arange(n - k)
        M[idx, idx + k] = band
        M[idx + k, idx] = band
    np.fill_diagonal(M, np.abs(M).sum(axis=1) + 2.0)
    return M


def _sparse_dense_matrix(m: int, n: int, density: float, rng) -> np.ndarray:
    """A structurally sparse ``m x n`` matrix (dense storage, mostly zeros).

    Each row has ``max(1, round(density * n))`` random nonzeros. Returned as a
    dense array (the verifier records dense coefficients), but the codegen detects
    the sparsity and emits a CSR matvec.
    """
    A = np.zeros((m, n))
    nnz_per_row = max(1, round(density * n))
    for i in range(m):
        cols = rng.choice(n, size=min(nnz_per_row, n), replace=False)
        A[i, cols] = rng.normal(size=cols.size)
    return A


def _spectral_norm_bound(A: np.ndarray) -> float:
    """Cheap upper bound on ``||A||_2`` via ``sqrt(||A||_1 * ||A||_inf)``.

    O(nnz), no eigendecomposition — and an *upper* bound, so PDHG step sizes
    ``tau = sigma = 0.9 / bound`` keep ``tau*sigma*||A||_2^2 <= 0.81 < 1`` (stable).
    Used for large/sparse instances where the exact ``eigvals`` is O(n^3).
    """
    one = float(np.max(np.abs(A).sum(axis=0))) if A.size else 0.0
    inf = float(np.max(np.abs(A).sum(axis=1))) if A.size else 0.0
    return float(np.sqrt(one * inf)) or 1.0


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


def _lasso_instance(n: int, m: int, seed: int) -> dict:
    """Shared data for the accelerated lasso examples.

    A milder threshold than :func:`ista_example` uses: at ``lambda = 1`` this
    instance has the all-zero solution and converges in a single step, which
    would make acceleration indistinguishable from plain ISTA.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m,))
    lambd = 0.05
    t = 1.0 / np.max(np.real(np.linalg.eigvals(A.T @ A)))
    return {
        "At": np.eye(n) - t * A.T @ A,
        "Bt": t * A.T,
        "lambd": lambd,
        "t": t,
        "z0": np.full(n, 2.0),
        "n": n,
        "m": m,
        "b": b,
    }


def fista_example(*, n: int = 2, m: int = 4, seed: int = 0, K: int = 6) -> dict:
    """Lasso FISTA, written as the loop a user would write to verify it.

    The momentum coefficient changes every iteration, so this example returns
    ``iterates`` -- the recorded run -- rather than a single ``next_state``.
    Generation emits the run as written, one block per iteration, each carrying
    its own coefficients. Nothing has to be declared and the momentum recurrence
    is ordinary Python, evaluated while recording.

    Contrast :func:`scheduled_example`, which is the same method declared as a
    schedule to get one block plus a table instead.
    """
    d = _lasso_instance(n, m, seed)
    At, Bt, z0 = d["At"], d["Bt"], d["z0"]
    threshold = d["lambd"] * d["t"]

    v = Verifier(method="simulator", obbt_solver=None)
    b_param = v.add_param(m, lb=d["b"] - 1.0, ub=d["b"] + 1.0)
    y0 = v.add_iterate(n, lb=z0, ub=z0)

    y_prev, y_curr, beta = y0, y0, 1.0
    iterates = []
    for _ in range(K):
        beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
        gamma = (beta - 1.0) / beta_next
        extrap = y_curr + gamma * (y_curr - y_prev)
        y_next = v.step.soft_threshold(At @ extrap + Bt @ b_param, lambd=threshold)
        y_prev, y_curr, beta = y_curr, y_next, beta_next
        iterates.append(y_next)

    return {
        "verifier": v,
        "iterates": iterates,
        "param": b_param,
        "name": "fista",
        "data": {**d, "K": K},
    }


def scheduled_example(*, n: int = 2, m: int = 4, seed: int = 0) -> dict:
    """The same accelerated lasso, declared as a schedule instead.

    Returns a ``build`` function rather than a finished verifier, which is what
    the scheduled path needs: the momentum coefficient stays a free variable so
    extraction can probe the recording at several of its values and recover how
    each coefficient depends on it (see :mod:`fomtools.codegen.scheduled`). The
    reward is one emitted block and a table of coefficients, whatever the horizon.

    The state carries **two** iterates. Written over a single iterate, the
    dependency cone of iteration ``k`` reaches back through every earlier step and
    extraction returns the unrolled prefix rather than a repeatable body -- which
    is the right answer for :func:`fista_example` and the wrong one here.
    Declaring ``(y^{k-1}, y^k)`` as the state closes the cone after one step.
    """
    d = _lasso_instance(n, m, seed)
    At, Bt, z0 = d["At"], d["Bt"], d["z0"]
    threshold = d["lambd"] * d["t"]

    def build(gamma):
        from fomtools.codegen.scheduled import Recording

        v = Verifier(method="simulator", obbt_solver=None)
        b_param = v.add_param(m, lb=d["b"] - 1.0, ub=d["b"] + 1.0)
        y_prev = v.add_iterate(n, lb=z0, ub=z0)
        y_curr = v.add_iterate(n, lb=z0, ub=z0)
        extrap = y_curr + gamma * (y_curr - y_prev)
        y_next = v.step.soft_threshold(At @ extrap + Bt @ b_param, lambd=threshold)
        # The state shifts: (y_prev, y_curr) <- (y_curr, y_next).
        return Recording(v, next_state=[y_curr, y_next], state=[y_prev, y_curr])

    def momentum():
        """FISTA's momentum in the 0-based index of the emitted loop.

        From ``beta_0 = 1``, ``beta_{k+1} = (1 + sqrt(1 + 4 beta_k^2))/2``, giving
        ``gamma_k = (beta_{k-1} - 1)/beta_k``. The textbook form is 1-based, so
        the first value is 0 -- exactly the degeneracy that makes a
        single-iteration recording of this method indistinguishable from ISTA,
        which is why the coefficient has to be tabulated rather than read off one
        recorded iteration. The closure caches its recurrence, so it must be
        called with non-decreasing indices, as ``as_schedule`` does.
        """
        betas = [1.0]

        def gamma(k: int) -> float:
            while len(betas) <= k + 1:
                last = betas[-1]
                betas.append((1.0 + np.sqrt(1.0 + 4.0 * last * last)) / 2.0)
            return (betas[k] - 1.0) / betas[k + 1]

        return gamma

    return {
        "build": build,
        "schedules": {"gamma": momentum()},
        "name": "scheduled",
        "data": d,
    }


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
        u = v.step.implicit_linear(z[-1] - q_param, output_mat=lhs.todense())
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
    sol = v.step.implicit_linear(rhs, output_mat=KKT)
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


def cg_qp_example(*, n: int = 6, seed: int = 3, bandwidth: int | None = None) -> dict:
    """A symmetric positive-definite linear-solve FOM: ``x <- solve(M, x + q)``.

    The affine system ``M`` is SPD, so its solve can use the *indirect* CG
    backend (``solve_method=CG``) on the large-scale targets, or a sparse direct
    factorization on the embedded target — the generated calling convention is
    identical. ``eig(M) >= 2`` makes the outer map a contraction (stable).

    ``bandwidth`` selects the matrix structure:

    * ``None`` (default): a *dense* random SPD matrix ``half @ half.T + 2 I`` —
      fine for small verification-sized instances.
    * an integer ``b``: a *sparse banded* SPD matrix (off-diagonals within ``b``
      of the diagonal). This is the GPU-scale form — the generated CG solve then
      runs a CSR sparse matrix-vector product instead of a dense ``n^2`` one, so
      memory and work stay ``O(n*b)``. Use this for large ``n``.
    """
    if bandwidth is None:
        rng = np.random.default_rng(seed)
        half = rng.normal(size=(n, n))
        M = half @ half.T + 2.0 * np.eye(n)
    else:
        M = _banded_spd(n, bandwidth, seed)

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(n, lb=-np.ones(n), ub=np.ones(n))
    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    iterates = [x]
    for _ in range(3):
        x = v.step.implicit_linear(x + q, output_mat=M)
        iterates.append(x)
    return {"verifier": v, "next_state": iterates[1], "name": "cgqp"}


def pdhg_lp_example(*, n: int = 4, m: int = 3, seed: int = 7, density: float | None = None) -> dict:
    r"""Plain PDHG (Chambolle-Pock) for an LP — no restarts.

    Solves ``min c'x  s.t.  A x >= b,  l <= x <= u`` via the saddle problem
    ``min_{l<=x<=u} max_{y>=0} c'x + y'(b - A x)``. One PDHG iteration (primal
    box projection, then dual nonnegative projection with primal extrapolation):

        x+   = clip(x - tau (c - A' y), l, u)
        y+   = relu(y + sigma (b - A (2 x+ - x)))

    Pure first-order: matrix-vector products + projections, no linear solve. The
    objective ``c`` is the solve-time parameter; ``A, b, l, u`` are fixed. This
    is the LP base of PDLP-style solvers, in its plain (restart-free) form.

    ``density`` selects the constraint matrix structure:

    * ``None`` (default): a *dense* random ``A`` with step sizes from the exact
      spectral norm (``eigvals``, O(n^3)) — verification-sized only.
    * a fraction in ``(0, 1]``: a *sparse* ``A`` (that fraction of entries
      nonzero per row) with step sizes from a cheap O(nnz) spectral-norm upper
      bound. This is the GPU-scale form — the generated code then runs CSR
      matrix-vector products. Use this for large ``n``/``m``.
    """
    rng = np.random.default_rng(seed)
    if density is None:
        A = rng.normal(size=(m, n))
        normA = float(np.sqrt(np.max(np.real(np.linalg.eigvals(A.T @ A)))))
    else:
        A = _sparse_dense_matrix(m, n, density, rng)
        normA = _spectral_norm_bound(A)
    x_feas = rng.uniform(-0.5, 0.5, size=n)
    b = A @ x_feas - 0.5  # feasible: A x_feas >= b
    c_nom = rng.normal(size=n)
    lo, hi = -np.ones(n), np.ones(n)
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
        x = v.step.implicit_linear(x + q, output_mat=K)
        iters.append(x)
    return {"verifier": v, "next_state": iters[1], "name": "indef_kkt"}


EXAMPLES = {
    "ista": ista_example,
    "fista": fista_example,
    "scheduled": scheduled_example,
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
