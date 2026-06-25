"""Per-sample subproblem evaluators for the Simulator backend.

Implicit-iterate proxes/steps whose output is defined by a KKT system with
*complementarity* (the active set varies per sample) cannot be evaluated with a
single stacked linear solve. This module provides the activation closures the
Simulator uses to evaluate them over a batch of ``N`` samples:

- **Batched, solver-free** projections onto the simplex and the ℓ1-ball (and, by
  Moreau, the ℓ∞-norm prox). Fully vectorized over the batch; run on the
  simulator's array module (numpy on CPU, cupy on GPU).
- **Condat's O(n) direct algorithm** for the 1-D total-variation prox, looped over
  samples (solver-free).
- **cvxpy** projection / QP / LP solves for the genuinely combinatorial cases
  (polyhedral projection, QP/LP with inequalities): one parametrized (DPP) problem
  compiled once and solved per sample.

Each public factory returns an ``activation(batch) -> out`` closure consumed by
``Simulator.register_subproblem_step``; ``batch`` is the assembled input of shape
``(N, d)`` and ``out`` has shape ``(N, n)``.
"""

import numpy as np

try:  # pragma: no cover - exercised only on GPU
    import cupy as _cp
except Exception:  # noqa: BLE001 - cupy is optional
    _cp = None


def _xp(arr):
    """Return the array module (cupy or numpy) backing ``arr``."""
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp
    return np


def _to_numpy(arr) -> np.ndarray:
    """Move ``arr`` to host numpy (no-op if already numpy)."""
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


def _to_like(host: np.ndarray, ref):
    """Move host numpy ``host`` back to the device of ``ref``."""
    if _cp is not None and isinstance(ref, _cp.ndarray):
        return _cp.asarray(host)
    return host


# ----------------------------------------------------------------------------
# Batched, solver-free projections
# ----------------------------------------------------------------------------


def project_simplex_batch(V, a: float):
    r"""
    Euclidean projection of each row of ``V`` onto ``{u >= 0, 1^T u = a}``.

    Vectorized Held–Wolfe / Duchi (2008) sort algorithm: for ``a > 0`` the
    threshold ``theta`` is found per row from the descending sort and cumulative
    sums, and the projection is ``max(V - theta, 0)``. No Python loop over the
    batch.

    Parameters
    ----------
    V : (N, n) array (numpy or cupy)
        Points to project (one per row).
    a : float
        Positive simplex sum.

    Returns
    -------
    (N, n) array
        The row-wise projections.
    """
    xp = _xp(V)
    N, n = V.shape
    U = xp.flip(xp.sort(V, axis=1), axis=1)  # each row sorted descending
    cssv = xp.cumsum(U, axis=1) - a
    ind = xp.arange(1, n + 1)
    rho = xp.sum((U - cssv / ind) > 0, axis=1)  # >= 1 since a > 0
    theta = cssv[xp.arange(N), rho - 1] / rho
    return xp.maximum(V - theta[:, None], 0.0)


def project_l1_ball_batch(V, r: float):
    r"""
    Euclidean projection of each row of ``V`` onto ``{u : ||u||_1 <= r}``.

    Rows already inside the ball are returned unchanged; rows outside are
    projected via ``sign(V) * project_simplex_batch(|V|, r)``. No Python loop.

    Parameters
    ----------
    V : (N, n) array (numpy or cupy)
        Points to project.
    r : float
        Ball radius (``>= 0``).

    Returns
    -------
    (N, n) array
        The row-wise projections.
    """
    xp = _xp(V)
    if r <= 0.0:
        return xp.zeros_like(V)
    absV = xp.abs(V)
    over = absV.sum(axis=1) > r
    w = xp.sign(V) * project_simplex_batch(absV, r)
    return xp.where(over[:, None], w, V)


def simplex_activation(a: float):
    """Activation projecting each sample onto the scaled simplex."""
    return lambda V: project_simplex_batch(V, a)


def l1_ball_activation(r: float):
    """Activation projecting each sample onto the ℓ1-ball of radius ``r``."""
    return lambda V: project_l1_ball_batch(V, r)


def norm_inf_activation(radius: float):
    """Activation for the ℓ∞-norm prox via Moreau: ``v - proj_{||.||_1 <= radius}(v)``."""
    return lambda V: V - project_l1_ball_batch(V, radius)


# ----------------------------------------------------------------------------
# 1-D total variation — Condat's direct algorithm
# ----------------------------------------------------------------------------


def condat_tv_1d(y: np.ndarray, lam: float) -> np.ndarray:
    r"""
    1-D total-variation prox via Condat's O(n) direct algorithm.

    Computes ``argmin_x 1/2 ||x - y||^2 + lam * sum_i |x_{i+1} - x_i|`` exactly in a
    single forward pass (Condat, "A Direct Algorithm for 1-D Total Variation
    Denoising", 2013). ``lam <= 0`` or length ``< 2`` returns a copy of ``y``.

    Parameters
    ----------
    y : (n,) np.ndarray
        Input signal.
    lam : float
        Total-variation weight.

    Returns
    -------
    (n,) np.ndarray
        The proximal point.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 2 or lam <= 0.0:
        return y.copy()

    x = np.empty(n, dtype=float)
    k = 0  # start index of the current segment under construction
    k0 = 0  # current sample being processed
    umin = lam
    umax = -lam
    vmin = y[0] - lam
    vmax = y[0] + lam
    kplus = 0  # last index where the upper slack saturated
    kminus = 0  # last index where the lower slack saturated

    while True:
        if k0 == n - 1:
            # Reached the end of the signal: flush the remaining segment.
            if umin < 0.0:
                x[k : kminus + 1] = vmin
                k = kminus = kminus + 1
                k0 = k
                vmin = y[k]
                umin = lam
                umax = y[k] + lam - vmax
            elif umax > 0.0:
                x[k : kplus + 1] = vmax
                k = kplus = kplus + 1
                k0 = k
                vmax = y[k]
                umax = -lam
                umin = y[k] - lam - vmin
            else:
                x[k:n] = vmin + umin / (k0 - k + 1)
                return x
        else:
            if y[k0 + 1] + umin < vmin - lam:
                # negative jump
                x[k : kminus + 1] = vmin
                k = k0 = kplus = kminus = kminus + 1
                vmin = y[k]
                vmax = y[k] + 2 * lam
                umin = lam
                umax = -lam
            elif y[k0 + 1] + umax > vmax + lam:
                # positive jump
                x[k : kplus + 1] = vmax
                k = k0 = kplus = kminus = kplus + 1
                vmax = y[k]
                vmin = y[k] - 2 * lam
                umin = lam
                umax = -lam
            else:
                # no jump yet: advance and update the slacks
                k0 += 1
                umin += y[k0] - vmin
                umax += y[k0] - vmax
                if umin >= lam:
                    vmin += (umin - lam) / (k0 - k + 1)
                    umin = lam
                    kminus = k0
                if umax <= -lam:
                    vmax += (umax + lam) / (k0 - k + 1)
                    umax = -lam
                    kplus = k0


def tv_activation(lam: float):
    """Activation applying Condat's TV prox to each sample (looped on host)."""

    def activation(V):
        Vnp = _to_numpy(V)
        out = np.stack([condat_tv_1d(row, lam) for row in Vnp])
        return _to_like(out, V)

    return activation


# ----------------------------------------------------------------------------
# cvxpy per-sample solves (parametrized, compiled once)
# ----------------------------------------------------------------------------


def polyhedral_activation(A: np.ndarray, b, n: int, m: int, b_is_param: bool):
    r"""
    Activation projecting each sample onto ``{u : A u <= b}`` with cvxpy.

    Builds a single parametrized (DPP) projection problem and solves it per sample.
    The input batch has the point in its first ``n`` columns; when ``b`` is a
    parameter (``b_is_param``), the next ``m`` columns carry the per-sample ``b``.

    Parameters
    ----------
    A : (m, n) np.ndarray
        Inequality matrix.
    b : (m,) np.ndarray or None
        Fixed right-hand side when ``b`` is not a parameter.
    n, m : int
        Point and constraint dimensions.
    b_is_param : bool
        Whether ``b`` varies per sample (stacked into the input batch).
    """
    import cvxpy as cp

    u = cp.Variable(n)
    x_par = cp.Parameter(n)
    if b_is_param:
        b_par = cp.Parameter(m)
        constraints = [A @ u <= b_par]
    else:
        b_par = None
        constraints = [A @ u <= np.asarray(b, dtype=float)]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u - x_par)), constraints)

    def activation(V):
        Vnp = _to_numpy(V)
        out = np.empty((Vnp.shape[0], n), dtype=float)
        for i in range(Vnp.shape[0]):
            x_par.value = Vnp[i, :n]
            if b_is_param:
                b_par.value = Vnp[i, n : n + m]
            prob.solve(solver=cp.CLARABEL)
            out[i] = u.value if u.value is not None else np.nan
        return _to_like(out, V)

    return activation


def qp_activation(P, A, C, n: int, p: int, m: int, is_lp: bool):
    r"""
    Activation solving a QP/LP per sample with cvxpy.

    Builds a single parametrized (DPP) problem ``minimize 0.5 x^T P x + q^T x s.t.
    A x = b, C x <= d`` (the quadratic term is dropped when ``is_lp``) and solves it
    per sample. The input batch stacks the present parametric blocks in the order
    ``[q (n) | b (p) | d (m)]``.

    Parameters
    ----------
    P : (n, n) array or None
        Quadratic term (ignored when ``is_lp``).
    A : (p, n) array or None
        Equality matrix (``None`` if absent).
    C : (m, n) array or None
        Inequality matrix (``None`` if absent).
    n, p, m : int
        Primal / equality / inequality dimensions.
    is_lp : bool
        Drop the quadratic term (linear program).
    """
    import cvxpy as cp
    import scipy.sparse as spa

    x = cp.Variable(n)
    q_par = cp.Parameter(n)
    objective = q_par @ x
    if not is_lp:
        P_dense = P.toarray() if spa.issparse(P) else np.asarray(P, dtype=float)
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P_dense)) + objective

    constraints = []
    b_par = d_par = None
    if A is not None:
        b_par = cp.Parameter(p)
        A_mat = A.toarray() if spa.issparse(A) else np.asarray(A, dtype=float)
        constraints.append(A_mat @ x == b_par)
    if C is not None:
        d_par = cp.Parameter(m)
        C_mat = C.toarray() if spa.issparse(C) else np.asarray(C, dtype=float)
        constraints.append(C_mat @ x <= d_par)
    prob = cp.Problem(cp.Minimize(objective), constraints)

    def activation(V):
        Vnp = _to_numpy(V)
        out = np.empty((Vnp.shape[0], n), dtype=float)
        for i in range(Vnp.shape[0]):
            row = Vnp[i]
            q_par.value = row[:n]
            off = n
            if A is not None:
                b_par.value = row[off : off + p]
                off += p
            if C is not None:
                d_par.value = row[off : off + m]
            prob.solve(solver=cp.CLARABEL)
            out[i] = x.value if x.value is not None else np.nan
        return _to_like(out, V)

    return activation
