import copy
from dataclasses import dataclass

import numpy as np

# Separation of the convex hulls of ReLU and saturated linear steps (Theorems 1 and 3
# of the MIP verification paper, corrected version). Each routine takes one output
# component ``w = phi(a^T y + k)`` with ``y`` in the box ``[L, U]`` and a relaxed
# point ``(y_hat, w_hat)``, and returns the most violated hull inequality, if any.


@dataclass
class HullCut:
    """The inequality ``sense * w <= sense * (alpha @ y + beta)``.

    ``sense = 1`` is an upper bound ``w <= alpha @ y + beta`` and ``sense = -1`` a
    lower bound ``w >= alpha @ y + beta``. The violation and the constraint are both
    built from ``(alpha, beta)``, so they cannot disagree.
    """

    sense: int
    alpha: np.ndarray
    beta: float

    def violation(self, y: np.ndarray, w: float) -> float:
        """Amount by which ``(y, w)`` violates the cut; positive means violated."""
        return float(self.sense * (w - (self.alpha @ y + self.beta)))

    def reflected(self) -> "HullCut":
        """The cut on ``-w``, used to turn an upper-bound routine into a lower one."""
        return HullCut(-self.sense, -self.alpha, -self.beta)

    def shifted(self, k: float) -> "HullCut":
        """The cut on ``w + k``."""
        return HullCut(self.sense, self.alpha, self.beta + k)


def _endpoints(a, L, U, tol=1e-12):
    """``l0``, ``u0`` (the box corners minimizing and maximizing each ``a_i y_i``) and ``J``.

    ``J`` holds the coordinates with ``a_i != 0`` and a nondegenerate box. Fixed
    coordinates stay out of ``J`` but keep contributing ``a_i u0_i`` to every
    ``ell_I`` (corrected Theorem 1).
    """
    l0 = np.where(a >= 0, L, U)
    u0 = np.where(a >= 0, U, L)
    J = np.flatnonzero((a != 0) & (U - L > tol))
    return l0, u0, J


def relu_pair_cut(a, L, U, y_hat, w_hat, t=0.0, offset=0.0):
    """Most violated pair inequality for ``w <= offset + ReLU(a^T y - t)``.

    The family is ``w <= offset + sum_{i in I} a_i (y_i - l0_i)
    + (ell_I - t) / (u0_o - l0_o) (y_o - l0_o)`` over the pairs ``(I, o)`` with
    ``ell_I >= t > ell_{I + o}``. Coordinates are added in nondecreasing order of
    ``(y_hat_i - l0_i) / (u0_i - l0_i)`` until ``ell_I - t`` changes sign, which gives
    the most violated pair (Tjandraatmadja et al., Prop. 2). Returns ``None`` when
    the family is empty, i.e. ``ell_empty < t`` or ``ell_J >= t``.
    """
    l0, u0, J = _endpoints(a, L, U)
    ell = float(a @ u0)
    if ell < t:
        return None
    frac = (y_hat[J] - l0[J]) / (u0[J] - l0[J])
    I = []
    for o in J[np.argsort(frac, kind="stable")]:
        ell_next = ell - a[o] * (u0[o] - l0[o])
        if ell_next < t:
            slope = (ell - t) / (u0[o] - l0[o])
            alpha = np.zeros_like(a, dtype=float)
            alpha[I] = a[I]
            alpha[o] += slope
            beta = offset - float(a[I] @ l0[I]) - slope * l0[o]
            return HullCut(1, alpha, beta)
        I.append(o)
        ell = ell_next
    return None


def satlin_subset_cut(a, L, U, y_hat, w_hat, b, c):
    """Most violated subset inequality of the upper family for ``w = S_[b, c](a^T y)``.

    The family is ``w <= b + (c - b) / (c - ell_I) sum_{i in I} a_i (y_i - l0_i)`` over
    ``I`` with ``ell_I <= b`` (line 3 of Q in the corrected Theorem 3). The candidate
    ``I* = {i in J : (c - b) x_i < w_hat - b}`` minimizes the cleared slack; if it is
    inadmissible, no inequality of the family is violated once the pair family holds.
    """
    l0, u0, J = _endpoints(a, L, U)
    if float(a @ l0) > b:
        return None
    x = (y_hat[J] - l0[J]) / (u0[J] - l0[J])
    I = J[(c - b) * x < w_hat - b]
    ell_I = float(a @ u0) - float(a[I] @ (u0[I] - l0[I]))
    if ell_I > b:
        return None
    kappa = (c - b) / (c - ell_I)
    alpha = np.zeros_like(a, dtype=float)
    alpha[I] = kappa * a[I]
    return HullCut(1, alpha, b - kappa * float(a[I] @ l0[I]))


def _most_violated(cuts, y_hat, w_hat, tol):
    best, best_viol = None, tol
    for cut in cuts:
        if cut is None:
            continue
        viol = cut.violation(y_hat, w_hat)
        if viol > best_viol:
            best, best_viol = cut, viol
    return best


def separate_relu(a, k, L, U, y_hat, w_hat, tol=1e-6):
    """Most violated hull cut for ``w = ReLU(a^T y + k)``, or ``None``.

    The constant is folded into the threshold, ``ReLU(a^T y + k) = ReLU(a^T y - (-k))``.
    The lower bounds ``w >= 0`` and ``w >= a^T y + k`` are already in the formulation.
    """
    return _most_violated([relu_pair_cut(a, L, U, y_hat, w_hat, t=-k)], y_hat, w_hat, tol)


def separate_satlin(a, k, L, U, y_hat, w_hat, b, c, tol=1e-6):
    """Most violated hull cut for ``w = S_[b, c](a^T y + k)``, or ``None``.

    With ``w~ = w - k``, ``b~ = b - k`` and ``c~ = c - k`` this is ``w~ = S_[b~, c~](a^T y)``.
    The upper families are the subset family and the pair family of ``b~ + ReLU(a^T y - b~)``.
    The lower families follow by the reflection ``(a, w, b, c) -> (-a, -w, -c, -b)``.
    Lines 1 and 2 of Q (``b <= w <= c`` and the one-sided ``w <= a^T y``, ``w >= a^T y``)
    are in the formulation already.
    """
    bt, ct, wt = b - k, c - k, w_hat - k
    upper = [
        satlin_subset_cut(a, L, U, y_hat, wt, bt, ct),
        relu_pair_cut(a, L, U, y_hat, wt, t=bt, offset=bt),
    ]
    lower = [
        cut.reflected() if cut is not None else None
        for cut in (
            satlin_subset_cut(-a, L, U, y_hat, -wt, -ct, -bt),
            relu_pair_cut(-a, L, U, y_hat, -wt, t=-ct, offset=-ct),
        )
    ]
    best = _most_violated(upper + lower, y_hat, wt, tol)
    return best.shifted(k) if best is not None else None


def legacy_cut_to_hullcut(a, Iint, lI, h, Lhat, Uhat):
    """The inequality ``create_new_constr`` builds from ``get_cuts`` output, as a ``HullCut``.

    Kept for the soft-threshold path, which still uses the original separation.
    """
    alpha = np.zeros_like(a, dtype=float)
    alpha[Iint] = a[Iint]
    slope = lI / (Uhat[h] - Lhat[h])
    alpha[h] += slope
    beta = -float(a[Iint] @ Lhat[Iint]) - slope * Lhat[h]
    return HullCut(1, alpha, beta)


def add_conv_cuts(wi, ai, y, lambd, y_l, y_u):
    """
    Generate convex hull cuts for MIP bound tightening.

    Implements the Huchette-Vielma algorithm to find violated convex hull
    inequalities for piecewise linear functions. Returns cut data if a
    violated inequality is found.

    Parameters
    ----------
    wi : np.ndarray
        Current objective coefficient values.
    ai : np.ndarray
        Constraint coefficients.
    y : np.ndarray
        Current variable values.
    lambd : float
        Threshold parameter for the cut.
    y_l : np.ndarray
        Lower bounds on variables.
    y_u : np.ndarray
        Upper bounds on variables.

    Returns
    -------
    tuple
        A tuple (Iint, lI, h, L_hat, U_hat) where:
        - Iint : np.ndarray or None - Indices for the cut
        - lI : float or None - Cut coefficient
        - h : int or None - Pivot index
        - L_hat : np.ndarray or None - Adjusted lower bounds
        - U_hat : np.ndarray or None - Adjusted upper bounds

        Returns (None, None, None, None, None) if no violated cut is found.
    """
    L_hat = np.zeros(y_l.shape)
    U_hat = np.zeros(y_u.shape)

    for j in range(y.shape[0]):
        if ai[j] >= 0:
            L_hat[j] = y_l[j]
            U_hat[j] = y_u[j]
        else:
            L_hat[j] = y_u[j]
            U_hat[j] = y_l[j]

    # Leaves with a zero coefficient do not enter the row; zero their bounds so an
    # infinite one does not turn 0 * inf into NaN. An infinite bound on a leaf that
    # does enter the row has no finite cut.
    L_hat[ai == 0] = 0.0
    U_hat[ai == 0] = 0.0
    if not (np.all(np.isfinite(L_hat)) and np.all(np.isfinite(U_hat))):
        return None, None, None, None, None

    Iint, rhs, lI, h = compute_v(ai, y, lambd, L_hat, U_hat)

    if Iint is not None and wi > rhs + 1e-6:
        return Iint, lI, h, L_hat, U_hat

    return None, None, None, None, None


def compute_v(wi, xi, lambd, Lhat, Uhat):
    """
    Compute a violated inequality using the Huchette-Vielma algorithm.

    Searches for a violated convex hull inequality by iteratively building
    an index set that separates the current point from the convex hull.

    Parameters
    ----------
    wi : np.ndarray
        Objective coefficient values.
    xi : np.ndarray
        Current variable values.
    lambd : float
        Threshold parameter.
    Lhat : np.ndarray
        Adjusted lower bounds (accounting for coefficient signs).
    Uhat : np.ndarray
        Adjusted upper bounds (accounting for coefficient signs).

    Returns
    -------
    tuple
        A tuple (Iint, rhs, lI, h) where:
        - Iint : np.ndarray or None - Index set for the cut
        - rhs : float or None - Right-hand side value
        - lI : float or None - Cut function value
        - h : int or None - Pivot index

        Returns (None, None, None, None) if no violated inequality exists.
    """
    idx = np.arange(len(wi))

    filtered_idx = np.array([j for j in idx if wi[j] != 0 and np.abs(Uhat[j] - Lhat[j]) > 1e-7])

    def key_func(j):
        return (xi[j] - Lhat[j]) / (Uhat[j] - Lhat[j])

    keys = np.array([key_func(j) for j in filtered_idx])
    sorted_idx = np.argsort(
        keys
    )  # this is nondecreasing, should be the corrct one according to paper
    # sorted_idx = np.argsort(keys)[::-1]
    filtered_idx = filtered_idx[sorted_idx]

    I = np.array([])
    Icomp = set(range(len(wi)))

    lI = compute_I(wi, lambd, Lhat, Uhat, I, np.array(list(Icomp)))
    if lI < 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = np.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        lI_new = compute_I(
            wi, lambd, Lhat, Uhat, Itest.astype(np.int32), np.array(list(Icomp_test))
        )
        if lI_new < 0:
            Iint = I.astype(np.int32)
            rhs = np.sum(np.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (
                xi[int(h)] - Lhat[int(h)]
            )
            return Iint, rhs, lI, int(h)

        I = Itest
        Icomp = Icomp_test
        lI = lI_new
    else:
        return None, None, None, None


def compute_I(w, lambd, Lhat, Uhat, I, Icomp):
    """
    Evaluate the cut function value for a given index partition.

    Computes the value of the piecewise linear function at the boundary
    defined by the index set I (using lower bounds) and its complement
    Icomp (using upper bounds), minus the threshold lambda.

    Parameters
    ----------
    w : np.ndarray
        Coefficient vector.
    lambd : float
        Threshold parameter to subtract.
    Lhat : np.ndarray
        Adjusted lower bounds.
    Uhat : np.ndarray
        Adjusted upper bounds.
    I : np.ndarray
        Indices using lower bounds.
    Icomp : np.ndarray
        Indices using upper bounds (complement of I).

    Returns
    -------
    float
        The cut function value: sum(w[I] * Lhat[I]) + sum(w[Icomp] * Uhat[Icomp]) - lambd.
    """
    if I.shape[0] == 0:
        return np.sum(np.multiply(w, Uhat)) - lambd
    if Icomp.shape[0] == 0:
        return np.sum(np.multiply(w, Lhat)) - lambd

    return np.sum(np.multiply(w[I], Lhat[I])) + np.sum(np.multiply(w[Icomp], Uhat[Icomp])) - lambd


def create_new_constr(w, a, y, Iint, lI, h, Lhat, Uhat):
    """
    Build a constraint from the cut data.

    Constructs the actual constraint expression from the cut information
    returned by ``compute_v``.

    Parameters
    ----------
    w : variable
        The variable to constrain (left-hand side).
    a : np.ndarray
        Coefficient vector.
    y : np.ndarray
        Variable vector for the right-hand side.
    Iint : np.ndarray
        Index set for the cut.
    lI : float
        Cut function value at the pivot.
    h : int
        Pivot index.
    Lhat : np.ndarray
        Adjusted lower bounds.
    Uhat : np.ndarray
        Adjusted upper bounds.

    Returns
    -------
    Constraint
        The constraint w <= (cut expression).
    """
    new_constr = 0
    for idx in Iint:
        new_constr += a[idx] * (y[idx] - Lhat[idx])
    # this lI func already factors in lambd so we dont subtract it again from the top
    new_constr += lI / (Uhat[h] - Lhat[h]) * (y[h] - Lhat[h])

    return w <= new_constr


def get_cuts(step, i, _w, _A, _y, l, u):
    """
    Dispatch cut generation based on step type.

    Determines which cut generation strategy to use based on the step
    type (ReLU or soft threshold) and the index configuration.

    Parameters
    ----------
    step : Step
        The optimization step (ReLU or SoftThreshold).
    i : int
        The component index to generate cuts for.
    _w : np.ndarray
        Objective coefficient values.
    _A : np.ndarray
        Constraint coefficient matrix row.
    _y : np.ndarray
        Current variable values.
    l : np.ndarray
        Lower bounds.
    u : np.ndarray
        Upper bounds.

    Returns
    -------
    tuple
        Cut data (Iint, lI, h, L_hat, U_hat) or (None, None, None, None, None)
        if no cut applies.
    """
    try:
        if i in step.proj_indices and i in step.idx_with_binary_vars:
            return add_conv_cuts(_w, _A, _y, 0, l, u)
    except AttributeError:
        _r, _l = i in step.idx_with_right_binary_vars, i in step.idx_with_left_binary_vars
        if _r and (not _l or _w > 0):
            return add_conv_cuts(_w, _A, _y, step.lambd, l, u)
        if _l and (not _r or _w < 0):
            Iint, uI, h, l_Lhat, l_Uhat = add_conv_cuts(_w, _A, _y, -step.lambd, u, l)
            return Iint, -uI if uI is not None else uI, h, l_Lhat, l_Uhat

        return None, None, None, None, None
    return None, None, None, None, None
