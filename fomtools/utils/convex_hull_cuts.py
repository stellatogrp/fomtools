from dataclasses import dataclass

import numpy as np

# Separation of the convex hulls of ReLU, soft-threshold and saturated linear steps
# (Theorems 1, 2 and 3 of the MIP verification paper, corrected version). Each routine takes one output
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
    family: str = ""

    def violation(self, y: np.ndarray, w: float) -> float:
        """Amount by which ``(y, w)`` violates the cut; positive means violated."""
        return float(self.sense * (w - (self.alpha @ y + self.beta)))

    def reflected(self) -> "HullCut":
        """The cut on ``-w``, used to turn an upper-bound routine into a lower one."""
        return HullCut(-self.sense, -self.alpha, -self.beta, self.family + "_lower")

    def shifted(self, k: float) -> "HullCut":
        """The cut on ``w + k``."""
        return HullCut(self.sense, self.alpha, self.beta + k, self.family)


def _endpoints(a, L, U, tol=1e-12):
    """``l0``, ``u0`` (the box corners minimizing and maximizing each ``a_i y_i``) and ``J``.

    ``J`` holds the coordinates with ``a_i != 0`` and a nondegenerate box. Fixed
    coordinates stay out of ``J`` but keep contributing ``a_i u0_i`` to every
    ``ell_I`` (corrected Theorem 1). Coordinates with ``a_i == 0`` get zero endpoints,
    so an infinite bound on a leaf that does not enter the row cannot turn
    ``0 * inf`` into NaN.
    """
    l0 = np.where(a > 0, L, np.where(a < 0, U, 0.0))
    u0 = np.where(a > 0, U, np.where(a < 0, L, 0.0))
    J = np.flatnonzero((a != 0) & (U - L > tol))
    return l0, u0, J


def relu_pair_cut(a, L, U, y_hat, w_hat, t=0.0, offset=0.0, family="relu_pair"):
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
            return HullCut(1, alpha, beta, family)
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
    return HullCut(1, alpha, b - kappa * float(a[I] @ l0[I]), "satlin_subset")


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
        relu_pair_cut(a, L, U, y_hat, wt, t=bt, offset=bt, family="satlin_pair"),
    ]
    lower = [
        cut.reflected() if cut is not None else None
        for cut in (
            satlin_subset_cut(-a, L, U, y_hat, -wt, -ct, -bt),
            relu_pair_cut(-a, L, U, y_hat, -wt, t=-ct, offset=-ct, family="satlin_pair"),
        )
    ]
    best = _most_violated(upper + lower, y_hat, wt, tol)
    return best.shifted(k) if best is not None else None


def soft_subset_cut(a, k, L, U, y_hat, w_hat, lambd):
    """Most violated subset inequality of the upper family for ``w = T_lambd(a^T y + k)``.

    The family is ``w <= rho_I (theta + lambd) + (1 - rho_I) sum_{i in I} a_i (y_i - l0_i)``
    with ``theta = a^T y + k`` and ``rho_I = (ell_I - lambd) / (ell_I + lambd)``, over ``I``
    with ``ell_I >= lambd`` (line 3 of Q in the corrected Theorem 2). The constant ``k`` is
    a fixed coordinate: it contributes to ``ell_I`` but is not in ``J``. Following the
    corrected proof of Prop. 2, ``I* = {i in J : 2 lambd x_i < theta_hat + lambd - w_hat}``
    minimizes the cleared slack; if it is inadmissible, no inequality of the family is
    violated once the pair family holds.
    """
    l0, u0, J = _endpoints(a, L, U)
    ell_empty = float(a @ u0) + k
    if ell_empty < lambd:
        return None
    theta_hat = float(a @ y_hat) + k
    x = (y_hat[J] - l0[J]) / (u0[J] - l0[J])
    I = J[2 * lambd * x < theta_hat + lambd - w_hat]
    ell_I = ell_empty - float(a[I] @ (u0[I] - l0[I]))
    if ell_I < lambd:
        return None
    rho = (ell_I - lambd) / (ell_I + lambd)
    alpha = rho * np.asarray(a, dtype=float)
    alpha[I] += (1 - rho) * a[I]
    beta = rho * (k + lambd) - (1 - rho) * float(a[I] @ l0[I])
    return HullCut(1, alpha, beta, "soft_subset")


def _soft_threshold_upper(a, k, L, U, y_hat, w_hat, lambd):
    """Candidate upper bounds on ``w = T_lambd(a^T y + k)`` (lines 1, 2, 3 and 5 of Q)."""
    l0, u0, _ = _endpoints(a, L, U)
    a = np.asarray(a, dtype=float)
    cuts = [
        HullCut(1, a.copy(), k + lambd, "soft_line1"),
        soft_subset_cut(a, k, L, U, y_hat, w_hat, lambd),
        relu_pair_cut(a, L, U, y_hat, w_hat, t=lambd - k, family="soft_pair"),
    ]
    if float(a @ u0) + k <= lambd:
        cuts.append(HullCut(1, np.zeros_like(a), 0.0, "soft_line2"))
    return cuts


def separate_soft_threshold(a, k, L, U, y_hat, w_hat, lambd, tol=1e-6):
    """Most violated hull cut for ``w = T_lambd(a^T y + k)``, or ``None``.

    Checks every line of Q in the corrected Theorem 2. The lower bounds (lines 1, 2,
    4 and 6) follow by the reflection ``(a, k, w) -> (-a, -k, -w)``, since
    ``-T_lambd(theta) = T_lambd(-theta)``.
    """
    if lambd <= 0:
        return None
    upper = _soft_threshold_upper(a, k, L, U, y_hat, w_hat, lambd)
    lower = [
        cut.reflected() if cut is not None else None
        for cut in _soft_threshold_upper(-np.asarray(a), -k, L, U, y_hat, -w_hat, lambd)
    ]
    return _most_violated(upper + lower, y_hat, w_hat, tol)
