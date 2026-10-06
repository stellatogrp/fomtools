import warnings

import cvxpy as cp
import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction

_SHORT_SIGNAL_MSG = (
    "TotalVariation1D received a length-{n} signal (< 2): total variation is identically "
    "zero, so the prox is the identity (returns the input unchanged)."
)


def diff_matrix(n: int) -> np.ndarray:
    r"""
    Build the first-difference operator $D$, with $(Dx)_i = x_{i+1} - x_i$.

    Parameters
    ----------
    n : int
        Signal length.

    Returns
    -------
    np.ndarray
        The matrix $D \in \mathbb{R}^{(n-1) \times n}$.
    """
    D = np.zeros((n - 1, n))
    rows = np.arange(n - 1)
    D[rows, rows] = -1.0
    D[rows, rows + 1] = 1.0
    return D


class TotalVariation1D(ProxableFunction):
    r"""
    1-D total variation.

    Represents

    $$
        f(x) = \lambda \|Dx\|_1 = \lambda \sum_i |x_{i+1} - x_i|,
    $$

    where $D$ is the first-difference operator (see ``diff_matrix``). Convex
    and piecewise linear.

    Parameters
    ----------
    lambd : float or LinearExpression, default 1.0
        Weight $\lambda \ge 0$, or a dim-1 ``LinearExpression`` treated as a
        parameter by the verification lowering.

    Raises
    ------
    ValueError
        If ``lambd`` is negative, or a ``LinearExpression`` ``lambd`` is not
        dim-1.

    See Also
    --------
    NormL1 : The case $D = I$.

    Notes
    -----
    **Proximal operator.** The 1-D fused-lasso signal approximator,

    $$
        \operatorname{prox}_{\gamma f}(v)
        = \arg\min_x\ \tfrac{1}{2}\|x - v\|_2^2 + \gamma\lambda\|Dx\|_1.
    $$

    For a signal of length $n < 2$, $f \equiv 0$ and the prox is the identity.

    **Verification lowering.** ``prox_totalvariation1d`` encodes the KKT system

    $$
        x = v - D^\top u,\quad Dx = z^+ - z^-,\quad
        0 \le (\gamma\lambda - u) \perp z^+ \ge 0,\quad
        0 \le (\gamma\lambda + u) \perp z^- \ge 0
    $$

    with complementarity constraints. All variable bounds are derived from the
    data, so no ``dual_ub`` is needed.
    """

    def __init__(self, lambd: float | LinearExpression = 1.0):
        super().__init__()
        if hasattr(lambd, "decomposition_dict"):  # dim-1 LinearExpression parameter
            if lambd.ndim != 1:
                raise ValueError("lambd must be a scalar or a dim-1 LinearExpression")
        elif lambd < 0:
            raise ValueError("lambd must be non-negative")
        self.lambd = lambd

    def _lambd_value(self) -> float:
        if hasattr(self.lambd, "decomposition_dict"):
            return float(self.lambd.const[0])
        return float(self.lambd)

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \lambda \sum_i |x_{i+1} - x_i|$.

        A ``LinearExpression`` ``lambd`` uses its constant value.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        x = np.asarray(x, dtype=float)
        return self._lambd_value() * float(np.sum(np.abs(np.diff(x))))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Solved numerically with CVXPY (Clarabel) as an independent reference; a
        ``LinearExpression`` ``lambd`` uses its constant value.

        Parameters
        ----------
        arg : np.ndarray
            Input signal.
        gamma : float, default 1.0
            Step size $\gamma > 0$.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Warns
        -----
        UserWarning
            If ``arg`` has fewer than 2 entries; ``arg`` is then returned
            unchanged.
        """
        arg = np.asarray(arg, dtype=float)
        n = arg.shape[0]
        if n < 2:
            warnings.warn(_SHORT_SIGNAL_MSG.format(n=n), UserWarning, stacklevel=2)
            return arg.copy()  # TV of a scalar is 0 -> identity
        x = cp.Variable(n)
        tau = gamma * self._lambd_value()
        obj = 0.5 * cp.sum_squares(x - arg) + tau * cp.norm1(diff_matrix(n) @ x)
        # Pin CLARABEL: deterministic and accurate (~1e-7) across platforms, unlike
        # cvxpy's auto-selected default (OSQP/SCS) which can return ~1e-3 solutions.
        cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
        return np.asarray(x.value, dtype=float)
