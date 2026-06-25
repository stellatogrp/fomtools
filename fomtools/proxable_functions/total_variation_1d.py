import warnings

import cvxpy as cp
import numpy as np

from fomtools.proxable_functions import ProxableFunction

_SHORT_SIGNAL_MSG = (
    "TotalVariation1D received a length-{n} signal (< 2): total variation is identically "
    "zero, so the prox is the identity (returns the input unchanged)."
)


def diff_matrix(n: int) -> np.ndarray:
    r"""
    First-difference operator :math:`D \in \mathbb{R}^{(n-1) \times n}`,
    with :math:`(Dx)_i = x_{i+1} - x_i`.
    """
    D = np.zeros((n - 1, n))
    rows = np.arange(n - 1)
    D[rows, rows] = -1.0
    D[rows, rows + 1] = 1.0
    return D


class TotalVariation1D(ProxableFunction):
    r"""
    Proximal operator of the 1-D total-variation penalty
    :math:`f(x) = \lambda \|Dx\|_1 = \lambda \sum_i |x_{i+1} - x_i|`.

    The prox is the 1-D fused-lasso signal approximator

    $$
        \operatorname{prox}_{\gamma f}(v)
        = \arg\min_x\ \tfrac12\|x - v\|_2^2 + \gamma\lambda\|Dx\|_1,
    $$

    lowered for verification (``prox_totalvariation1d``) via the KKT system
    ``x = v - Dᵀu``, ``Dx = z⁺ - z⁻``, ``0 ≤ (γλ ∓ u) ⊥ z^{±} ≥ 0``.

    Parameters
    ----------
    lambd : float or LinearExpression
        Non-negative scalar weight, or a dim-1 ``LinearExpression`` parameter.
    """

    def __init__(self, lambd=1.0):
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
        r"""Evaluate :math:`f(x) = \lambda \sum_i |x_{i+1} - x_i|`."""
        x = np.asarray(x, dtype=float)
        return self._lambd_value() * float(np.sum(np.abs(np.diff(x))))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute :math:`\operatorname{prox}_{\gamma \lambda \|D\cdot\|_1}(\text{arg})`.

        Parameters
        ----------
        arg : np.ndarray
            Input signal.
        gamma : float, default 1.0
            Step size.

        Returns
        -------
        np.ndarray
            The proximal point.
        """
        arg = np.asarray(arg, dtype=float)
        n = arg.shape[0]
        if n < 2:
            warnings.warn(_SHORT_SIGNAL_MSG.format(n=n), UserWarning, stacklevel=2)
            return arg.copy()  # TV of a scalar is 0 -> identity
        x = cp.Variable(n)
        tau = gamma * self._lambd_value()
        obj = 0.5 * cp.sum_squares(x - arg) + tau * cp.norm1(diff_matrix(n) @ x)
        cp.Problem(cp.Minimize(obj)).solve()
        return np.asarray(x.value, dtype=float)
