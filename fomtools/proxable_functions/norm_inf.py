import cvxpy as cp
import numpy as np

from fomtools.proxable_functions import ProxableFunction


class NormInf(ProxableFunction):
    r"""
    Proximal operator of the scaled :math:`\ell_\infty` norm
    :math:`f(x) = \lambda \|x\|_\infty`.

    By Moreau decomposition the prox is dual to projection onto the
    :math:`\ell_1`-ball:

    $$
        \operatorname{prox}_{\gamma\lambda\|\cdot\|_\infty}(v)
        = v - \operatorname{proj}_{\{\|\cdot\|_1 \le \gamma\lambda\}}(v),
    $$

    which is how the verification-side lowering (``prox_norminf``) is implemented —
    reusing the ℓ1-ball projection decomposer.

    Parameters
    ----------
    lambd : float or LinearExpression
        Non-negative scalar weight (the :math:`\ell_\infty` norm is not separable, so a
        single scalar), or a dim-1 ``LinearExpression`` parameter.
    dual_ub : None, float, or np.ndarray, optional
        Upper bound forwarded to the underlying ℓ1-ball projection's dual (the prox is
        ``v - proj_{||.||_1 <= gamma*lambda}(v)`` by Moreau). ``None`` (default) lets
        ``IndBallL1`` auto-derive a valid bound from the input box; supply a value to
        override it when the auto bound is loose (e.g. large asymmetric inputs).
    """

    def __init__(self, lambd=1.0, dual_ub=None):
        super().__init__()
        if hasattr(lambd, "decomposition_dict"):  # dim-1 LinearExpression parameter
            if lambd.ndim != 1:
                raise ValueError("lambd must be a scalar or a dim-1 LinearExpression")
        elif lambd < 0:
            raise ValueError("lambd must be non-negative")
        self.lambd = lambd
        self.dual_ub = dual_ub

    def _lambd_value(self) -> float:
        """Concrete scalar weight (a parameter uses its stored constant value)."""
        if hasattr(self.lambd, "decomposition_dict"):
            return float(self.lambd.const[0])
        return float(self.lambd)

    def eval(self, x: np.ndarray) -> float:
        r"""Evaluate :math:`f(x) = \lambda \|x\|_\infty`."""
        return self._lambd_value() * float(np.max(np.abs(np.asarray(x, dtype=float))))

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute :math:`\operatorname{prox}_{\gamma \lambda \|\cdot\|_\infty}(\text{rhs})`.

        Solved directly from the defining problem (an independent reference, not via the
        Moreau identity used for the verification lowering).

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size.

        Returns
        -------
        np.ndarray
            The proximal point.
        """
        rhs = np.asarray(rhs, dtype=float)
        u = cp.Variable(rhs.shape[0])
        obj = 0.5 * cp.sum_squares(u - rhs) + gamma * self._lambd_value() * cp.norm_inf(u)
        cp.Problem(cp.Minimize(obj)).solve()
        return np.asarray(u.value, dtype=float)
