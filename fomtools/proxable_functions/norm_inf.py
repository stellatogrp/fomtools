import cvxpy as cp
import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class NormInf(ProxableFunction):
    r"""
    Scaled $\ell_\infty$ norm.

    Represents

    $$
        f(x) = \lambda \|x\|_\infty = \lambda \max_i |x_i|.
    $$

    Convex and piecewise linear.

    Parameters
    ----------
    lambd : float or LinearExpression, default 1.0
        Weight $\lambda \ge 0$ (a single scalar, since the $\ell_\infty$ norm is
        not separable), or a dim-1 ``LinearExpression`` treated as a parameter
        by the verification lowering.
    dual_ub : float or None, default None
        Upper bound forwarded to the underlying ``IndBallL1`` projection.
        ``None`` auto-derives a valid bound from the input box; supply a value
        when that bound is loose (e.g. large asymmetric inputs).

    Raises
    ------
    ValueError
        If ``lambd`` is negative, or a ``LinearExpression`` ``lambd`` is not
        dim-1.

    See Also
    --------
    IndBallL1 : Projection used by the Moreau decomposition below.
    NormL1 : The $\ell_1$ norm.

    Notes
    -----
    **Proximal operator.** By Moreau decomposition, the prox is the residual of
    the projection onto the dual-norm ball,

    $$
        \operatorname{prox}_{\gamma f}(v)
        = v - \operatorname{proj}_{\{\|u\|_1 \le \gamma\lambda\}}(v).
    $$

    **Verification lowering.** ``prox_norminf`` applies this identity, reusing
    ``prox_indballl1`` with radius $\gamma\lambda$ and the given ``dual_ub``.
    """

    def __init__(
        self,
        lambd: float | LinearExpression = 1.0,
        dual_ub: float | None = None,
    ):
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
        r"""
        Evaluate $f(x) = \lambda \|x\|_\infty$.

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
        return self._lambd_value() * float(np.max(np.abs(np.asarray(x, dtype=float))))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Solved numerically with CVXPY (Clarabel) from the defining problem, as
        a reference independent of the Moreau identity used by the verification
        lowering; a ``LinearExpression`` ``lambd`` uses its constant value.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size $\gamma > 0$.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.
        """
        arg = np.asarray(arg, dtype=float)
        u = cp.Variable(arg.shape[0])
        obj = 0.5 * cp.sum_squares(u - arg) + gamma * self._lambd_value() * cp.norm_inf(u)
        # Pin CLARABEL: deterministic and accurate (~1e-7) across platforms, unlike
        # cvxpy's auto-selected default (OSQP/SCS) which can return ~1e-3 solutions.
        cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
        return np.asarray(u.value, dtype=float)
