import cvxpy as cp
import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndBallL1(ProxableFunction):
    r"""
    Indicator of the $\ell_1$ ball $\{x : \|x\|_1 \le r\}$.

    Represents

    $$
        f(x) = I_{\{\|x\|_1 \le r\}}(x) = \begin{cases}
            0       & \text{if } \|x\|_1 \le r \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    r : float or LinearExpression
        Ball radius $r \ge 0$, or a dim-1 ``LinearExpression`` treated as a
        parameter by the verification lowering.
    dual_ub : float or None, default None
        Upper bound on the KKT multipliers used by the verification lowering.
        ``None`` auto-derives a valid bound from the input box.

    Raises
    ------
    ValueError
        If ``r`` is negative, or a ``LinearExpression`` ``r`` is not dim-1.

    See Also
    --------
    IndBallLinf : The $\ell_\infty$ ball.
    NormInf : Its prox is computed from this projection by Moreau decomposition.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the ball,

    $$
        \operatorname{prox}_{\gamma f}(v)
        = \arg\min_{u}\ \tfrac{1}{2}\|u - v\|_2^2
        \quad\text{s.t.}\quad \|u\|_1 \le r,
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indballl1`` encodes the KKT system of the
    lifted problem

    $$
        \min_{u, t}\ \tfrac{1}{2}\|u - v\|_2^2 \quad\text{s.t.}\quad
        -t \le u \le t,\ \mathbf{1}^\top t \le r
    $$

    with complementarity constraints, bounding the multipliers by ``dual_ub``.
    """

    def __init__(
        self,
        r: float | LinearExpression,
        dual_ub: float | None = None,
    ):
        super().__init__()
        if hasattr(r, "decomposition_dict"):  # dim-1 LinearExpression parameter
            if r.ndim != 1:
                raise ValueError("r must be a scalar or a dim-1 LinearExpression")
        elif r < 0:
            raise ValueError("r must be non-negative")
        self.r = r
        self.dual_ub = dual_ub

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the $\ell_1$ ball of radius $r$.

        Solved numerically with CVXPY (Clarabel) as an independent reference; a
        ``LinearExpression`` ``r`` uses its constant value.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size; unused, since the projection is independent of $\gamma$.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.
        """
        arg = np.asarray(arg, dtype=float)
        # Numeric reference: a parameter ``r`` uses its stored constant value (cf. IndAffine).
        r = float(self.r.const[0]) if hasattr(self.r, "decomposition_dict") else float(self.r)
        u = cp.Variable(arg.shape[0])
        # Pin CLARABEL: deterministic and accurate (~1e-7) across platforms, unlike
        # cvxpy's auto-selected default (OSQP/SCS) which can return ~1e-3 solutions.
        cp.Problem(cp.Minimize(cp.sum_squares(u - arg)), [cp.norm1(u) <= r]).solve(
            solver=cp.CLARABEL
        )
        return np.asarray(u.value, dtype=float)
