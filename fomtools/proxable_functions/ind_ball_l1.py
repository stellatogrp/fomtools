import cvxpy as cp
import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndBallL1(ProxableFunction):
    r"""
    Indicator of the $\ell_1$-ball $\{u : \|u\|_1 \le r\}$.

    The proximal operator is the Euclidean projection onto the ball, independent of
    $\gamma$. The verification-side lowering (``prox_indballl1`` in the step
    decomposer) uses the lifted KKT system of

    $$
        \min_{u, t}\ \tfrac12\|u - x\|_2^2 \quad\text{s.t.}\quad
        -t \le u \le t,\ \mathbf{1}^\top t \le r,
    $$

    encoded with the complementarity primitive.

    Parameters
    ----------
    r : float or LinearExpression
        Ball radius. A scalar (``r >= 0``) or a dim-1 ``LinearExpression`` parameter
        (the KKT stays linear in the variables when ``r`` varies).
    dual_ub : None, float, or np.ndarray, optional
        Override for the upper bound on the KKT duals. When ``None`` (default) a valid
        finite bound is auto-derived from the input box, so no user input is needed.
    """

    def __init__(self, r, dual_ub=None):
        super().__init__()
        if hasattr(r, "decomposition_dict"):  # dim-1 LinearExpression parameter
            if r.ndim != 1:
                raise ValueError("r must be a scalar or a dim-1 LinearExpression")
        elif r < 0:
            raise ValueError("r must be non-negative")
        self.r = r
        self.dual_ub = dual_ub

    def prox(self, arg, gamma=1.0):
        """
        Project ``arg`` onto ``{u : ||u||_1 <= r}`` (independent of ``gamma``).

        Parameters
        ----------
        arg : np.ndarray
            Point to project.
        gamma : float, default 1.0
            Step size (irrelevant for an indicator; accepted for interface consistency).

        Returns
        -------
        np.ndarray
            The Euclidean projection of ``arg`` onto the ball.
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
