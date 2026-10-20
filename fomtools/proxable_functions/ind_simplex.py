import cvxpy as cp
import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndSimplex(ProxableFunction):
    r"""
    Indicator of the scaled simplex $\{x : x \ge 0,\ \mathbf{1}^\top x = a\}$.

    Represents

    $$
        f(x) = I_{\{x \ge 0,\ \mathbf{1}^\top x = a\}}(x) = \begin{cases}
            0       & \text{if } x \ge 0 \text{ and } \mathbf{1}^\top x = a \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    For $a = 1$ this is the probability simplex.

    Parameters
    ----------
    a : float or LinearExpression, default 1.0
        Simplex sum $a > 0$, or a dim-1 ``LinearExpression`` treated as a
        parameter by the verification lowering.

    Raises
    ------
    ValueError
        If ``a`` is not positive, or a ``LinearExpression`` ``a`` is not dim-1.

    See Also
    --------
    IndPolyhedral : General polyhedron $\{Ax \le b\}$.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the simplex,

    $$
        \operatorname{prox}_{\gamma f}(v)
        = \arg\min_{u}\ \tfrac{1}{2}\|u - v\|_2^2
        \quad\text{s.t.}\quad u \ge 0,\ \mathbf{1}^\top u = a,
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indsimplex`` encodes the KKT system of
    this projection with complementarity constraints. The simplex is bounded,
    so the multiplier bounds are derived from the input box and no ``dual_ub``
    is needed.
    """

    def __init__(self, a: float | LinearExpression = 1.0):
        super().__init__()
        if hasattr(a, "decomposition_dict"):  # dim-1 LinearExpression parameter
            if a.ndim != 1:
                raise ValueError("a must be a scalar or a dim-1 LinearExpression")
        elif a <= 0:
            raise ValueError("a must be positive")
        self.a = a

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the simplex $\{x : x \ge 0,\ \mathbf{1}^\top x = a\}$.

        Solved numerically with CVXPY (Clarabel) as an independent reference; a
        ``LinearExpression`` ``a`` uses its constant value.

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
        a = float(self.a.const[0]) if hasattr(self.a, "decomposition_dict") else float(self.a)
        u = cp.Variable(arg.shape[0])
        # Pin CLARABEL: deterministic and accurate (~1e-7) across platforms, unlike
        # cvxpy's auto-selected default (OSQP/SCS) which can return ~1e-3 solutions.
        cp.Problem(cp.Minimize(cp.sum_squares(u - arg)), [u >= 0, cp.sum(u) == a]).solve(
            solver=cp.CLARABEL
        )
        return np.asarray(u.value, dtype=float)
