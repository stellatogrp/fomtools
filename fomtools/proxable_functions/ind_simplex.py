import cvxpy as cp
import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndSimplex(ProxableFunction):
    r"""
    Indicator of the (scaled probability) simplex
    :math:`\{u : u \ge 0,\ \mathbf{1}^\top u = a\}`.

    The proximal operator is the Euclidean projection onto the simplex, independent of
    :math:`\gamma`. The verification-side lowering (``prox_indsimplex``) uses the KKT
    system of the mixed polyhedron ``{u >= 0, 1^T u = a}`` via the shared polyhedral-KKT
    helper.

    Parameters
    ----------
    a : float or LinearExpression
        Positive simplex sum (scalar), or a dim-1 ``LinearExpression`` parameter.
    """

    def __init__(self, a=1.0):
        super().__init__()
        if hasattr(a, "decomposition_dict"):  # dim-1 LinearExpression parameter
            if a.ndim != 1:
                raise ValueError("a must be a scalar or a dim-1 LinearExpression")
        elif a <= 0:
            raise ValueError("a must be positive")
        self.a = a

    def prox(self, arg, gamma=1.0):
        """
        Project ``arg`` onto ``{u : u >= 0, 1^T u = a}`` (independent of ``gamma``).

        Parameters
        ----------
        arg : np.ndarray
            Point to project.
        gamma : float, default 1.0
            Step size (irrelevant for an indicator; accepted for interface consistency).

        Returns
        -------
        np.ndarray
            The Euclidean projection of ``arg`` onto the simplex.
        """
        arg = np.asarray(arg, dtype=float)
        a = float(self.a.const[0]) if hasattr(self.a, "decomposition_dict") else float(self.a)
        u = cp.Variable(arg.shape[0])
        cp.Problem(cp.Minimize(cp.sum_squares(u - arg)), [u >= 0, cp.sum(u) == a]).solve()
        return np.asarray(u.value, dtype=float)
