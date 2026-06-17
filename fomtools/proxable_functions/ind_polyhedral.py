import cvxpy as cp
import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndPolyhedral(ProxableFunction):
    r"""
    Indicator of the polyhedron :math:`\mathcal{P} = \{u : Au \le b\}`.

    The proximal operator is the Euclidean projection onto :math:`\mathcal{P}`,

    $$
        \operatorname{prox}_{\gamma\,\mathbb{I}_{\mathcal P}}(x)
        = \arg\min_{u}\; \tfrac12\|u - x\|_2^2 \quad\text{s.t.}\quad Au \le b,
    $$

    which is independent of :math:`\gamma`. The verification-side lowering
    (``prox_indpolyhedral`` in the step decomposer) encodes the KKT system

    $$
        u = x - A^\top \lambda,\qquad 0 \le \lambda \perp (b - Au) \ge 0
    $$

    using the complementarity primitive. The MIP big-M encoding of the
    complementarity needs a finite upper bound on the dual :math:`\lambda`;
    supply it via ``dual_ub`` (OBBT tightens it further when enabled).

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix of shape ``(m, n)``.
    b : float, np.ndarray, or LinearExpression
        Constraint right-hand side. Scalars/arrays are stored as a length-``m``
        vector; a ``LinearExpression`` of dim ``m`` is kept as a parameter.
    dual_ub : None, float, or np.ndarray, optional
        Upper bound(s) on the dual variable :math:`\lambda \ge 0`. ``None``
        leaves :math:`\lambda` unbounded above (relies on OBBT; the MIP encoding
        raises if it cannot infer a finite bound).
    """

    def __init__(self, A, b, dual_ub=None):
        super().__init__()
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be a 2-D matrix; got shape {A.shape}")
        m, n = A.shape

        if hasattr(b, "decomposition_dict"):  # LinearExpression parameter
            if b.ndim != m:
                raise ValueError(f"b dim {b.ndim} must match A rows {m}")
        else:
            b = np.broadcast_to(np.asarray(b, dtype=float), (m,)).copy()

        self.A = A
        self.b = b
        self.m = m
        self.n = n
        self.dual_ub = dual_ub

    def prox(self, arg, gamma=1.0):
        """
        Project ``arg`` onto ``{u : Au <= b}``.

        Parameters
        ----------
        arg : np.ndarray
            Point to project, of dimension ``n``.
        gamma : float, default 1.0
            Step size (irrelevant for an indicator; accepted for interface
            consistency).

        Returns
        -------
        np.ndarray
            The Euclidean projection of ``arg`` onto the polyhedron.
        """
        arg = np.asarray(arg, dtype=float)
        b = self.b if isinstance(self.b, np.ndarray) else np.asarray(self.b.const, dtype=float)
        u = cp.Variable(self.n)
        cp.Problem(cp.Minimize(cp.sum_squares(u - arg)), [self.A @ u <= b]).solve()
        return np.asarray(u.value, dtype=float)
