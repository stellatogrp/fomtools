import cvxpy as cp
import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndPolyhedral(ProxableFunction):
    r"""
    Indicator of the polyhedron $\{x : Ax \le b\}$.

    Represents

    $$
        f(x) = I_{\{Ax \le b\}}(x) = \begin{cases}
            0       & \text{if } Ax \le b \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix of shape ``(m, n)``.
    b : float, np.ndarray, or LinearExpression
        Right-hand side. A scalar or array is broadcast to shape ``(m,)``; a
        dim-``m`` ``LinearExpression`` is treated as a parameter by the
        verification lowering.
    dual_ub : float, np.ndarray, or None, default None
        Upper bound(s) on the multipliers $\lambda \ge 0$. Required for
        verification: the multipliers of a general polyhedron cannot be bounded
        automatically, so the verification lowering raises ``ValueError`` when
        this is ``None``. Not used by ``prox``.

    Raises
    ------
    ValueError
        If ``A`` is not 2-D, or a ``LinearExpression`` ``b`` does not have
        dimension ``m``.

    See Also
    --------
    IndHalfSpace : Single half-space, with a closed-form projection.
    IndAffine : Equality-constrained counterpart $\{Ax = b\}$.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the polyhedron,

    $$
        \operatorname{prox}_{\gamma f}(v)
        = \arg\min_{u}\ \tfrac{1}{2}\|u - v\|_2^2
        \quad\text{s.t.}\quad Au \le b,
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indpolyhedral`` encodes the KKT system

    $$
        u = v - A^\top \lambda,\qquad 0 \le \lambda \perp (b - Au) \ge 0
    $$

    with complementarity constraints. The MIP big-M encoding needs the finite
    bound ``dual_ub`` on $\lambda$; OBBT tightens it further when enabled.
    """

    def __init__(
        self,
        A: np.ndarray,
        b: float | np.ndarray | LinearExpression,
        dual_ub: float | np.ndarray | None = None,
    ):
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

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the polyhedron $\{x : Ax \le b\}$.

        Solved numerically with CVXPY (Clarabel) as an independent reference; a
        ``LinearExpression`` ``b`` uses its constant value.

        Parameters
        ----------
        arg : np.ndarray
            Input point of dimension ``n``.
        gamma : float, default 1.0
            Step size; unused, since the projection is independent of $\gamma$.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.
        """
        arg = np.asarray(arg, dtype=float)
        b = self.b if isinstance(self.b, np.ndarray) else np.asarray(self.b.const, dtype=float)
        u = cp.Variable(self.n)
        # Pin CLARABEL: deterministic and accurate (~1e-7) across platforms, unlike
        # cvxpy's auto-selected default (OSQP/SCS) which can return ~1e-3 solutions.
        cp.Problem(cp.Minimize(cp.sum_squares(u - arg)), [self.A @ u <= b]).solve(
            solver=cp.CLARABEL
        )
        return np.asarray(u.value, dtype=float)
