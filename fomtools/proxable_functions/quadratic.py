import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class Quadratic(ProxableFunction):
    r"""
    Convex quadratic.

    Represents

    $$
        f(x) = \tfrac{1}{2}\,x^\top Q x + q^\top x,
    $$

    with $Q \in \mathbb{R}^{n \times n}$ positive semidefinite. The linear
    term ``q`` may be numeric data or a :class:`LinearExpression`
    parameter — the verification-side lowering treats both uniformly via
    ``LinearExpression`` arithmetic.

    Notes
    -----
    The proximal operator is the unique smooth-stationarity solution

    $$
        (I + \gamma Q)\, u \;=\; \text{rhs} - \gamma q,
    $$

    which is a single linear system. The :meth:`prox` numpy reference
    uses :func:`numpy.linalg.solve`; the verification-side lowering posts
    the equality directly and lets the solver handle it — no inverse or
    pseudoinverse is formed.
    """

    def __init__(
        self,
        Q: np.ndarray,
        q: float | np.ndarray | LinearExpression,
    ):
        super().__init__()
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be a square 2-D matrix; got shape {Q.shape}")
        n = Q.shape[0]
        if isinstance(q, np.ndarray) and q.shape != (n,):
            raise ValueError(
                f"q must have shape ({n},) to match Q; got shape {q.shape}"
            )
        self.Q = Q
        self.q = q
        self.n = n

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \tfrac{1}{2}\,x^\top Q x + q^\top x$.

        Numeric ``q`` only — passing a ``LinearExpression`` for ``q``
        here is supported only by the verification-side lowering, not
        this numpy reference.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value.

        Raises
        ------
        TypeError
            If ``self.q`` is a :class:`LinearExpression`.
        """
        if hasattr(self.q, "decomposition_dict"):
            raise TypeError(
                "Quadratic.eval requires numeric q; got LinearExpression. "
                "Symbolic q is supported only by the verification-side lowering."
            )
        return 0.5 * float(x @ self.Q @ x) + float(np.dot(self.q, x))

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point.

        Solves $(I + \gamma Q)\, u = \text{rhs} - \gamma q$ via
        :func:`numpy.linalg.solve`.

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point ``u``.
        """
        M = np.eye(self.n) + gamma * self.Q
        return np.linalg.solve(M, rhs - gamma * self.q)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at $x$: $\nabla f(x) = Qx + q$.

        Numeric ``q`` only.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient.

        Raises
        ------
        TypeError
            If ``self.q`` is a :class:`LinearExpression`.
        """
        if hasattr(self.q, "decomposition_dict"):
            raise TypeError(
                "Quadratic.grad requires numeric q; got LinearExpression. "
                "Symbolic q is supported only by the verification-side lowering."
            )
        return self.Q @ x + self.q
