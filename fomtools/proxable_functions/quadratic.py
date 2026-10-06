import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction
from fomtools.utils.matrix_utils import _PSD_TOL, _min_eig, _sym_error


class Quadratic(ProxableFunction):
    r"""
    Convex quadratic.

    Represents

    $$
        f(x) = \tfrac{1}{2}\,x^\top Q x + q^\top x.
    $$

    Convex and smooth.

    Parameters
    ----------
    Q : np.ndarray or scipy.sparse matrix
        Symmetric positive semidefinite matrix of shape ``(n, n)``. Sparse input
        is kept sparse (CSR).
    q : float, np.ndarray, or LinearExpression
        Linear term of shape ``(n,)``. A ``LinearExpression`` is treated as a
        parameter by the verification lowering.

    Raises
    ------
    ValueError
        If ``Q`` is not square, not symmetric, or not positive semidefinite
        (up to a numerical tolerance), or an array ``q`` has the wrong shape.

    See Also
    --------
    LeastSquares : The case $Q = A^\top A$, $q = -A^\top b$.
    SqL2Norm : The case $Q = I$, $q = 0$.

    Notes
    -----
    **Proximal operator.** The unique solution $u$ of the linear system

    $$
        (I + \gamma Q)\, u = v - \gamma q.
    $$

    **Verification lowering.** ``prox_quadratic`` posts this single linear
    equality on a fresh iterate $u$; no inverse is formed and no duals or
    complementarity constraints are introduced.
    """

    def __init__(
        self,
        Q: np.ndarray,
        q: float | np.ndarray | LinearExpression,
    ):
        super().__init__()
        # ``Q`` may be dense or scipy.sparse; normalize sparse to CSR (supports T,
        # arithmetic, and ``LinearExpression.__rmatmul__``), mirroring QuadraticProgram.
        if spa.issparse(Q):
            Q = Q.tocsr()
        else:
            Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be a square 2-D matrix; got shape {Q.shape}")
        n = Q.shape[0]
        # The prox (I + gamma*Q) x = ... is well posed only for symmetric PSD Q.
        sym_err = _sym_error(Q)
        if sym_err > _PSD_TOL:
            raise ValueError(f"Q must be symmetric; max|Q - Q^T| = {sym_err:.3e}")
        min_eig = _min_eig(Q)
        if min_eig < -_PSD_TOL:
            raise ValueError(
                f"Q must be positive semidefinite; smallest eigenvalue {min_eig:.3e} < -{_PSD_TOL:g}"
            )
        if isinstance(q, np.ndarray) and q.shape != (n,):
            raise ValueError(f"q must have shape ({n},) to match Q; got shape {q.shape}")
        self.Q = Q
        self.q = q
        self.n = n

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \tfrac{1}{2}\,x^\top Q x + q^\top x$.

        Requires numeric ``q``; a ``LinearExpression`` ``q`` is supported only by
        the verification lowering.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.

        Raises
        ------
        TypeError
            If ``q`` is a ``LinearExpression``.
        """
        if hasattr(self.q, "decomposition_dict"):
            raise TypeError(
                "Quadratic.eval requires numeric q; got LinearExpression. "
                "Symbolic q is supported only by the verification-side lowering."
            )
        return 0.5 * float(x @ self.Q @ x) + float(np.dot(self.q, x))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Solves $(I + \gamma Q)\, u = \text{arg} - \gamma q$ directly (sparse
        solve when ``Q`` is sparse).

        Requires numeric ``q``; a ``LinearExpression`` ``q`` is supported only by
        the verification lowering.

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
        rhs = arg - gamma * self.q
        if spa.issparse(self.Q):
            # Keep (I + gamma*Q) sparse and solve directly — a dense ``np.eye(n) +
            # gamma*Q`` would silently produce a numpy.matrix of the wrong shape.
            M = spa.eye(self.n, format="csr") + gamma * self.Q
            return spa.linalg.spsolve(M.tocsc(), rhs)
        M = np.eye(self.n) + gamma * self.Q
        return np.linalg.solve(M, rhs)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at ``x``.

        $\nabla f(x) = Qx + q$.

        Requires numeric ``q``; a ``LinearExpression`` ``q`` is supported only by
        the verification lowering.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $\nabla f(x)$.

        Raises
        ------
        TypeError
            If ``q`` is a ``LinearExpression``.
        """
        if hasattr(self.q, "decomposition_dict"):
            raise TypeError(
                "Quadratic.grad requires numeric q; got LinearExpression. "
                "Symbolic q is supported only by the verification-side lowering."
            )
        return self.Q @ x + self.q
