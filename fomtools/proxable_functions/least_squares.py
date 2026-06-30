import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions.quadratic import Quadratic


class LeastSquares(Quadratic):
    r"""
    Least-squares loss.

    Represents the convex smooth function

    $$
        f(x) = \tfrac{1}{2}\|Ax - b\|_2^2.
    $$

    Expanding the square gives

    $$
        f(x) = \tfrac{1}{2}\,x^\top (A^\top A)\, x
               \;+\; (-A^\top b)^\top x
               \;+\; \tfrac{1}{2}\|b\|^2,
    $$

    so the prox is identical to :class:`Quadratic` with
    $Q = A^\top A$ and $q = -A^\top b$ — the constant $\tfrac{1}{2}\|b\|^2$
    drops out of $\arg\min$. The gradient $A^\top(Ax - b)$ equals
    $Qx + q$, so it is inherited as well. Only :meth:`eval` is overridden,
    to add back the $\tfrac{1}{2}\|b\|^2$ constant.

    ``b`` may be numeric data or a :class:`LinearExpression` parameter —
    the constructor computes ``q = -A.T @ b`` uniformly via
    ``LinearExpression.__rmatmul__``.
    """

    def __init__(
        self,
        A: np.ndarray,
        b: float | np.ndarray | LinearExpression,
    ):
        # ``A`` may be dense or scipy.sparse; normalize sparse to CSR so ``Q = A^T A``
        # stays sparse and flows into the (sparse-aware) Quadratic base.
        if spa.issparse(A):
            A = A.tocsr()
        else:
            A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be a 2-D matrix; got shape {A.shape}")
        m = A.shape[0]
        if isinstance(b, np.ndarray) and b.shape != (m,):
            raise ValueError(
                f"b must have shape ({m},) to match A's row count; got shape {b.shape}"
            )
        Q = A.T @ A  # symmetric PSD; sparse A -> sparse Q (Quadratic normalizes to CSR)
        q = -A.T @ b  # numeric b -> 1-D ndarray; LinearExpression b -> __rmatmul__
        super().__init__(Q=Q, q=q)
        self.A = A
        self.b = b

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \tfrac{1}{2}\|Ax - b\|_2^2$.

        Numeric ``A`` and ``b`` only (matches :meth:`Quadratic.eval`
        numeric-only precedent — ``LinearExpression`` ``b`` is supported
        for the verification-side lowering but not for this numpy
        reference).

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
            If ``self.b`` is a :class:`LinearExpression`.
        """
        if hasattr(self.b, "decomposition_dict"):
            raise TypeError(
                "LeastSquares.eval requires numeric b; got LinearExpression. "
                "Symbolic b is supported only by the verification-side lowering."
            )
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)
