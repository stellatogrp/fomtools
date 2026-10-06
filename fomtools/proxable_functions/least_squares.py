import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions.quadratic import Quadratic


class LeastSquares(Quadratic):
    r"""
    Least-squares loss.

    Represents

    $$
        f(x) = \tfrac{1}{2}\|Ax - b\|_2^2
             = \tfrac{1}{2}\,x^\top (A^\top A)\, x - (A^\top b)^\top x
               + \tfrac{1}{2}\|b\|_2^2.
    $$

    Convex and smooth. This is a ``Quadratic`` with $Q = A^\top A$ and
    $q = -A^\top b$ plus a constant, so ``prox`` and ``grad`` are inherited;
    only ``eval`` is overridden to add back $\tfrac{1}{2}\|b\|_2^2$.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix
        Data matrix of shape ``(m, n)``. Sparse input is kept sparse (CSR).
    b : float, np.ndarray, or LinearExpression
        Target of shape ``(m,)``. A ``LinearExpression`` is treated as a
        parameter by the verification lowering.

    Raises
    ------
    ValueError
        If ``A`` is not 2-D, or an array ``b`` has the wrong shape.

    See Also
    --------
    Quadratic : Parent class.

    Notes
    -----
    **Proximal operator.**

    $$
        (I + \gamma A^\top A)\, u = v + \gamma A^\top b.
    $$

    **Verification lowering.** Registered to ``prox_quadratic``: a single linear
    equality on a fresh iterate, as for ``Quadratic``.
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

        Requires numeric ``b``; a ``LinearExpression`` ``b`` is supported only by
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
            If ``b`` is a ``LinearExpression``.
        """
        if hasattr(self.b, "decomposition_dict"):
            raise TypeError(
                "LeastSquares.eval requires numeric b; got LinearExpression. "
                "Symbolic b is supported only by the verification-side lowering."
            )
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)
