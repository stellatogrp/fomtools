import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndAffine(ProxableFunction):
    r"""
    Indicator of the affine set $\{x : Ax = b\}$.

    Represents

    $$
        f(x) = I_{\{Ax = b\}}(x) = \begin{cases}
            0       & \text{if } Ax = b \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix of shape ``(m, n)``, assumed to have full row rank so
        that $AA^\top$ is invertible.
    b : float, np.ndarray, or LinearExpression
        Right-hand side of shape ``(m,)``. A scalar is broadcast; a
        ``LinearExpression`` is treated as a parameter by the verification
        lowering.

    Raises
    ------
    ValueError
        If ``A`` is not 2-D, or ``b`` does not match the row count of ``A``.

    See Also
    --------
    IndPolyhedral : Inequality-constrained counterpart $\{Ax \le b\}$.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the affine set,
    independent of $\gamma$. Its KKT conditions form the block system

    $$
        \begin{bmatrix} I_n & A^\top \\ A & 0_{m\times m} \end{bmatrix}
        \begin{bmatrix} u \\ \nu \end{bmatrix}
        =
        \begin{bmatrix} v \\ b \end{bmatrix},
    $$

    whose solution is $u = v - A^\top (AA^\top)^{-1}(Av - b)$.

    **Verification lowering.** ``prox_indaffine`` posts the two KKT rows as
    linear equalities on fresh iterates $u$ and $\nu$; it never forms
    $(AA^\top)^{-1}$ and needs no complementarity constraints.
    """

    def __init__(
        self,
        A: np.ndarray,
        b: float | np.ndarray | LinearExpression,
    ):
        super().__init__()
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be a 2-D matrix; got shape {A.shape}")
        m, n = A.shape
        if isinstance(b, np.ndarray):
            if b.shape != (m,):
                raise ValueError(
                    f"b must have shape ({m},) to match A's row count; got shape {b.shape}"
                )
        elif hasattr(b, "decomposition_dict"):  # LinearExpression
            if b.ndim != m:
                raise ValueError(f"b LinearExpression must have ndim {m}; got ndim {b.ndim}")
        # else: scalar — broadcasts in numpy arithmetic; LE-stacking lifts as a const.
        self.A = A
        self.b = b
        self.m = m
        self.n = n

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = I_{\{Ax = b\}}(x)$.

        Requires numeric ``b``; a ``LinearExpression`` ``b`` is supported only by
        the verification lowering.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $Ax = b$ (within floating-point tolerance), ``+inf``
            otherwise.

        Raises
        ------
        TypeError
            If ``b`` is a ``LinearExpression``.
        """
        if hasattr(self.b, "decomposition_dict"):
            raise TypeError(
                "IndAffine.eval requires numeric b; got LinearExpression. "
                "Symbolic b is supported only by the verification-side lowering."
            )
        if np.allclose(self.A @ x, self.b):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the affine set $\{x : Ax = b\}$.

        Solves the $m \times m$ system $AA^\top \nu = A\,\text{arg} - b$ and
        returns $\text{arg} - A^\top \nu$.

        Requires numeric ``b``; a ``LinearExpression`` ``b`` is supported only by
        the verification lowering.

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

        Raises
        ------
        TypeError
            If ``b`` is a ``LinearExpression``.
        """
        if hasattr(self.b, "decomposition_dict"):
            raise TypeError(
                "IndAffine.prox requires numeric b; got LinearExpression. "
                "Symbolic b is supported only by the verification-side lowering."
            )
        nu = np.linalg.solve(self.A @ self.A.T, self.A @ arg - self.b)
        return arg - self.A.T @ nu

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
