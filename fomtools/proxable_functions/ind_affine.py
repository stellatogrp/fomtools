import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndAffine(ProxableFunction):
    r"""
    Indicator of the affine set $\{x \in \mathbb{R}^n : Ax = b\}$.

    Represents

    $$
        I_{\{Ax = b\}}(x) = \begin{cases}
            0       & \text{if } Ax = b \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    The matrix $A \in \mathbb{R}^{m \times n}$ is assumed to have full
    row rank (so $AA^\top$ is invertible and the projection is uniquely
    defined). ``b`` may be numeric data or a :class:`LinearExpression`
    parameter — both flow through the verification-side lowering
    uniformly.

    Notes
    -----
    The proximal operator is the Euclidean projection onto the affine
    set. KKT conditions on the Lagrangian $\tfrac{1}{2}\|u-v\|^2 +
    \nu^\top(Au - b)$ yield the block system

    $$
        \begin{bmatrix} I_n & A^\top \\ A & 0_{m\times m} \end{bmatrix}
        \begin{bmatrix} u \\ \nu \end{bmatrix}
        =
        \begin{bmatrix} v \\ b \end{bmatrix},
    $$

    independent of $\gamma$. Eliminating $u$ gives the small-system
    closed form $\nu = (AA^\top)^{-1}(Av - b)$, $u = v - A^\top \nu$
    used by the numpy :meth:`prox` reference. The verification model
    posts the two KKT rows as linear equalities and never forms
    $(AA^\top)^{-1}$.
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
                    f"b must have shape ({m},) to match A's row count; "
                    f"got shape {b.shape}"
                )
        elif hasattr(b, "decomposition_dict"):  # LinearExpression
            if b.ndim != m:
                raise ValueError(
                    f"b LinearExpression must have ndim {m}; got ndim {b.ndim}"
                )
        # else: scalar — broadcasts in numpy arithmetic; LE-stacking lifts as a const.
        self.A = A
        self.b = b
        self.m = m
        self.n = n

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $I_{\{Ax = b\}}(x)$.

        Numeric ``b`` only — passing a ``LinearExpression`` for ``b``
        here is supported only by the verification-side lowering, not
        this numpy reference.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $Ax = b$ (within numerical tolerance), ``+inf``
            otherwise.

        Raises
        ------
        TypeError
            If ``self.b`` is a :class:`LinearExpression`.
        """
        if hasattr(self.b, "decomposition_dict"):
            raise TypeError(
                "IndAffine.eval requires numeric b; got LinearExpression. "
                "Symbolic b is supported only by the verification-side lowering."
            )
        if np.allclose(self.A @ x, self.b):
            return 0.0
        return np.inf

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``rhs`` onto the affine set $\{Ax = b\}$.

        The prox of an indicator function is the projection onto its
        feasible set, independent of $\gamma$. Solves the small
        $m \times m$ system $\nu = (AA^\top)^{-1}(A\,\text{rhs} - b)$
        and returns $\text{rhs} - A^\top \nu$. Forming $AA^\top$ is
        allowed at the numpy reference level; the verification model
        never forms it.

        Numeric ``b`` only — passing a ``LinearExpression`` for ``b``
        here is supported only by the verification-side lowering, not
        this numpy reference.

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size (ignored for indicator functions).

        Returns
        -------
        np.ndarray
            The projection.

        Raises
        ------
        TypeError
            If ``self.b`` is a :class:`LinearExpression`.
        """
        if hasattr(self.b, "decomposition_dict"):
            raise TypeError(
                "IndAffine.prox requires numeric b; got LinearExpression. "
                "Symbolic b is supported only by the verification-side lowering."
            )
        nu = np.linalg.solve(self.A @ self.A.T, self.A @ rhs - self.b)
        return rhs - self.A.T @ nu

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
