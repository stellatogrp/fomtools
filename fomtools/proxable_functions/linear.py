import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class Linear(ProxableFunction):
    r"""
    Linear cost.

    Represents

    $$
        f(x) = c^\top x.
    $$

    Convex and smooth.

    Parameters
    ----------
    c : float, np.ndarray, or LinearExpression
        Cost vector. A scalar is broadcast; a ``LinearExpression`` is treated as
        a parameter by the verification lowering.

    Notes
    -----
    **Proximal operator.**

    $$
        \operatorname{prox}_{\gamma f}(v) = v - \gamma\, c.
    $$

    **Verification lowering.** ``prox_linear`` is a pure affine shift; no step
    and no fresh leaf variables are introduced.
    """

    def __init__(self, c: float | np.ndarray | LinearExpression):
        super().__init__()
        self.c = c

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = c^\top x$.

        Requires numeric ``c``; a ``LinearExpression`` ``c`` is supported only by
        the verification lowering.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        return float(np.dot(self.c, x))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form
        $\operatorname{prox}_{\gamma f}(\text{arg}) = \text{arg} - \gamma c$.

        Requires numeric ``c``; a ``LinearExpression`` ``c`` is supported only by
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
        return arg - gamma * self.c

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at ``x``.

        $\nabla f(x) = c$, broadcast to the shape of ``x``.

        Requires numeric ``c``; a ``LinearExpression`` ``c`` is supported only by
        the verification lowering.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $\nabla f(x)$.
        """
        return np.broadcast_to(self.c, x.shape).copy()
