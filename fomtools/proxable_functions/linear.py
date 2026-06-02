import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class Linear(ProxableFunction):
    r"""
    Linear cost function.

    Represents the convex function

    $$
        f(x) = c^\top x,
    $$

    where ``c`` may be a scalar (broadcast), an array of per-coordinate
    weights, or a ``LinearExpression`` parameter — the symbolic lowering
    treats all three uniformly via ``LinearExpression`` arithmetic.

    Notes
    -----
    The proximal operator admits the closed form

    $$
        \operatorname{prox}_{\gamma f}(v) \;=\; v - \gamma\, c,
    $$

    which is a pure affine shift — no fresh leaf variables or step
    constraints are introduced by the decomposer.
    """

    def __init__(self, c: float | np.ndarray | LinearExpression):
        super().__init__()
        self.c = c

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = c^\top x$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $c^\top x$.
        """
        return float(np.dot(self.c, x))

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point of the linear function.

        Uses the closed form
        $\operatorname{prox}_{\gamma f}(\text{rhs}) = \text{rhs} - \gamma c$.

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point $\text{rhs} - \gamma c$.
        """
        return rhs - gamma * self.c

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at $x$.

        Returns $c$, broadcast to the shape of ``x``.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $\nabla f(x) = c$.
        """
        return np.broadcast_to(self.c, x.shape).copy()
