import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SqL2Norm(ProxableFunction):
    r"""
    Squared $\ell_2$ norm.

    Represents

    $$
        f(x) = \tfrac{1}{2}\|x\|_2^2 = \tfrac{1}{2}\sum_i x_i^2.
    $$

    Convex and smooth.

    See Also
    --------
    Quadratic : General convex quadratic $\tfrac{1}{2} x^\top Q x + q^\top x$.

    Notes
    -----
    **Proximal operator.**

    $$
        \operatorname{prox}_{\gamma f}(v) = \frac{v}{1 + \gamma}.
    $$

    **Verification lowering.** ``prox_sql2norm`` is a pure scalar shrinkage; no
    step and no fresh leaf variables are introduced.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \tfrac{1}{2}\|x\|_2^2$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        return 0.5 * float(np.sum(x * x))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form
        $\operatorname{prox}_{\gamma f}(\text{arg}) = \text{arg}/(1 + \gamma)$.

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
        return arg / (1.0 + gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at ``x``.

        $\nabla f(x) = x$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $\nabla f(x)$.
        """
        return x.copy()
