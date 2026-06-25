import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SqL2Norm(ProxableFunction):
    r"""
    Squared $\ell_2$-norm.

    Represents the smooth convex function

    $$
        f(x) = \tfrac{1}{2}\|x\|_2^2 = \tfrac{1}{2}\sum_i x_i^2.
    $$

    Notes
    -----
    The proximal operator admits the closed form

    $$
        \operatorname{prox}_{\gamma f}(v) \;=\; \frac{v}{1 + \gamma},
    $$

    a pure scalar shrinkage — no fresh leaf variables or step constraints
    are introduced by the decomposer.
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
            The function value $\tfrac{1}{2}\|x\|_2^2$.
        """
        return 0.5 * float(np.sum(x * x))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point of the squared $\ell_2$-norm.

        Uses the closed form
        $\operatorname{prox}_{\gamma f}(\text{arg}) = \text{arg}/(1 + \gamma)$.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point $\text{arg}/(1 + \gamma)$.
        """
        return arg / (1.0 + gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at $x$.

        Returns $\nabla f(x) = x$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $x$.
        """
        return x.copy()
