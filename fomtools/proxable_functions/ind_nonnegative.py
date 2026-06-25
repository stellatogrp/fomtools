import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndNonNegative(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the non-negative orthant.

    Represents

    $$
        I_{\geq 0}(x) = \begin{cases}
            0       & \text{if } x \geq 0 \text{ (element-wise)} \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator function $I_{\geq 0}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if every entry of ``x`` is non-negative, ``+inf`` otherwise.
        """
        if np.all(x >= 0):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the non-negative orthant.

        The prox of an indicator function is the projection onto its
        feasible set; for $I_{\geq 0}$ this is element-wise
        $\max(\text{arg}, 0)$, independent of $\gamma$.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size (ignored for indicator functions; kept for interface
            consistency).

        Returns
        -------
        np.ndarray
            The projection $\max(\text{arg}, 0)$.
        """
        return np.maximum(arg, 0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
