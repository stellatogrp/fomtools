import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndNonPositive(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the non-positive orthant.

    Represents

    $$
        I_{\leq 0}(x) = \begin{cases}
            0       & \text{if } x \leq 0 \text{ (element-wise)} \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator function $I_{\leq 0}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if every entry of ``x`` is non-positive, ``+inf`` otherwise.
        """
        if np.all(x <= 0):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the non-positive orthant.

        The prox of an indicator function is the projection onto its
        feasible set; for $I_{\leq 0}$ this is element-wise
        $\min(\text{arg}, 0)$, independent of $\gamma$.

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
            The projection $\min(\text{arg}, 0)$.
        """
        return np.minimum(arg, 0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
