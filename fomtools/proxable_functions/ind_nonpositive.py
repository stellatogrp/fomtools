import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndNonPositive(ProxableFunction):
    r"""
    Indicator of the non-positive orthant.

    Represents

    $$
        f(x) = I_{\{x \leq 0\}}(x) = \begin{cases}
            0       & \text{if } x \leq 0 \text{ (element-wise)} \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    See Also
    --------
    IndNonNegative : The non-negative orthant.
    IndBox : General box $[l, u]$.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the orthant,

    $$
        \operatorname{prox}_{\gamma f}(v) = \min(v, 0) \quad \text{(element-wise)},
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indnonpositive`` posts a single ``Relu``
    step via $\min(v, 0) = -\operatorname{ReLU}(-v)$.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = I_{\{x \leq 0\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $x \leq 0$ element-wise, ``+inf`` otherwise.
        """
        if np.all(x <= 0):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the non-positive orthant.

        Computes $\min(\text{arg}, 0)$ element-wise.

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
        """
        return np.minimum(arg, 0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
