import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndNonNegative(ProxableFunction):
    r"""
    Indicator of the non-negative orthant.

    Represents

    $$
        f(x) = I_{\{x \geq 0\}}(x) = \begin{cases}
            0       & \text{if } x \geq 0 \text{ (element-wise)} \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    See Also
    --------
    IndNonPositive : The non-positive orthant.
    IndBox : General box $[l, u]$.
    SoftNonNegative : Smooth quadratic relaxation.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the orthant,

    $$
        \operatorname{prox}_{\gamma f}(v) = \max(v, 0) \quad \text{(element-wise)},
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indnonnegative`` posts a single ``Relu``
    step.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = I_{\{x \geq 0\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $x \geq 0$ element-wise, ``+inf`` otherwise.
        """
        if np.all(x >= 0):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the non-negative orthant.

        Computes $\max(\text{arg}, 0)$ element-wise.

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
        return np.maximum(arg, 0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
