import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndPoint(ProxableFunction):
    r"""
    Indicator of the singleton $\{p\}$.

    Represents

    $$
        f(x) = I_{\{p\}}(x) = \begin{cases}
            0       & \text{if } x = p \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    p : float or np.ndarray
        The point. A scalar is broadcast against the input.

    See Also
    --------
    IndBox : The box $[l, u]$; $\{p\}$ is the case $l = u = p$.

    Notes
    -----
    **Proximal operator.** The constant map

    $$
        \operatorname{prox}_{\gamma f}(v) = p,
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indpoint`` posts a degenerate
    ``SaturatedLinear`` step with $l = u = p$.
    """

    def __init__(self, p: float | np.ndarray):
        super().__init__()
        self.p = p

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = I_{\{p\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $x = p$ (within floating-point tolerance), ``+inf``
            otherwise.
        """
        if np.allclose(x, self.p):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto $\{p\}$.

        Returns $p$ broadcast to the shape of ``arg``; ``arg`` is used only for
        its shape.

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
        return np.broadcast_to(self.p, arg.shape).copy()

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
