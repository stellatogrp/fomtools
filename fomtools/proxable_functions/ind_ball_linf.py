import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndBallLinf(ProxableFunction):
    r"""
    Indicator of the $\ell_\infty$ ball $\{x : \|x\|_\infty \le r\}$.

    Represents

    $$
        f(x) = I_{\{\|x\|_\infty \leq r\}}(x) = \begin{cases}
            0       & \text{if } \|x\|_\infty \leq r \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    r : float
        Ball radius $r \ge 0$.

    Raises
    ------
    ValueError
        If ``r`` is negative.

    See Also
    --------
    IndBox : General box $[l, u]$.
    IndBallL1 : The $\ell_1$ ball.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the ball, an
    element-wise clip,

    $$
        \operatorname{prox}_{\gamma f}(v) = \operatorname{clip}(v, -r, r),
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indballlinf`` posts a single
    ``SaturatedLinear`` step clipping to $[-r, r]$.
    """

    def __init__(self, r: float):
        super().__init__()
        if r < 0:
            raise ValueError("r must be non-negative")
        self.r = r

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = I_{\{\|x\|_\infty \leq r\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $\|x\|_\infty \leq r$, ``+inf`` otherwise.
        """
        if np.linalg.norm(x, np.inf) <= self.r:
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the $\ell_\infty$ ball of radius $r$.

        Computes $\operatorname{clip}(\text{arg}, -r, r)$ element-wise.

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
        return np.clip(arg, -self.r, self.r)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
