import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndBallLinf(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the $\ell_\infty$ ball
    $\{ x : \|x\|_\infty \leq r \}$.

    Represents

    $$
        I_{\|\cdot\|_\infty \leq r}(x) = \begin{cases}
            0       & \text{if } \|x\|_\infty \leq r \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$
    """

    def __init__(self, r: float):
        super().__init__()
        if r < 0:
            raise ValueError("r must be non-negative")
        self.r = r

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator of the $\ell_\infty$ ball.

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
        Project ``arg`` onto the $\ell_\infty$ ball, element-wise clip to
        $[-r, r]$.

        The prox of an indicator function is the projection onto its
        feasible set, independent of $\gamma$.

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
            The projection $\operatorname{clip}(\text{arg}, -r, r)$.
        """
        return np.clip(arg, -self.r, self.r)

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
