import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndHalfSpace(ProxableFunction):
    r"""
    Indicator of the closed half-space $\{x \in \mathbb{R}^n : a^\top x \le b\}$.

    Represents

    $$
        I_{\{a^\top x \le b\}}(x) = \begin{cases}
            0       & \text{if } a^\top x \le b \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    The defining vector ``a`` must be nonzero; otherwise the half-space
    is either all of $\mathbb{R}^n$ (when $b \ge 0$) or empty (when
    $b < 0$), and the projection denominator $\|a\|^2$ vanishes.

    Notes
    -----
    The proximal operator is the Euclidean projection onto the half-space:

    $$
        \operatorname{prox}_{\gamma I}(v) \;=\; v - \frac{\max(0,\, a^\top v - b)}{\|a\|^2}\, a,
    $$

    independent of $\gamma$.
    """

    def __init__(self, a: np.ndarray, b: float):
        super().__init__()
        a = np.asarray(a, dtype=float)
        norm_a_sq = float(np.dot(a, a))
        if norm_a_sq == 0.0:
            raise ValueError("a must be nonzero")
        self.a = a
        self.b = float(b)
        self._norm_a_sq = norm_a_sq

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $I_{\{a^\top x \le b\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $a^\top x \le b$, ``+inf`` otherwise.
        """
        if float(np.dot(self.a, x)) <= self.b:
            return 0.0
        return np.inf

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``rhs`` onto the half-space $\{a^\top x \le b\}$.

        The prox of an indicator function is the projection onto its
        feasible set, independent of $\gamma$.

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size (ignored for indicator functions; kept for interface
            consistency).

        Returns
        -------
        np.ndarray
            The projection $\text{rhs} - \max(0, a^\top\text{rhs} - b)/\|a\|^2 \cdot a$.
        """
        excess = max(0.0, float(np.dot(self.a, rhs)) - self.b)
        return rhs - (excess / self._norm_a_sq) * self.a

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
