import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndHalfSpace(ProxableFunction):
    r"""
    Indicator of the closed half-space $\{x : a^\top x \le b\}$.

    Represents

    $$
        f(x) = I_{\{a^\top x \le b\}}(x) = \begin{cases}
            0       & \text{if } a^\top x \le b \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    a : np.ndarray
        Nonzero normal vector of shape ``(n,)``. (For $a = 0$ the set would be
        all of $\mathbb{R}^n$ or empty, and the projection is undefined.)
    b : float
        Offset.

    Raises
    ------
    ValueError
        If ``a`` is the zero vector.

    See Also
    --------
    IndPolyhedral : Intersection of several half-spaces $\{Ax \le b\}$.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the half-space,

    $$
        \operatorname{prox}_{\gamma f}(v)
        = v - \frac{\max(0,\, a^\top v - b)}{\|a\|_2^2}\, a,
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indhalfspace`` posts a single scalar
    ``Relu`` step on $(a^\top v - b)/\|a\|_2^2$ and lifts it back along $a$.
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
        Evaluate $f(x) = I_{\{a^\top x \le b\}}(x)$.

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

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the half-space $\{x : a^\top x \le b\}$.

        Computes $\text{arg} - \max(0,\, a^\top \text{arg} - b)\, a / \|a\|_2^2$.

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
        excess = max(0.0, float(np.dot(self.a, arg)) - self.b)
        return arg - (excess / self._norm_a_sq) * self.a

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
