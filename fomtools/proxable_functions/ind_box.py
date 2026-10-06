import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndBox(ProxableFunction):
    r"""
    Indicator of the box $[l, u]$.

    Represents

    $$
        f(x) = I_{[l, u]}(x) = \begin{cases}
            0       & \text{if } l \leq x \leq u \text{ (element-wise)} \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    Parameters
    ----------
    l : float, np.ndarray, or LinearExpression
        Lower bound. A scalar is broadcast; a ``LinearExpression`` is treated as
        a parameter by the verification lowering.
    u : float, np.ndarray, or LinearExpression
        Upper bound, with the same conventions as ``l``.

    See Also
    --------
    IndBallLinf : Symmetric box $[-r, r]$.
    IndNonNegative : One-sided box $[0, +\infty)$.

    Notes
    -----
    **Proximal operator.** The Euclidean projection onto the box,

    $$
        \operatorname{prox}_{\gamma f}(v) = \min(\max(v, l), u)
        \quad \text{(element-wise)},
    $$

    independent of $\gamma$.

    **Verification lowering.** ``prox_indbox`` posts a single
    ``SaturatedLinear`` step clipping to $[l, u]$. ``LinearExpression`` bounds
    enter through their interval bounds during bound propagation.
    """

    def __init__(
        self,
        l: float | np.ndarray | LinearExpression,
        u: float | np.ndarray | LinearExpression,
    ):
        super().__init__()
        self.l = l
        self.u = u

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = I_{[l, u]}(x)$.

        Requires numeric ``l`` and ``u``; ``LinearExpression`` bounds are
        supported only by the verification lowering.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $l \leq x \leq u$ element-wise, ``+inf`` otherwise.

        Raises
        ------
        TypeError
            If ``l`` or ``u`` is a ``LinearExpression``.
        """
        if hasattr(self.l, "decomposition_dict") or hasattr(self.u, "decomposition_dict"):
            raise TypeError(
                "IndBox.eval requires numeric l and u; got LinearExpression. "
                "Symbolic bounds are supported only by the verification-side lowering."
            )
        if np.all((x >= self.l) & (x <= self.u)):
            return 0.0
        return np.inf

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``arg`` onto the box $[l, u]$.

        Computes $\min(\max(\text{arg}, l), u)$ element-wise.

        Requires numeric ``l`` and ``u``; ``LinearExpression`` bounds are
        supported only by the verification lowering.

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

        Raises
        ------
        TypeError
            If ``l`` or ``u`` is a ``LinearExpression``.
        """
        if hasattr(self.l, "decomposition_dict") or hasattr(self.u, "decomposition_dict"):
            raise TypeError(
                "IndBox.prox requires numeric l and u; got LinearExpression. "
                "Symbolic bounds are supported only by the verification-side lowering."
            )
        return np.minimum(np.maximum(arg, self.l), self.u)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Not defined: indicator functions are not differentiable.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Indicator functions are not differentiable.")
