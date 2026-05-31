import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction


class IndBox(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the box $[l, u]$.

    Represents

    $$
        I_{[l, u]}(x) = \begin{cases}
            0       & \text{if } l \leq x \leq u \text{ (element-wise)} \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    The bounds ``l`` and ``u`` may be scalars, arrays, or
    ``LinearExpression`` instances. When ``l``/``u`` are
    ``LinearExpression``s, the symbolic lowering treats them as
    parameters and uses their interval bounds for satlin propagation.

    The numpy reference methods :meth:`eval` and :meth:`prox` are
    numeric-only; passing a ``LinearExpression`` for ``l`` or ``u`` is
    supported solely by the verification-side lowering.
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
        Evaluate the indicator function $I_{[l, u]}(x)$.

        Numeric ``l``/``u`` only — passing a ``LinearExpression`` for
        either bound here is supported only by the verification-side
        lowering, not this numpy reference.

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
            If either ``self.l`` or ``self.u`` is a :class:`LinearExpression`.
        """
        if hasattr(self.l, "decomposition_dict") or hasattr(self.u, "decomposition_dict"):
            raise TypeError(
                "IndBox.eval requires numeric l and u; got LinearExpression. "
                "Symbolic bounds are supported only by the verification-side lowering."
            )
        if np.all((x >= self.l) & (x <= self.u)):
            return 0.0
        return np.inf

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Project ``rhs`` onto the box $[l, u]$ element-wise.

        The prox of an indicator function is the projection onto its
        feasible set, independent of $\gamma$.

        Numeric ``l``/``u`` only — passing a ``LinearExpression`` for
        either bound here is supported only by the verification-side
        lowering, not this numpy reference.

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
            The projection $\min(\max(\text{rhs}, l), u)$.

        Raises
        ------
        TypeError
            If either ``self.l`` or ``self.u`` is a :class:`LinearExpression`.
        """
        if hasattr(self.l, "decomposition_dict") or hasattr(self.u, "decomposition_dict"):
            raise TypeError(
                "IndBox.prox requires numeric l and u; got LinearExpression. "
                "Symbolic bounds are supported only by the verification-side lowering."
            )
        return np.minimum(np.maximum(rhs, self.l), self.u)

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
