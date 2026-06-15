import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SumPositive(ProxableFunction):
    r"""
    Sum of positive parts.

    Represents the asymmetric piecewise-linear convex function

    $$
        f(x) = \sum_i \max(0, x_i),
    $$

    a one-sided variant of $\|x\|_1$ that penalizes only the positive
    coordinates and leaves negative ones untouched.

    Notes
    -----
    The proximal operator is element-wise separable. For a single
    coordinate $v$,

    $$
        \operatorname{prox}_{\gamma f}(v) = \begin{cases}
            v - \gamma & v > \gamma \\
            0          & 0 \le v \le \gamma \\
            v          & v < 0,
        \end{cases}
    $$

    which collapses to

    $$
        \operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v;\, 0,\, \gamma).
    $$
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \sum_i \max(0, x_i)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value.
        """
        return float(np.sum(np.maximum(0.0, x)))

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point.

        Uses the closed form
        $\operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v; 0, \gamma)$.

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point.
        """
        return rhs - np.minimum(np.maximum(rhs, 0.0), gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at $x$.

        Returns the indicator $1\{x_i > 0\}$ element-wise; at
        $x_i = 0$ returns $0$ (an element of the subdifferential
        $[0, 1]$).

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the subgradient.

        Returns
        -------
        np.ndarray
            A subgradient $g \in \partial f(x)$.
        """
        return (x > 0).astype(float)
