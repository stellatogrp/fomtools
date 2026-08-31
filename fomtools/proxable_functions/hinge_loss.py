import numpy as np

from fomtools.proxable_functions import ProxableFunction


class HingeLoss(ProxableFunction):
    r"""
    Standard hinge loss (no labels).

    Represents the element-wise sum

    $$
        g(x) = \sum_i \max(0, 1 - x_i),
    $$

    convex and piecewise linear.

    Notes
    -----
    The proximal operator is element-wise separable. Per coordinate $v$,

    $$
        \operatorname{prox}_{\gamma g}(v) = \begin{cases}
            v + \gamma & v < 1 - \gamma \\
            1          & 1 - \gamma \le v \le 1 \\
            v          & v > 1,
        \end{cases}
    $$

    which collapses to

    $$
        \operatorname{prox}_{\gamma g}(v) = v + \mathrm{satlin}(1 - v;\, 0,\, \gamma).
    $$

    Labels are not part of this class — see ``LabeledHingeLoss``
    for the labeled variant $g_y(x) = \sum_i \max(0, 1 - y_i x_i)$.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""Evaluate $g(x) = \sum_i \max(0, 1 - x_i)$."""
        return float(np.sum(np.maximum(0.0, 1.0 - x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point.

        Uses $\operatorname{prox}_{\gamma g}(v) = v + \mathrm{satlin}(1 - v; 0, \gamma)$.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point.
        """
        return arg + np.minimum(np.maximum(1.0 - arg, 0.0), gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Subgradient, per coordinate:

        $$
            g_i = \begin{cases}
                -1  & x_i < 1 \\
                0   & x_i \geq 1.
            \end{cases}
        $$

        At $x_i = 1$ the canonical choice $0 \in [-1, 0]$ is returned.
        """
        return -(x < 1).astype(float)
