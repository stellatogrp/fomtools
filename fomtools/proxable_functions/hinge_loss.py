import numpy as np

from fomtools.proxable_functions import ProxableFunction


class HingeLoss(ProxableFunction):
    r"""
    Hinge loss (unlabeled).

    Represents

    $$
        f(x) = \sum_i \max(0, 1 - x_i).
    $$

    Convex and piecewise linear.

    See Also
    --------
    LabeledHingeLoss : Labeled variant $\sum_i \max(0, 1 - y_i x_i)$.

    Notes
    -----
    **Proximal operator.** Element-wise separable; per coordinate,

    $$
        \operatorname{prox}_{\gamma f}(v) = \begin{cases}
            v + \gamma & v < 1 - \gamma \\
            1          & 1 - \gamma \le v \le 1 \\
            v          & v > 1,
        \end{cases}
    $$

    which collapses to
    $\operatorname{prox}_{\gamma f}(v) = v + \mathrm{satlin}(1 - v;\, 0,\, \gamma)$.

    **Verification lowering.** ``prox_hingeloss`` posts a single
    ``SaturatedLinear`` clip of the margin residual $1 - v$ to $[0, \gamma]$
    and adds it back to the input.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \sum_i \max(0, 1 - x_i)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        return float(np.sum(np.maximum(0.0, 1.0 - x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma f}(\text{arg})
            = \text{arg} + \mathrm{satlin}(1 - \text{arg};\, 0,\, \gamma).
        $$

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Step size $\gamma > 0$.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.
        """
        return arg + np.minimum(np.maximum(1.0 - arg, 0.0), gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at ``x``.

        Per coordinate,

        $$
            g_i = \begin{cases}
                -1  & x_i < 1 \\
                0   & x_i \geq 1,
            \end{cases}
        $$

        where $0 \in [-1, 0]$ is the canonical choice at the kink $x_i = 1$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the subgradient.

        Returns
        -------
        np.ndarray
            A subgradient $g \in \partial f(x)$.
        """
        return -(x < 1).astype(float)
