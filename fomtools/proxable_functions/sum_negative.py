import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SumNegative(ProxableFunction):
    r"""
    Sum of negative parts.

    Represents

    $$
        f(x) = \sum_i \max(0, -x_i).
    $$

    Convex and piecewise linear. Penalizes only the negative coordinates and
    leaves non-negative ones untouched.

    See Also
    --------
    SumPositive : Mirror image under $x \mapsto -x$.
    NormL1 : Two-sided counterpart $\lambda \|x\|_1$.

    Notes
    -----
    **Proximal operator.** Element-wise separable; per coordinate,

    $$
        \operatorname{prox}_{\gamma f}(v) = \begin{cases}
            v          & v > 0 \\
            0          & -\gamma \le v \le 0 \\
            v + \gamma & v < -\gamma,
        \end{cases}
    $$

    which collapses to
    $\operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v;\, -\gamma,\, 0)$.

    **Verification lowering.** ``prox_sumnegative`` posts a single
    ``SaturatedLinear`` clip to $[-\gamma, 0]$ and subtracts it from the input.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \sum_i \max(0, -x_i)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        return float(np.sum(np.maximum(0.0, -x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma f}(\text{arg})
            = \text{arg} - \mathrm{satlin}(\text{arg};\, -\gamma,\, 0).
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
        return arg - np.minimum(np.maximum(arg, -gamma), 0.0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at ``x``.

        Per coordinate,

        $$
            g_i = \begin{cases}
                -1  & x_i < 0 \\
                0   & x_i \geq 0,
            \end{cases}
        $$

        where $0 \in [-1, 0]$ is the canonical choice at the kink $x_i = 0$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the subgradient.

        Returns
        -------
        np.ndarray
            A subgradient $g \in \partial f(x)$.
        """
        return -(x < 0).astype(float)
