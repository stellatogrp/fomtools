import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SumPositive(ProxableFunction):
    r"""
    Sum of positive parts.

    Represents

    $$
        f(x) = \sum_i \max(0, x_i).
    $$

    Convex and piecewise linear. A one-sided variant of $\|x\|_1$ that
    penalizes only the positive coordinates and leaves negative ones
    untouched.

    See Also
    --------
    SumNegative : Mirror image under $x \mapsto -x$.
    NormL1 : Two-sided counterpart $\lambda \|x\|_1$.

    Notes
    -----
    **Proximal operator.** Element-wise separable; per coordinate,

    $$
        \operatorname{prox}_{\gamma f}(v) = \begin{cases}
            v - \gamma & v > \gamma \\
            0          & 0 \le v \le \gamma \\
            v          & v < 0,
        \end{cases}
    $$

    which collapses to
    $\operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v;\, 0,\, \gamma)$.

    **Verification lowering.** ``prox_sumpositive`` posts a single
    ``SaturatedLinear`` clip to $[0, \gamma]$ and subtracts it from the input.
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
            The function value $f(x)$.
        """
        return float(np.sum(np.maximum(0.0, x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma f}(\text{arg})
            = \text{arg} - \mathrm{satlin}(\text{arg};\, 0,\, \gamma).
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
        return arg - np.minimum(np.maximum(arg, 0.0), gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at ``x``.

        Per coordinate,

        $$
            g_i = \begin{cases}
                1   & x_i > 0 \\
                0   & x_i \leq 0,
            \end{cases}
        $$

        where $0 \in [0, 1]$ is the canonical choice at the kink $x_i = 0$.

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
