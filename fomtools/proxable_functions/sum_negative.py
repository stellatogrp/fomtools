import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SumNegative(ProxableFunction):
    r"""
    Sum of negative parts.

    Represents the asymmetric piecewise-linear convex function

    $$
        f(x) = \sum_i \max(0, -x_i),
    $$

    the mirror image of ``SumPositive`` under $x \mapsto -x$. It
    penalizes only the negative coordinates and leaves non-negative ones
    untouched.

    Notes
    -----
    The proximal operator is element-wise separable. For a single
    coordinate $v$,

    $$
        \operatorname{prox}_{\gamma f}(v) = \begin{cases}
            v          & v > 0 \\
            0          & -\gamma \le v \le 0 \\
            v + \gamma & v < -\gamma,
        \end{cases}
    $$

    which collapses to

    $$
        \operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v;\, -\gamma,\, 0).
    $$

    Equivalently, $\operatorname{prox}_{\gamma f}(v)
    = -\operatorname{prox}_{\gamma f_+}(-v)$ where $f_+$ is
    ``SumPositive``.
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
            The function value.
        """
        return float(np.sum(np.maximum(0.0, -x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point.

        Uses the closed form
        $\operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v; -\gamma, 0)$.

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
        return arg - np.minimum(np.maximum(arg, -gamma), 0.0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at $x$.

        Per coordinate,

        $$
            g_i = \begin{cases}
                -1  & x_i < 0 \\
                0   & x_i \geq 0.
            \end{cases}
        $$

        At $x_i = 0$ this returns $0$, an element of the
        subdifferential $[-1, 0]$.

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
