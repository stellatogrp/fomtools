import numpy as np

from fomtools.proxable_functions import ProxableFunction


class HuberLoss(ProxableFunction):
    r"""
    Element-wise Huber loss.

    The Huber function with threshold $\delta \geq 0$ is applied
    per-coordinate and summed:

    $$
        h(x_i) = \begin{cases}
            \tfrac{1}{2} x_i^2                          & |x_i| \leq \delta \\
            \delta |x_i| - \tfrac{1}{2}\delta^2          & |x_i| > \delta,
        \end{cases}
        \qquad
        f(x) = \sum_i h(x_i).
    $$

    This is the standard element-wise robust loss — *not* the
    multivariate norm-based "vector Huber" parameterized by a smoothing
    parameter and a radius.

    Notes
    -----
    The proximal operator admits the closed form (per-coordinate):

    $$
        \operatorname{prox}_{\gamma f}(v)_i = \begin{cases}
            \dfrac{v_i}{1+\gamma}
                & |v_i| \leq \delta(1+\gamma), \\[6pt]
            v_i - \gamma\,\delta\,\operatorname{sign}(v_i)
                & |v_i| > \delta(1+\gamma).
        \end{cases}
    $$

    The two branches join continuously at the kink $|v_i| = \delta(1+\gamma)$
    (both evaluate to $\delta\,\operatorname{sign}(v_i)$). The same prox can
    be written using element-wise soft thresholding,

    $$
        \operatorname{prox}_{\gamma f}(v)
            = \frac{v + \gamma\, S_{\delta(1+\gamma)}(v)}{1+\gamma},
    $$

    which is the form used by the step decomposer's
    ``prox_huberloss`` lowering.
    """

    def __init__(self, delta: float):
        super().__init__()
        if delta < 0:
            raise ValueError("delta must be non-negative")
        self.delta = delta

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \sum_i h(x_i)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $\sum_i h(x_i)$.
        """
        ax = np.abs(x)
        per_coord = np.where(
            ax <= self.delta,
            0.5 * x * x,
            self.delta * ax - 0.5 * self.delta**2,
        )
        return float(np.sum(per_coord))

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point of the element-wise Huber loss.

        Uses the closed form per coordinate

        $$
            \operatorname{prox}_{\gamma f}(\text{rhs})_i = \begin{cases}
                \text{rhs}_i / (1+\gamma)
                    & |\text{rhs}_i| \leq \delta(1+\gamma) \\
                \text{rhs}_i - \gamma\,\delta\,\operatorname{sign}(\text{rhs}_i)
                    & |\text{rhs}_i| > \delta(1+\gamma).
            \end{cases}
        $$

        The two branches agree at the kink, so the result is a
        continuous piecewise-linear function of ``rhs``.

        Parameters
        ----------
        rhs : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{rhs})$.
        """
        theta = self.delta * (1.0 + gamma)
        return np.where(
            np.abs(rhs) <= theta,
            rhs / (1.0 + gamma),
            rhs - gamma * self.delta * np.sign(rhs),
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at $x$.

        Per coordinate, $h'(x_i) = x_i$ for $|x_i| \leq \delta$ and
        $h'(x_i) = \delta\,\operatorname{sign}(x_i)$ otherwise. The two
        branches join smoothly at $|x_i| = \delta$ (both evaluate to
        $\delta\,\operatorname{sign}(x_i)$), so $f$ is $C^1$ and there
        is no subgradient ambiguity.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $\nabla f(x)$.
        """
        return np.where(np.abs(x) <= self.delta, x, self.delta * np.sign(x))
