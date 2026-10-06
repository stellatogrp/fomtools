import numpy as np

from fomtools.proxable_functions import ProxableFunction


class HuberLoss(ProxableFunction):
    r"""
    Element-wise Huber loss.

    Represents

    $$
        f(x) = \sum_i h(x_i),
        \qquad
        h(t) = \begin{cases}
            \tfrac{1}{2} t^2                        & |t| \leq \delta \\
            \delta |t| - \tfrac{1}{2}\delta^2        & |t| > \delta.
        \end{cases}
    $$

    Convex and $C^1$. This is the standard element-wise robust loss, *not* the
    multivariate norm-based "vector Huber" function.

    Parameters
    ----------
    delta : float
        Threshold $\delta \ge 0$ between the quadratic and linear regimes.

    Raises
    ------
    ValueError
        If ``delta`` is negative.

    Notes
    -----
    **Proximal operator.** Element-wise separable; per coordinate,

    $$
        \operatorname{prox}_{\gamma f}(v)_i = \begin{cases}
            \dfrac{v_i}{1+\gamma}
                & |v_i| \leq \delta(1+\gamma), \\[6pt]
            v_i - \gamma\,\delta\,\operatorname{sign}(v_i)
                & |v_i| > \delta(1+\gamma).
        \end{cases}
    $$

    The branches join continuously at $|v_i| = \delta(1+\gamma)$. Equivalently,
    with element-wise soft thresholding $S$,

    $$
        \operatorname{prox}_{\gamma f}(v)
            = \frac{v + \gamma\, S_{\delta(1+\gamma)}(v)}{1+\gamma}.
    $$

    **Verification lowering.** ``prox_huberloss`` uses the soft-threshold form:
    a single ``SoftThreshold`` step with threshold $\delta(1+\gamma)$ followed
    by affine arithmetic.
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
            The function value $f(x)$.
        """
        ax = np.abs(x)
        per_coord = np.where(
            ax <= self.delta,
            0.5 * x * x,
            self.delta * ax - 0.5 * self.delta**2,
        )
        return float(np.sum(per_coord))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form, per coordinate,

        $$
            \operatorname{prox}_{\gamma f}(\text{arg})_i = \begin{cases}
                \text{arg}_i / (1+\gamma)
                    & |\text{arg}_i| \leq \delta(1+\gamma) \\
                \text{arg}_i - \gamma\,\delta\,\operatorname{sign}(\text{arg}_i)
                    & |\text{arg}_i| > \delta(1+\gamma).
            \end{cases}
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
        theta = self.delta * (1.0 + gamma)
        return np.where(
            np.abs(arg) <= theta,
            arg / (1.0 + gamma),
            arg - gamma * self.delta * np.sign(arg),
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at ``x``.

        Per coordinate,

        $$
            \nabla f(x)_i = h'(x_i) = \begin{cases}
                x_i                                     & |x_i| \leq \delta \\
                \delta\,\operatorname{sign}(x_i)         & |x_i| > \delta.
            \end{cases}
        $$

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
