import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SoftNonnegativePenalty(ProxableFunction):
    r"""
    One-sided convex quadratic penalty whose proximal operator is
    (a scaled) LeakyReLU.

    For a slope $\alpha \in (0, 1]$, the function is

    $$
        f(x) = \frac{1 - \alpha}{2\,\alpha}\,
               \sum_i \bigl(\min(x_i,\, 0)\bigr)^2
             = \frac{1 - \alpha}{2\,\alpha}\,
               \sum_i \bigl(\operatorname{ReLU}(-x_i)\bigr)^2,
    $$

    convex, smooth, zero on the non-negative orthant. The bracketing
    cases match familiar operators:

    - $\alpha = 1$: $f \equiv 0$, so $\operatorname{prox}_{\gamma f}$ is
      the identity.
    - $\alpha \to 0^+$: the penalty on the negative orthant grows
      unboundedly and $f$ becomes the indicator $I_{\mathbb{R}_+^n}(x)$;
      the prox recovers the standard
      :class:`fomtools.steps.relu.Relu` activation.

    Notes
    -----
    For a step size $\gamma > 0$, the proximal operator is

    $$
        \operatorname{prox}_{\gamma f}(v)
            = \tilde\alpha(\gamma)\, v
              + \bigl(1 - \tilde\alpha(\gamma)\bigr)\operatorname{ReLU}(v),
        \qquad
        \tilde\alpha(\gamma) = \frac{\alpha}{\alpha + \gamma(1 - \alpha)},
    $$

    i.e. an element-wise LeakyReLU with effective slope
    $\tilde\alpha(\gamma)$. At $\gamma = 1$ this reduces to the standard
    LeakyReLU with slope $\alpha$.
    """

    def __init__(self, alpha: float):
        super().__init__()
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = float(alpha)

    def eval(self, x: np.ndarray) -> float:
        r"""Evaluate $\frac{1-\alpha}{2\alpha}\sum_i \max(0, -x_i)^2$."""
        coeff = (1.0 - self.alpha) / (2.0 * self.alpha)
        neg = np.maximum(0.0, -x)
        return float(coeff * np.sum(neg * neg))

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point — a LeakyReLU with effective slope
        $\tilde\alpha(\gamma) = \alpha / (\alpha + \gamma(1-\alpha))$.

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
        alpha_tilde = self.alpha / (self.alpha + gamma * (1.0 - self.alpha))
        return alpha_tilde * rhs + (1.0 - alpha_tilde) * np.maximum(rhs, 0.0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Gradient (smooth function):
        $\nabla f(x)_i = \frac{1-\alpha}{\alpha}\,\min(x_i, 0)$.
        """
        coeff = (1.0 - self.alpha) / self.alpha
        return coeff * np.minimum(x, 0.0)
