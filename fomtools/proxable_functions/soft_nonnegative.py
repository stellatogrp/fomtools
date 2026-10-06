import numpy as np

from fomtools.proxable_functions import ProxableFunction


class SoftNonNegative(ProxableFunction):
    r"""
    One-sided quadratic penalty on negative coordinates.

    Represents

    $$
        f(x) = \frac{1 - \alpha}{2\,\alpha}\,
               \sum_i \bigl(\min(x_i,\, 0)\bigr)^2.
    $$

    Convex, smooth, and zero on the non-negative orthant. Its prox is a scaled
    LeakyReLU, and the slope $\alpha$ interpolates between two familiar cases:

    - $\alpha = 1$: $f \equiv 0$, so the prox is the identity.
    - $\alpha \to 0^+$: $f$ tends to the indicator $I_{\mathbb{R}_+^n}$, and
      the prox tends to ReLU.

    Parameters
    ----------
    alpha : float
        LeakyReLU slope $\alpha \in (0, 1]$.

    Raises
    ------
    ValueError
        If ``alpha`` is not in $(0, 1]$.

    See Also
    --------
    IndNonNegative : The limit $\alpha \to 0^+$.

    Notes
    -----
    **Proximal operator.**

    $$
        \operatorname{prox}_{\gamma f}(v)
            = \tilde\alpha(\gamma)\, v
              + \bigl(1 - \tilde\alpha(\gamma)\bigr)\operatorname{ReLU}(v),
        \qquad
        \tilde\alpha(\gamma) = \frac{\alpha}{\alpha + \gamma(1 - \alpha)},
    $$

    an element-wise LeakyReLU with effective slope $\tilde\alpha(\gamma)$; at
    $\gamma = 1$ the slope is $\alpha$.

    **Verification lowering.** ``prox_softnonnegative`` posts a single ``Relu``
    step and combines it affinely with the input.
    """

    def __init__(self, alpha: float):
        super().__init__()
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = float(alpha)

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \frac{1-\alpha}{2\alpha}\sum_i \min(x_i, 0)^2$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        coeff = (1.0 - self.alpha) / (2.0 * self.alpha)
        neg = np.maximum(0.0, -x)
        return float(coeff * np.sum(neg * neg))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Applies a LeakyReLU with effective slope
        $\tilde\alpha(\gamma) = \alpha / (\alpha + \gamma(1-\alpha))$.

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
        alpha_tilde = self.alpha / (self.alpha + gamma * (1.0 - self.alpha))
        return alpha_tilde * arg + (1.0 - alpha_tilde) * np.maximum(arg, 0.0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of $f$ at ``x``.

        $\nabla f(x)_i = \frac{1-\alpha}{\alpha}\,\min(x_i, 0)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        np.ndarray
            The gradient $\nabla f(x)$.
        """
        coeff = (1.0 - self.alpha) / self.alpha
        return coeff * np.minimum(x, 0.0)
