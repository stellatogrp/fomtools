import numpy as np

from fomtools.proxable_functions import ProxableFunction


class ElasticNet(ProxableFunction):
    r"""
    Elastic-net regularizer.

    Represents the convex function

    $$
        f(x) = \alpha \|x\|_1 + \beta \|x\|_2^2,
    $$

    with non-negative scalars $\alpha, \beta$. Note that the $\ell_2$
    term has no leading $1/2$ factor: $\nabla(\beta \|x\|_2^2) = 2\beta x$.

    Notes
    -----
    The proximal operator admits the closed form

    $$
        \operatorname{prox}_{\gamma f}(v) \;=\;
            \frac{S_{\alpha\gamma}(v)}{1 + 2\beta\gamma},
    $$

    where $S_\lambda(w) = \operatorname{sign}(w)\,\max(|w| - \lambda, 0)$
    is element-wise soft thresholding. Two boundary cases recover
    familiar operators:

    - $\beta = 0$: pure $\ell_1$, identical to ``NormL1(lambd=alpha)``.
    - $\alpha = 0$: pure ridge, $v / (1 + 2\beta\gamma)$.
    """

    def __init__(self, alpha: float, beta: float):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        self.alpha = alpha
        self.beta = beta

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \alpha\|x\|_1 + \beta\|x\|_2^2$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $\alpha\|x\|_1 + \beta\|x\|_2^2$.
        """
        return float(self.alpha * np.sum(np.abs(x)) + self.beta * np.sum(x * x))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point of the elastic-net regularizer.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma f}(\text{arg}) \;=\;
                \frac{S_{\alpha\gamma}(\text{arg})}{1 + 2\beta\gamma}.
        $$

        Soft-thresholding is sign-equivariant, so the formula treats
        positive and negative entries symmetrically; the
        $1/(1 + 2\beta\gamma)$ factor is the ridge shrinkage piece.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.
        """
        thresh = self.alpha * gamma
        soft = np.sign(arg) * np.maximum(np.abs(arg) - thresh, 0.0)
        return soft / (1.0 + 2.0 * self.beta * gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at $x$.

        Returns $2\beta x + \alpha \operatorname{sign}(x)$, with
        $\operatorname{sign}(0) = 0$ selecting the canonical
        subgradient at zero for the $\ell_1$ part.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the subgradient.

        Returns
        -------
        np.ndarray
            A subgradient $g \in \partial f(x)$.
        """
        return 2.0 * self.beta * x + self.alpha * np.sign(x)
