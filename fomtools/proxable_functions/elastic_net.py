import numpy as np

from fomtools.proxable_functions import ProxableFunction


class ElasticNet(ProxableFunction):
    r"""
    Elastic-net regularizer.

    Represents

    $$
        f(x) = \alpha \|x\|_1 + \beta \|x\|_2^2.
    $$

    Convex; strongly convex when $\beta > 0$. Note that the $\ell_2$ term has
    no leading $1/2$ factor: $\nabla(\beta \|x\|_2^2) = 2\beta x$.

    Parameters
    ----------
    alpha : float
        Weight $\alpha \ge 0$ of the $\ell_1$ term.
    beta : float
        Weight $\beta \ge 0$ of the squared $\ell_2$ term.

    Raises
    ------
    ValueError
        If ``alpha`` or ``beta`` is negative.

    See Also
    --------
    NormL1 : The case $\beta = 0$, identical to ``NormL1(lambd=alpha)``.

    Notes
    -----
    **Proximal operator.**

    $$
        \operatorname{prox}_{\gamma f}(v) =
            \frac{S_{\alpha\gamma}(v)}{1 + 2\beta\gamma},
    $$

    where $S_\lambda(w) = \operatorname{sign}(w)\,\max(|w| - \lambda, 0)$ is
    element-wise soft thresholding. For $\alpha = 0$ it reduces to the ridge
    shrinkage $v / (1 + 2\beta\gamma)$.

    **Verification lowering.** ``prox_elasticnet`` posts a single
    ``SoftThreshold`` step with threshold $\alpha\gamma$ and scales its output
    by $1 / (1 + 2\beta\gamma)$.
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
            The function value $f(x)$.
        """
        return float(self.alpha * np.sum(np.abs(x)) + self.beta * np.sum(x * x))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma f}(\text{arg}) =
                \frac{S_{\alpha\gamma}(\text{arg})}{1 + 2\beta\gamma}.
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
        thresh = self.alpha * gamma
        soft = np.sign(arg) * np.maximum(np.abs(arg) - thresh, 0.0)
        return soft / (1.0 + 2.0 * self.beta * gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at ``x``.

        Per coordinate,

        $$
            g_i = 2\beta x_i + \alpha \begin{cases}
                1   & x_i > 0 \\
                0   & x_i = 0 \\
                -1  & x_i < 0,
            \end{cases}
        $$

        where $0 \in [-1, 1]$ is the canonical choice at the kink $x_i = 0$.

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
