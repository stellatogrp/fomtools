import numpy as np

from algoverify.proxable_functions import ProxableFunction


class NormL1(ProxableFunction):
    r"""
    A proximal operator for a positively weighted L1 norm.

    This class represents the convex function

    $$
        f(x) = \lambda \|x\|_1
             = \lambda \sum_i |x_i|
    $$

    when ``lambd`` is a scalar, or more generally

    $$
        f(x) = \sum_i \lambda_i |x_i|
    $$

    when ``lambd`` is an array of non-negative weights.

    Notes
    -----
    The ``prox`` method returns the value $f(y)$ evaluated at the
    proximal point $y = \operatorname{prox}_{\gamma f}(x)$, **not**
    the proximal point itself.
    """

    def __init__(self, lambd: float = 1):
        super().__init__()
        if np.any(lambd < 0):
            raise ValueError("Lambda must be positive")

        self.lambd = lambd

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the weighted $l_1$  norm $f(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The value $f(x) = \sum_i \lambda_i |x_i|$.
        """
        return np.sum(self.lambd * np.abs(x))

    def prox(self, x: np.ndarray, gamma=1):
        r"""
        Compute the proximal objective value of the $l_1$ norm at ``x``.

        This method computes the proximal point

        $$
            y = \operatorname{prox}_{\gamma f}(x),
        $$

        The proximal operator of $f$ with (possibly coordinate-wise)
        step size $\gamma$ is defined by:

        $$
            \operatorname{prox}_{\gamma f}(x) = \arg\!\min_{u} \left\{ f(u)
            + \frac{1}{2} \sum_i \frac{(u_i - x_i)^2}{\gamma_i} \right\}
        $$

        where: $\gamma$ can be a scalar or an array.

        Parameters
        ----------
        x : np.ndarray
            Input point at which to apply the proximal operator.

        gamma : float or array_like
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``x``.

        Returns
        -------
        float
            The value $f(y)$ at the proximal point
            $y = \operatorname{prox}_{\gamma f}(x)$.
        """
        gl = gamma * self.lambd
        y = x + np.where(x <= -gl, gl, np.where(x >= gl, -gl, -x))

        return self.eval(y)
