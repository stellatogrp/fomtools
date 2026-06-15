import numpy as np

from fomtools.proxable_functions import ProxableFunction


class NormL1(ProxableFunction):
    r"""
    A proximal operator for a positively weighted $\ell_1$ norm.

    This class represents the convex function

    $$
        f(x) = \lambda \|x\|_1
             = \lambda \sum_i |x_i|,
    $$

    when ``lambd`` is a scalar, or more generally

    $$
        f(x) = \sum_i \lambda_i |x_i|,
    $$

    when ``lambd`` is an array of non-negative weights.
    """

    def __init__(self, lambd: float | np.ndarray = 1.0):
        super().__init__()
        if np.any(lambd < 0):
            raise ValueError("Lambda must be positive")
        if isinstance(lambd, np.ndarray) and lambd.ndim > 1:
            raise ValueError("Lambda cannot have dimension larger than 1")
        self.lambd = np.array(lambd)

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the weighted $\ell_1$ norm $f(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The value $f(x) = \sum_i \lambda_i |x_i|$.
        """
        if x.ndim >= 2:
            raise ValueError("Input should be one-dimensional")
        if self.lambd.ndim == 1 and x.shape[0] != self.lambd.shape[0]:
            raise ValueError("Input should be of the same size as Lambda")
        return np.sum(np.multiply(self.lambd, np.abs(x)))

    def prox(
        self, rhs: np.ndarray, gamma: float | np.ndarray = 1.0
    ) -> np.ndarray:
        r"""
        Compute the proximal point of the $\ell_1$ norm at ``rhs``.

        This method computes

        $$
            y = \operatorname{prox}_{\gamma f}(\text{rhs})
              = \arg\!\min_{u} \left\{ f(u)
                + \frac{1}{2} \sum_i \frac{(u_i - \text{rhs}_i)^2}{\gamma_i}
                \right\},
        $$

        where $\gamma$ can be a scalar or a 1-D array broadcastable to
        the shape of ``rhs``. The closed form is element-wise soft
        thresholding by $\gamma \lambda$.

        Parameters
        ----------
        rhs : np.ndarray
            Input point at which to apply the proximal operator.
        gamma : float or array_like, default 1.0
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``rhs``.

        Returns
        -------
        np.ndarray
            The proximal point
            $y = \operatorname{prox}_{\gamma f}(\text{rhs})$.
        """
        if rhs.ndim >= 2:
            raise ValueError("Input should be one-dimensional")
        if self.lambd.ndim == 1 and rhs.shape != self.lambd.shape:
            raise ValueError("Input should be of the same size as Lambda")
        gl = gamma * self.lambd
        return rhs + np.where(rhs <= -gl, gl, np.where(rhs >= gl, -gl, -rhs))

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of the weighted $\ell_1$ norm at ``x``.

        Returns the sign of ``x`` element-wise; at ``x_i = 0`` returns
        ``0`` (an element of the subdifferential).

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the subgradient.

        Returns
        -------
        np.ndarray
            A subgradient $g \in \partial f(x)$.
        """
        return np.sign(x)
