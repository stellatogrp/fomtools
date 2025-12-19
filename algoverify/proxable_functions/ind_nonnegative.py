import numpy as np

from algoverify.proxable_functions import ProxableFunction
from algoverify.steps import Relu


class IndNonNegative(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the non-negative orthant.

    This class represents the convex indicator function

    $$
        I_{\geq 0}(x) = \begin{cases}
            0 & \text{if } x \geq 0 \\
            +\infty & \text{otherwise}
        \end{cases}
    $$

    where the inequality $x \geq 0$ is interpreted element-wise.

    Notes
    -----
    The ``prox`` method returns the value $I_{\geq 0}(y)$ evaluated at the
    proximal point $y = \operatorname{prox}_{\gamma I_{\geq 0}}(x)$, **not**
    the proximal point itself.
    """

    def __init__(self):
        super().__init__()

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator function $I_{\geq 0}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The value $I_{\geq 0}(x)$, which is $0$ if $x \geq 0$
            (element-wise) and $+\infty$ otherwise.
        """
        if np.all(x >= 0):
            return 0.0
        else:
            return np.inf

    def prox(self, x: np.ndarray, gamma=1):
        r"""
        Compute the proximal objective value of the non-negative indicator at ``x``.

        This method computes the proximal point

        $$
            y = \operatorname{prox}_{\gamma I_{\geq 0}}(x),
        $$

        The proximal operator of $I_{\geq 0}$ with step size $\gamma$ is defined by:

        $$
            \operatorname{prox}_{\gamma I_{\geq 0}}(x) = \arg\!\min_{u \geq 0} \left\{
            I_{\geq 0}(u) + \frac{1}{2\gamma} \|u - x\|^2 \right\}
        $$

        which simplifies to the projection onto the non-negative orthant:

        $$
            \operatorname{prox}_{\gamma I_{\geq 0}}(x) = \max(x, 0)
        $$

        Note that the result is independent of $\gamma$ for indicator functions.

        Parameters
        ----------
        x : np.ndarray
            Input point at which to apply the proximal operator.

        gamma : float or array_like
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``x``. Ignored for indicator functions
            but kept for interface consistency.

        Returns
        -------
        float
            The value $I_{\geq 0}(y)$ at the proximal point
            $y = \operatorname{prox}_{\gamma I_{\geq 0}}(x)$.
            Since $y \geq 0$, this always returns $0$.
        """
        y = np.maximum(x, 0)

        return self.eval(y)

    def prox_step(self, gamma: float = 1., **kwargs):
        r"""
        Interface for constructing and executing proximal step of ``IndNonNegative`` 
        with step size $\gamma$ at ``rhs``. 

        Parameters
        ----------
        gamma : float or array_like
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``rhs``. Ignored for indicator functions
            but kept for interface consistency.

        Returns
        -------
        Relu :
            Explicit form of the proximal step in terms of ``algoverify.steps``.

        kwargs :
            The keyword arguments required to compute proximal operator of ``IndNonNegative``
            in terms of ``Relu`` step.
        """
        return Relu, kwargs
