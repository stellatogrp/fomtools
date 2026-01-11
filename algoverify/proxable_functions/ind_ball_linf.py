import numpy as np

from algoverify.proxable_functions import ProxableFunction
from algoverify.steps import SaturatedLinear


class IndBallLinf(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the L-infinity ball.

    This class represents the convex indicator function

    $$
        I_{\{x : \|x\|_\infty \leq r\}}(x) = \begin{cases}
            0 & \text{if } \|x\|_\infty \leq r \\
            +\infty & \text{otherwise}
        \end{cases}
    $$

    where $\|x\|_\infty = \max_i |x_i|$ is the infinity norm and $r > 0$ is the radius.

    Notes
    -----
    The ``prox`` method returns the value $I_{\{x : \|x\|_\infty \leq r\}}(y)$ evaluated at the
    proximal point $y = \operatorname{prox}_{\gamma I_{\{x : \|x\|_\infty \leq r\}}}(x)$, **not**
    the proximal point itself.
    """

    def __init__(self, r: float):
        super().__init__()
        if r <= 0:
            raise ValueError("Radius r must be positive")
        self.r = r

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator function $I_{\{x : \|x\|_\infty \leq r\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The value $I_{\{x : \|x\|_\infty \leq r\}}(x)$, which is $0$ if
            $\|x\|_\infty \leq r$ and $+\infty$ otherwise.
        """
        if np.max(np.abs(x)) <= self.r:
            return 0.0
        else:
            return np.inf

    def prox(self, x: np.ndarray, gamma=1):
        r"""
        Compute the proximal objective value of the L-infinity ball indicator at ``x``.

        This method computes the proximal point

        $$
            y = \operatorname{prox}_{\gamma I_{\{x : \|x\|_\infty \leq r\}}}(x),
        $$

        The proximal operator of $I_{\{x : \|x\|_\infty \leq r\}}$ with step size $\gamma$ is defined by:

        $$
            \operatorname{prox}_{\gamma I_{\{x : \|x\|_\infty \leq r\}}}(x) = \arg\!\min_{u : \|u\|_\infty \leq r} \left\{
            I_{\{x : \|x\|_\infty \leq r\}}(u) + \frac{1}{2\gamma} \|u - x\|^2 \right\}
        $$

        which simplifies to the projection onto the L-infinity ball:

        $$
            \operatorname{prox}_{\gamma I_{\{x : \|x\|_\infty \leq r\}}}(x) = \min(\max(x, -r), r)
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
            The value $I_{\{x : \|x\|_\infty \leq r\}}(y)$ at the proximal point
            $y = \operatorname{prox}_{\gamma I_{\{x : \|x\|_\infty \leq r\}}}(x)$.
            Since $\|y\|_\infty \leq r$, this always returns $0$.
        """
        y = np.clip(x, -self.r, self.r)

        return self.eval(y)

    def prox_step(self, gamma: float = 1., **kwargs):
        r"""
        Interface for constructing and executing proximal step of ``IndBallLinf``
        with step size $\gamma$ at ``rhs``. 

        Parameters
        ----------
        gamma : float or array_like
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``rhs``. Ignored for indicator functions
            but kept for interface consistency.

        Returns
        -------
        SaturatedLinear :
            Explicit form of the proximal step in terms of ``algoverify.steps``.

        kwargs :
            The keyword arguments required to compute proximal operator of ``IndBallLinf``
            in terms of ``SaturatedLinear`` step. Requires ``l`` (lower bound, set to $-r$)
            and ``u`` (upper bound, set to $r$).
        """
        kwargs['l'] = -self.r
        kwargs['u'] = self.r
        return SaturatedLinear, kwargs
