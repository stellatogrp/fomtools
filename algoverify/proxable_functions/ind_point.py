import numpy as np

from algoverify.proxable_functions import ProxableFunction
from algoverify.steps import Affine


class IndPoint(ProxableFunction):
    r"""
    A proximal operator for the indicator function of a single point.

    $$
        I_{\{p\}}(x) = \begin{cases}
            0 & \text{if } x = p \\
            +\infty & \text{otherwise}
        \end{cases}
    $$

    """

    def __init__(self, p: float | np.ndarray):
        super().__init__()
        self.p = np.asarray(p)

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator function $I_{\{p\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The value $I_{\{p\}}(x)$, which is $0$ if $x = p$
            (within numerical tolerance) and $+\infty$ otherwise.
        """
        if np.allclose(x, self.p):
            return 0.0
        else:
            return np.inf

    def prox(self, x: np.ndarray, gamma=1):
        r"""
        Compute the proximal objective value of the point indicator at ``x``.

        This method computes the proximal point

        $$
            y = \operatorname{prox}_{\gamma I_{\{p\}}}(x),
        $$

        The proximal operator of $I_{\{p\}}$ with step size $\gamma$ is defined by:

        $$
            \operatorname{prox}_{\gamma I_{\{p\}}}(x) = \arg\!\min_{u} \left\{
            I_{\{p\}}(u) + \frac{1}{2\gamma} \|u - x\|^2 \right\}
        $$

        which simplifies to the projection onto the point:

        $$
            \operatorname{prox}_{\gamma I_{\{p\}}}(x) = p
        $$

        Note that the result is independent of $\gamma$ and $x$ for indicator functions
        of a single point.

        Parameters
        ----------
        x : np.ndarray
            Input point at which to apply the proximal operator.
            Ignored since the projection always returns $p$.

        gamma : float or array_like
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``x``. Ignored for indicator functions
            but kept for interface consistency.

        Returns
        -------
        float
            The value $I_{\{p\}}(y)$ at the proximal point
            $y = \operatorname{prox}_{\gamma I_{\{p\}}}(x)$.
            Since $y = p$, this always returns $0$.
        """
        y = self.p.copy()

        return self.eval(y)

    def prox_step(self, gamma: float = 1., **kwargs):
        r"""
        Interface for constructing and executing proximal step of ``IndPoint``
        with step size $\gamma$ at ``rhs``. 

        Parameters
        ----------
        gamma : float or array_like
            Step size(s) $\gamma$. Can be a scalar or an array
            broadcastable to the shape of ``rhs``. Ignored for indicator functions
            but kept for interface consistency.

        Returns
        -------
        Affine :
            Explicit form of the proximal step in terms of ``algoverify.steps``.

        kwargs :
            The keyword arguments required to compute proximal operator of ``IndPoint``
            in terms of ``Affine`` step. Requires ``lhs_mat`` (identity matrix) and
            ``lhs_mat_factorization`` (identity matrix factorization), with the point
            $p$ passed as a constant term that will be handled by the verifier.
        """
        n = len(self.p)
        # For projection onto a point p, we use Affine step with M = I and By = p
        # The verifier will need to construct the appropriate LinearExpression
        kwargs['lhs_mat'] = np.eye(n)
        kwargs['lhs_mat_factorization'] = np.eye(n)
        kwargs['p'] = self.p  # Pass the point as a parameter
        return Affine, kwargs
