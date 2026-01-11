import numpy as np

from algoverify.proxable_functions import ProxableFunction
from algoverify.steps import Affine


class IndHalfSpace(ProxableFunction):
    r"""
    A proximal operator for the indicator function of a half-space.

    This class represents the convex indicator function

    $$
        I_{\{x : a^T x \leq b\}}(x) = \begin{cases}
            0 & \text{if } a^T x \leq b \\
            +\infty & \text{otherwise}
        \end{cases}
    $$

    where $a \in \mathbf{R}^n$ is a nonzero vector and $b \in \mathbf{R}$ is a scalar.

    Notes
    -----
    The ``prox`` method returns the value $I_{\{x : a^T x \leq b\}}(y)$ evaluated at the
    proximal point $y = \operatorname{prox}_{\gamma I_{\{x : a^T x \leq b\}}}(x)$, **not**
    the proximal point itself.
    """

    def __init__(self, a: np.ndarray, b: float, norm_a: float):
        super().__init__()
        self.a = np.asarray(a)
        if np.allclose(self.a, 0):
            raise ValueError("Vector a must be nonzero")
        self.b = float(b)
        self.norm_a = float(norm_a)
        # Validate that norm_a is approximately ||a||^2
        expected_norm_sq = np.dot(self.a, self.a)
        if not np.isclose(self.norm_a, expected_norm_sq, rtol=1e-6):
            raise ValueError(f"norm_a ({self.norm_a}) should equal ||a||^2 ({expected_norm_sq})")

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator function $I_{\{x : a^T x \leq b\}}(x)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The value $I_{\{x : a^T x \leq b\}}(x)$, which is $0$ if
            $a^T x \leq b$ and $+\infty$ otherwise.
        """
        a_dot_x = np.dot(self.a, x)
        # Use small tolerance for floating point comparison
        if a_dot_x <= self.b + 1e-10:
            return 0.0
        else:
            return np.inf

    def prox(self, x: np.ndarray, gamma=1):
        r"""
        Compute the proximal objective value of the half-space indicator at ``x``.

        This method computes the proximal point

        $$
            y = \operatorname{prox}_{\gamma I_{\{x : a^T x \leq b\}}}(x),
        $$

        The proximal operator of $I_{\{x : a^T x \leq b\}}$ with step size $\gamma$ is defined by:

        $$
            \operatorname{prox}_{\gamma I_{\{x : a^T x \leq b\}}}(x) = \arg\!\min_{u : a^T u \leq b} \left\{
            I_{\{x : a^T x \leq b\}}(u) + \frac{1}{2\gamma} \|u - x\|^2 \right\}
        $$

        which simplifies to the projection onto the half-space:

        $$
            \operatorname{prox}_{\gamma I_{\{x : a^T x \leq b\}}}(x) = \begin{cases}
                x & \text{if } a^T x \leq b \\
                x - \frac{a^T x - b}{\|a\|^2} a & \text{if } a^T x > b
            \end{cases}
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
            The value $I_{\{x : a^T x \leq b\}}(y)$ at the proximal point
            $y = \operatorname{prox}_{\gamma I_{\{x : a^T x \leq b\}}}(x)$.
            Since $a^T y \leq b$, this always returns $0$.
        """
        a_dot_x = np.dot(self.a, x)
        if a_dot_x <= self.b:
            y = x.copy()
        else:
            # Project onto boundary: y = x - ((a^T x - b) / ||a||^2) * a
            y = x - ((a_dot_x - self.b) / self.norm_a) * self.a

        return self.eval(y)

    def prox_step(self, gamma: float = 1., **kwargs):
        r"""
        Interface for constructing and executing proximal step of ``IndHalfSpace``
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
            Note: The half-space projection is conditional and may require special
            handling in the verifier.

        kwargs :
            The keyword arguments required to compute proximal operator of ``IndHalfSpace``
            in terms of ``Affine`` step. Requires ``lhs_mat`` (identity matrix),
            ``lhs_mat_factorization`` (identity matrix factorization), ``a`` (half-space
            normal vector), ``b`` (half-space offset), and ``norm_a`` (squared norm of a).
        """
        n = len(self.a)
        kwargs['lhs_mat'] = np.eye(n)
        kwargs['lhs_mat_factorization'] = np.eye(n)
        kwargs['a'] = self.a
        kwargs['b'] = self.b
        kwargs['norm_a'] = self.norm_a
        return Affine, kwargs
