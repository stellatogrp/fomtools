import numpy as np

from fomtools.proxable_functions import ProxableFunction


class NormL1(ProxableFunction):
    r"""
    Weighted $\ell_1$ norm.

    Represents

    $$
        f(x) = \sum_i \lambda_i |x_i|,
    $$

    which is $\lambda \|x\|_1$ when ``lambd`` is a scalar. Convex and piecewise
    linear.

    Parameters
    ----------
    lambd : float or np.ndarray, default 1.0
        Non-negative weight: a scalar, or a 1-D array of per-coordinate weights.

    Raises
    ------
    ValueError
        If any weight is negative, or ``lambd`` has more than one dimension.

    See Also
    --------
    ElasticNet : Adds a squared $\ell_2$ term.
    NormInf : The $\ell_\infty$ norm.

    Notes
    -----
    **Proximal operator.** Element-wise soft thresholding,

    $$
        \operatorname{prox}_{\gamma f}(v)_i
        = \operatorname{sign}(v_i)\,\max(|v_i| - \gamma\lambda_i,\, 0).
    $$

    **Verification lowering.** ``prox_norml1`` posts a single ``SoftThreshold``
    step with threshold $\gamma\lambda$.
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
        Evaluate $f(x) = \sum_i \lambda_i |x_i|$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.

        Raises
        ------
        ValueError
            If ``x`` is not 1-D, or its length does not match an array ``lambd``.
        """
        if x.ndim >= 2:
            raise ValueError("Input should be one-dimensional")
        if self.lambd.ndim == 1 and x.shape[0] != self.lambd.shape[0]:
            raise ValueError("Input should be of the same size as Lambda")
        return np.sum(np.multiply(self.lambd, np.abs(x)))

    def prox(self, arg: np.ndarray, gamma: float | np.ndarray = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Applies element-wise soft thresholding by $\gamma \lambda$. With an
        array ``gamma``, this is the prox in the scaled metric

        $$
            \arg\!\min_{u} \left\{ f(u)
            + \frac{1}{2} \sum_i \frac{(u_i - \text{arg}_i)^2}{\gamma_i} \right\}.
        $$

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float or np.ndarray, default 1.0
            Step size $\gamma > 0$: a scalar, or an array broadcastable to the
            shape of ``arg``.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Raises
        ------
        ValueError
            If ``arg`` is not 1-D, or its shape does not match an array ``lambd``.
        """
        if arg.ndim >= 2:
            raise ValueError("Input should be one-dimensional")
        if self.lambd.ndim == 1 and arg.shape != self.lambd.shape:
            raise ValueError("Input should be of the same size as Lambda")
        gl = gamma * self.lambd
        return arg + np.where(arg <= -gl, gl, np.where(arg >= gl, -gl, -arg))

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at ``x``.

        Per coordinate,

        $$
            g_i = \begin{cases}
                1   & x_i > 0 \\
                0   & x_i = 0 \\
                -1  & x_i < 0,
            \end{cases}
        $$

        where $0$ is the canonical choice at the kink $x_i = 0$.

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
