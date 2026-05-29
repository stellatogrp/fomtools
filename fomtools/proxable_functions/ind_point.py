import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndPoint(ProxableFunction):
    r"""
    A proximal operator for the indicator function of the singleton
    $\{ p \}$.

    Represents

    $$
        I_{\{p\}}(x) = \begin{cases}
            0       & \text{if } x = p \\
            +\infty & \text{otherwise.}
        \end{cases}
    $$

    The proximal operator is the constant map $x \mapsto p$.
    ``p`` may be a scalar (broadcast against the input) or an array.
    """

    def __init__(self, p: float | np.ndarray):
        super().__init__()
        self.p = p

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate the indicator of the singleton $\{ p \}$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            ``0.0`` if $x = p$ (within floating-point tolerance),
            ``+inf`` otherwise.
        """
        if np.allclose(x, self.p):
            return 0.0
        return np.inf

    def prox(self, rhs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Return $p$, broadcast to the shape of ``rhs``.

        The prox of an indicator function is the projection onto its
        feasible set; for a singleton this is the constant map returning
        $p$, independent of $\gamma$.

        Parameters
        ----------
        rhs : np.ndarray
            Input point (used only for shape broadcasting).
        gamma : float, default 1.0
            Step size (ignored for indicator functions; kept for interface
            consistency).

        Returns
        -------
        np.ndarray
            The constant $p$ broadcast to ``rhs.shape``.
        """
        return np.broadcast_to(self.p, rhs.shape).copy()

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator functions are not differentiable.")
