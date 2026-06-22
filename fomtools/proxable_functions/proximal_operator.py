import numpy as np


class ProxableFunction:
    r"""
    Abstract base class for proxable functions.

    A function $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is proxable
    if its proximal operator

    $$
        \operatorname{prox}_{\gamma f}(x) = \arg\!\min_{u} \left\{ f(u)
        + \frac{1}{2\gamma} \|u - x\|^2 \right\}
    $$

    can be evaluated efficiently (in closed form or via a structured subproblem).

    Subclasses provide the numerical interface (``eval``, ``prox``, ``grad``).
    The mapping from a proxable function to one or more steps lives in the
    step decomposer's ``_PROX_DECOMPOSERS`` registry — it is *not* a method
    on this class.
    """

    def __init__(self):
        pass

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x)$.

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
        NotImplementedError
            Raised by the base class; must be implemented by subclasses.
        """
        raise NotImplementedError

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point of the function.

        Solves

        $$
            \operatorname{prox}_{\gamma f}(\text{arg}) = \arg\!\min_{u} \left\{
            f(u) + \frac{1}{2\gamma} \|u - \text{arg}\|^2 \right\}.
        $$

        Parameters
        ----------
        arg : np.ndarray
            Input point at which to evaluate the proximal operator.
        gamma : float, default 1.0
            Step size $\gamma > 0$.

        Returns
        -------
        np.ndarray
            The proximal point $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Raises
        ------
        NotImplementedError
            Raised by the base class; must be implemented by subclasses.
        """
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient (or a subgradient) of the function at ``x``.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the (sub)gradient.

        Returns
        -------
        np.ndarray
            Gradient $\nabla f(x)$ or a subgradient when $f$ is non-smooth.

        Raises
        ------
        NotImplementedError
            Raised by the base class; must be implemented by subclasses.
        """
        raise NotImplementedError
