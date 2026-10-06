import numpy as np

from fomtools.proxable_functions import ProxableFunction


class LabeledHingeLoss(ProxableFunction):
    r"""
    Labeled hinge loss for binary classification.

    Represents

    $$
        f(x) = \sum_i \max(0, 1 - y_i x_i),
    $$

    with labels $y \in \{-1, +1\}^n$ fixed at construction. Convex and
    piecewise linear; for $y = \mathbf{1}$ it reduces to ``HingeLoss``.

    Parameters
    ----------
    y : np.ndarray
        1-D array of labels, each exactly $\pm 1$ ($\{0, 1\}$-encoded labels
        are rejected).

    Raises
    ------
    ValueError
        If ``y`` is not 1-D or has an entry other than $\pm 1$.

    See Also
    --------
    HingeLoss : Unlabeled case $y = \mathbf{1}$.

    Notes
    -----
    **Proximal operator.** The substitution $w = y u$, $z = y v$ (using
    $y_i^2 = 1$) reduces the labeled prox to the unlabeled one; back-substituting,

    $$
        \operatorname{prox}_{\gamma f}(v)
            = v + y \odot \mathrm{satlin}(1 - y \odot v;\, 0,\, \gamma).
    $$

    **Verification lowering.** ``prox_labeledhingeloss`` posts a single
    ``SaturatedLinear`` clip of $1 - y \odot v$ to $[0, \gamma]$; the
    element-wise scaling by $y$ is a sparse diagonal matrix.
    """

    def __init__(self, y: np.ndarray):
        super().__init__()
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}")
        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("y entries must be ±1")
        self.y = y

    def eval(self, x: np.ndarray) -> float:
        r"""
        Evaluate $f(x) = \sum_i \max(0, 1 - y_i x_i)$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            The function value $f(x)$.
        """
        return float(np.sum(np.maximum(0.0, 1.0 - self.y * x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute $\operatorname{prox}_{\gamma f}(\text{arg})$.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma f}(\text{arg})
            = \text{arg} + y \odot \mathrm{satlin}(1 - y \odot \text{arg};\, 0,\, \gamma).
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
        return arg + self.y * np.minimum(np.maximum(1.0 - self.y * arg, 0.0), gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute a subgradient of $f$ at ``x``.

        Per coordinate,

        $$
            g_i = \begin{cases}
                -y_i    & y_i x_i < 1 \\
                0       & y_i x_i \geq 1,
            \end{cases}
        $$

        where $0$ is the canonical choice at the kink $y_i x_i = 1$.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the subgradient.

        Returns
        -------
        np.ndarray
            A subgradient $g \in \partial f(x)$.
        """
        return -self.y * (self.y * x < 1).astype(float)
