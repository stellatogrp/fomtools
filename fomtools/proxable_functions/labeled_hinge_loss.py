import numpy as np

from fomtools.proxable_functions import ProxableFunction


class LabeledHingeLoss(ProxableFunction):
    r"""
    Labeled hinge loss for binary classification.

    Represents the element-wise sum

    $$
        g_y(x) = \sum_i \max(0, 1 - y_i x_i),
    $$

    with labels $y \in \{-1, +1\}^n$ fixed at construction time. For
    $y = \mathbf{1}$ this reduces to the standard ``HingeLoss``;
    for $y = -\mathbf{1}$ the residual direction flips.

    Notes
    -----
    The substitution $w = y u$, $z = y v$ (using $y^2 = 1$ per
    coordinate) reduces the labeled prox to the standard hinge prox in
    $(w, z)$. Back-substituting:

    $$
        \operatorname{prox}_{\gamma g_y}(v)
            \;=\; v + y \cdot \mathrm{satlin}(1 - y v;\, 0,\, \gamma).
    $$

    Labels must be exactly $\pm 1$ — using $\{0, 1\}$-encoded labels is
    rejected at construction with a clear error.
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
        r"""Evaluate $g_y(x) = \sum_i \max(0, 1 - y_i x_i)$."""
        return float(np.sum(np.maximum(0.0, 1.0 - self.y * x)))

    def prox(self, arg: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        r"""
        Compute the proximal point.

        Uses $\operatorname{prox}_{\gamma g_y}(v)
        = v + y \cdot \mathrm{satlin}(1 - y v; 0, \gamma)$.

        Parameters
        ----------
        arg : np.ndarray
            Input point.
        gamma : float, default 1.0
            Proximal step size.

        Returns
        -------
        np.ndarray
            The proximal point.
        """
        return arg + self.y * np.minimum(np.maximum(1.0 - self.y * arg, 0.0), gamma)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r"""
        Subgradient, per coordinate:

        $$
            g_i = \begin{cases}
                -y_i    & y_i x_i < 1 \\
                0       & y_i x_i \geq 1.
            \end{cases}
        $$

        At $y_i x_i = 1$ the canonical choice $0$ is returned.
        """
        return -self.y * (self.y * x < 1).astype(float)
