from algoverify.steps.step import Step
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class SoftThreshold(Step):
    """
    A soft-thresholding operator for shrinkage, defined as:

    .. math::
        \[
        (\phi(v,x))_i = \mathcal{T}_{\lambda}(a^T y) =
        \begin{cases}
        a^T y - \lambda & a^T y > \lambda, \\[6pt]
        0 & |a^T y| \leq \lambda, \\[6pt]
        a^T y + \lambda & a^T y < -\lambda,
        \end{cases}
        \quad \text{with } y = (v,x),
        \]

    where :math:`$\mathcal{T}_{\lambda}$` is the :math:`\textit{soft-thresholding}`
        operator with constant :math:`$\lambda > 0$`, and :math:`$a \in \mathbf{R}^d$`.

    Attributes:
    ----------
    lam: float
        The threshold (λ) to decide how much to shrink values toward zero. Must be non-negative.

    idx_with_right_binary_vars: set[int]
        ...

    idx_with_left_binary_vars: set[int]
        ...
    """

    lambd: float
    idx_with_right_binary_vars: set[int] = field(default_factory=set)
    idx_with_left_binary_vars: set[int] = field(default_factory=set)

    def __hash__(self):
        return hash((self.rhs, self.lhs))
