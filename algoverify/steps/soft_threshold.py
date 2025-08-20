from algoverify.steps.step import Step


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

    def __init__(self, lhs, rhs, lam, relax_binary_vars=False):
        """
        Initialize a `SoftThreshold` object.

        Parameters
        ----------
        lhs:
            ...

        rhs:
            ...

        lam: float
            The threshold (λ) to decide how much to shrink values toward zero. Must be non-negative.

        relax_binary_vars: bool
            ...
        """
        super().__init__(lhs, rhs, relax_binary_vars=relax_binary_vars)
        self.lambd = lam
        self.idx_with_right_binary_vars = set([])
        self.idx_with_left_binary_vars = set([])
