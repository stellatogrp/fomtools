import numpy as np

from algoverify.linearexpression import LinearExpression
from algoverify.steps.step import Step


class SoftThreshold(Step):
    r"""
    A soft-thresholding operator for shrinkage, defined as:

    $$
        (\phi(v,x))_i=\mathcal{T}_{\lambda}(a^\top y)=
        \begin{cases}
        a^\top y-\lambda,& a^\top y>\lambda,\\[6pt]
        0,& |a^\top y|\le \lambda,\\[6pt]
        a^\top y+\lambda,& a^\top y<-\lambda,
        \end{cases}
        \quad \text{with } y=(v,x),
    $$

    where $\mathcal{T}_{\lambda}$ is the soft-thresholding operator with constant $\lambda>0$,
    and $a\in\mathbf{R}^d$.

    Attributes
    ----------
    lhs: LinearExpression
        Left-hand side linear expression in the operator definition.
    rhs: LinearExpression
        Right-hand side linear expression in the operator definition.
    lb: float
        Lower bound associated with the transformed values.
    ub: float
        Lower bound associated with the transformed values.
    relax_binary_vars: bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    lambd : float
        The threshold ($\lambda$) controlling shrinkage. Must be non-negative.
    idx_with_right_binary_vars : set[int]
        Indices tied to “right-branch” indicators in a MILP reformulation (e.g., $t>\lambda$).
    idx_with_left_binary_vars : set[int]
        Indices tied to “left-branch” indicators (e.g., $t<-\lambda$).
    """

    def __init__(self, rhs: LinearExpression, lambd: float, relax_binary_vars: bool = False):
        super().__init__(rhs, relax_binary_vars)
        self.lambd = lambd
        self.idx_with_right_binary_vars = set()
        self.idx_with_left_binary_vars = set()

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    def soft_threshold(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the soft-thresholding operator to the input array.

        Parameters
        ----------
        x: np.ndarray
            A 1-D array of values on which to apply the activation.

        Returns
        -------
        np.ndarray
            A new array representing obtained by applying the soft-thresholding
            operator element-wise. Values with magnitude less than `self.lambd`
            are mapped to zero, while others are shrunk toward zero by
            `self.lambd`.

        """
        return np.sign(x) * np.maximum(np.abs(x) - self.lambd, 0)

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update the bounds of the left-land side after applying
        soft-thresholding operator.

        Parameters
        ----------
        lb: float or np.ndarray
            The new lower bound(s) before applying soft-thresholding.

        ub: float or np.ndarray
            The new upper bound(s) before applying soft-thresholding.
        """
        self.lhs.update_bounds(lb=self.soft_threshold(lb), ub=self.soft_threshold(ub))
