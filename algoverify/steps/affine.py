
import numpy as np

from algoverify.linearexpression import LinearExpression
from algoverify.steps.step import Step


class Affine(Step):
    r"""
    An affine operator of the following form:

    $$
      \phi(v,x) = \{ w \in \mathbf{R}^d \mid Mw = By,\ \ y = (v,x) \},
    $$

    where $M \in \mathbf{R}^{ d \times d }$ is an invertible matrix and
    $B \in \mathbf{R}^{d \times (p+d)}$.

    Attributes
    ----------
    lhs: LinearExpression
        Left-hand side linear expression in the operator definition.
    rhs: LinearExpression
        Right-hand side linear expression in the operator definition.
    lb: float
        Lower bound associated with the transformed values.
    ub: float
        Upper bound associated with the transformed values.
    relax_binary_vars: bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    lhs_mat: np.matrix
        Coefficient matrix on the left-hand side of the operator equation.
    lhs_mat_factorization: np.matrix
        ...
    """

    # todo: complete the docstring
    # # TODO: Can't we retrive this from the decomposition dict of the lhs linearexpression?
    # TODO: the code for when lhs_mat_factorization is missing is not implemented yet, do it

    def __init__(
        self,
        rhs: LinearExpression,
        lhs_mat,
        lhs_mat_factorization: np.matrix,
        relax_binary_vars: bool = False,
    ):
        super().__init__(rhs, relax_binary_vars)
        self.lhs_mat = lhs_mat
        self.lhs_mat_factorization = lhs_mat_factorization
        self.lb, self.ub = self.lhs_mat_inverse().bound_propagation()

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update the bounds of the left-hand side with provided
        upper and lower bounds.
        """
        self.lhs.update_bounds(lb=lb, ub=ub)

    def lhs_mat_inverse(self) -> LinearExpression:
        """
        Return a `LinearExpression` formed by applying $A^{-1}$ (with $A=$ `self.lhs_mat`) to each
        RHS component $b_k$—i.e., solving $A x_k = b_k$ for all $k$."""
        return LinearExpression(
            decomposition_dict={
                k: np.squeeze(np.asarray(np.linalg.solve(self.lhs_mat, v.todense())))
                for k, v in self.rhs.decomposition_dict.items()
            },
        )
