import numpy as np

from algoverify.linearexpression import LinearExpression
from algoverify.steps.step import Step
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Affine(Step):
    """
    An affine operator of the following form:

    .. math::
        \[
        \phi(v,x) = \{ w \in \mathbf{R}^d \mid Mw = By,\ \ y = (v,x) \}
        \]

    where :math:`$M \in \mathbf{R}^{d \times d}$` is an invertible matrix and
    :math:`$B \in \mathbf{R}^{d \times (p+d)}$`.

    Attributes:
    ----------


    """

    lhs_mat: np.matrix
    lhs_mat_factorization: np.matrix
    # TODO: the code for when lhs_mat_factorization is missing is not implemented yet, do it

    def Minv_rhs(self):
        """


        Returns
        -------
            A linear expression object that ..

        """
        return LinearExpression(
            self.lhs.get_output_dim(),
            decomposition_dict={
                k: np.squeeze(np.asarray(np.linalg.solve(self.lhs_mat, v.todense())))
                for k, v in self.rhs.decomposition_dict.items()
            },
        )
