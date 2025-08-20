import numpy as np

from algoverify.linearexpression import LinearExpression
from algoverify.steps.step import Step


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

    def __init__(
        self,
        lhs_expr,
        rhs_expr,
        lb=-np.inf,
        ub=np.inf,
        lhs_mat=None,
        lhs_mat_factorization=None,
    ):
        super().__init__(lhs_expr, rhs_expr, lb=lb, ub=ub)
        self.lhs_mat = lhs_mat
        if lhs_mat_factorization is None:
            # TODO: implement this
            raise NotImplementedError("not implemented when factorization missing")
        else:
            self.lhs_mat_factorization = lhs_mat_factorization

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
