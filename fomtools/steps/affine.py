from functools import cached_property

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.steps.step import Step


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
    lb: np.ndarray
        Lower bounds on the INPUT (rhs), before the affine transform.
    ub: np.ndarray
        Upper bounds on the INPUT (rhs), before the affine transform.
    relax_binary_vars: bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    lhs_mat: np.matrix
        Coefficient matrix on the left-hand side of the operator equation.
    """

    skip_obbt: bool = True  # Bounds computed analytically via linear solve
    # Note: lhs_mat_inverse is a @cached_property that caches the solved
    # coefficient matrices, not the bounds. bound_propagation() reads the
    # current k.lb/k.ub from the underlying leaf variables each time. If
    # skip_obbt is ever removed, tighter input bounds from OBBT would need
    # to be propagated back to the underlying variable objects for the
    # output bound estimate to improve.

    def __init__(
        self,
        rhs: LinearExpression,
        lhs_mat: np.matrix,
        relax_binary_vars: bool = False,
    ):
        super().__init__(rhs, relax_binary_vars)
        self.lhs_mat = lhs_mat
        self.update_bounds(*self.rhs.bound_propagation())

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update INPUT bounds (on rhs) and propagate analytically to OUTPUT bounds on lhs.
        """
        self.lb, self.ub = lb, ub
        out_lb, out_ub = self.lhs_mat_inverse.bound_propagation()
        self.lhs.update_bounds(lb=out_lb, ub=out_ub)

    @cached_property
    def lhs_mat_inverse(self) -> LinearExpression:
        r"""
        Return a `LinearExpression` formed by applying $A^{-1}$ (with $A=$ `self.lhs_mat`) to each
        RHS component $b_k$—i.e., solving $A x_k = b_k$ for all $k$.
        Supports both dense and sparse lhs_mat.
        """
        if spa.issparse(self.lhs_mat):
            lu = spa.linalg.splu(self.lhs_mat.tocsc())
            solve = lambda b: lu.solve(np.asarray(b.todense()))
        else:
            solve = lambda b: np.asarray(np.linalg.solve(self.lhs_mat, b.todense()))

        return LinearExpression(
            decomposition_dict={
                k: np.squeeze(solve(v))
                for k, v in self.rhs.decomposition_dict.items()
            },
        )
