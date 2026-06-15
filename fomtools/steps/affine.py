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

    Can also be constructed via ``weight`` and optional ``bias`` to represent
    the explicit transform $w = W \cdot \text{rhs} + b$, where ``rhs`` is the
    input expression. In this form ``lhs_mat`` is set to identity internally.

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

    def __init__(
        self,
        rhs: LinearExpression,
        lhs_mat: np.matrix = None,
        weight: np.ndarray = None,
        bias: np.ndarray = None,
        relax_binary_vars: bool = False,
    ):
        if weight is not None:
            rhs = weight @ rhs
            if bias is not None:
                rhs = rhs + bias
            lhs_mat = np.eye(weight.shape[0])
        elif lhs_mat is None:
            raise ValueError("Either lhs_mat or weight must be provided.")
        super().__init__(rhs, relax_binary_vars)
        self.lhs_mat = lhs_mat
        self.lb, self.ub = self.rhs.bound_propagation()
        # from_int defaults lb/ub to scalars; force array form so subsequent
        # steps' bound_propagation() doesn't hit sparse @ scalar NaN errors.
        ndim = self.rhs.ndim
        self.lhs.update_bounds(np.full(ndim, -np.inf), np.full(ndim, np.inf))

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update INPUT bounds (on rhs) and propagate analytically to OUTPUT bounds on lhs.
        """
        self.lb, self.ub = lb, ub
        self.sync_lhs_bounds()

    def sync_lhs_bounds(self) -> None:
        """Propagate current input bounds to lhs via lhs_mat_inverse."""
        out_lb, out_ub = self.lhs_mat_inverse.bound_propagation()
        self.lhs.update_bounds(lb=out_lb, ub=out_ub)

    @cached_property
    def lhs_mat_inverse(self) -> LinearExpression:
        r"""
        Return a `LinearExpression` formed by applying $A^{-1}$ (with $A=$ `self.lhs_mat`) to each
        RHS component $b_k$—i.e., solving $A x_k = b_k$ for all $k$.
        Supports both dense and sparse lhs_mat.
        """
        def _to_dense(b):
            return np.asarray(b.todense()) if spa.issparse(b) else np.asarray(b)

        if spa.issparse(self.lhs_mat):
            lu = spa.linalg.splu(self.lhs_mat.tocsc())

            def solve(b):
                return lu.solve(_to_dense(b))
        else:

            def solve(b):
                return np.asarray(np.linalg.solve(self.lhs_mat, _to_dense(b)))

        return LinearExpression(
            decomposition_dict={
                k: np.squeeze(solve(v))
                for k, v in self.rhs.decomposition_dict.items()
            },
            const=np.squeeze(solve(self.rhs.const)),
        )
