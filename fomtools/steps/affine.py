from functools import cached_property

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.steps.step import Step
from fomtools.utils.math_utils import repair_crossed_bounds
from fomtools.utils.matrix_utils import is_identity_matrix


class Affine(Step):
    r"""
    An affine operator of the following form:

    $$
      \phi(v,x) = \{ w \in \mathbf{R}^d \mid Mw = By,\ \ y = (v,x) \},
    $$

    where $M \in \mathbf{R}^{ d \times d }$ is an invertible matrix and
    $B \in \mathbf{R}^{d \times (p+d)}$.

    Can also be constructed via ``weight`` and optional ``bias`` to represent
    the explicit transform $w = W \cdot \text{arg} + b$, where ``arg`` is the
    input expression. In this form ``output_mat`` is set to identity internally.

    Attributes
    ----------
    output: LinearExpression
        Left-hand side linear expression in the operator definition.
    arg: LinearExpression
        Right-hand side linear expression in the operator definition.
    lb: np.ndarray
        Lower bounds on the INPUT (arg), before the affine transform.
    ub: np.ndarray
        Upper bounds on the INPUT (arg), before the affine transform.
    relax_binary_vars: bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    output_mat: np.matrix
        Coefficient matrix on the left-hand side of the operator equation.
    obbt_output: bool
        If true, OBBT is also run on the output leaf after the affine equality
        is added to the OBBT LP (see ``StepDecomposer._process_primitive``).
    """

    obbt_output: bool = True

    def __init__(
        self,
        arg: LinearExpression,
        output_mat: np.matrix = None,
        weight: np.ndarray = None,
        bias: np.ndarray = None,
        relax_binary_vars: bool = False,
    ):
        if weight is not None:
            arg = weight @ arg
            if bias is not None:
                arg = arg + bias
            output_mat = np.eye(weight.shape[0])
        elif output_mat is None:
            raise ValueError("Either output_mat or weight must be provided.")
        super().__init__(arg, relax_binary_vars)
        self.output_mat = output_mat
        self.lb, self.ub = self.arg.bound_propagation()
        # from_int defaults lb/ub to scalars; force array form so subsequent
        # steps' bound_propagation() doesn't hit sparse @ scalar NaN errors.
        ndim = self.arg.ndim
        self.output.update_bounds(np.full(ndim, -np.inf), np.full(ndim, np.inf))

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update INPUT bounds (on arg) and propagate analytically to OUTPUT bounds on output.
        """
        self.lb, self.ub = lb, ub
        self.sync_output_bounds()

    def sync_output_bounds(self) -> None:
        """
        Tighten output bounds from the current input bounds.

        Interval-propagates ``output_mat_inverse`` (built by linear solves, never an
        explicit inverse). When ``output_mat`` is the identity, ``output == arg``, so
        the (possibly OBBT-tightened) input box ``self.lb``/``self.ub`` also bounds the
        output directly. ``output.update_bounds`` intersects with the existing output
        bounds, so tightening applied earlier is never discarded. For general
        ``output_mat`` the tightened input box is exploited by OBBT on the output leaf
        (see ``StepDecomposer._process_primitive``).
        """
        out_lb, out_ub = self.output_mat_inverse.bound_propagation()
        if is_identity_matrix(self.output_mat):
            out_lb = np.maximum(out_lb, self.lb)
            out_ub = np.minimum(out_ub, self.ub)
        self.output.update_bounds(lb=out_lb, ub=out_ub)
        self.output.lb, self.output.ub = repair_crossed_bounds(self.output.lb, self.output.ub)

    @cached_property
    def output_mat_inverse(self) -> LinearExpression:
        r"""
        Return a `LinearExpression` formed by applying $A^{-1}$ (with $A=$ `self.output_mat`) to each
        RHS component $b_k$—i.e., solving $A x_k = b_k$ for all $k$.
        Supports both dense and sparse output_mat.
        """

        def _to_dense(b):
            return np.asarray(b.todense()) if spa.issparse(b) else np.asarray(b)

        if spa.issparse(self.output_mat):
            lu = spa.linalg.splu(self.output_mat.tocsc())

            def solve(b):
                return lu.solve(_to_dense(b))
        else:

            def solve(b):
                return np.asarray(np.linalg.solve(self.output_mat, _to_dense(b)))

        return LinearExpression(
            decomposition_dict={
                k: np.squeeze(solve(v)) for k, v in self.arg.decomposition_dict.items()
            },
            const=np.squeeze(solve(self.arg.const)),
        )
