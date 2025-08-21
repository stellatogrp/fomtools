from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from algoverify.linearexpression import LinearExpression


@dataclass(frozen=True)
class Step:
    """
    A base class for thresholding and activation steps. This class
    defines a common interface for step-like operators that transform
    inputs element wise.

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
    """

    lhs: LinearExpression
    rhs: LinearExpression
    lb: float = -np.inf
    ub: float = np.inf
    relax_binary_vars: bool = False

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.rhs, self.lhs))


@dataclass(frozen=True, kw_only=True)
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
    # TODO: Can't we retrive this from the decomposition dict of the lhs linearexpression?
    lhs_mat: np.matrix
    lhs_mat_factorization: np.matrix
    # TODO: the code for when lhs_mat_factorization is missing is not implemented yet, do it

    def lhs_mat_inverse(self):
        """
        Return a `LinearExpression` formed by applying $A^{-1}$ (with $A=$ `self.lhs_mat`) to each
        RHS component $b_k$—i.e., solving $A x_k = b_k$ for all $k$."""
        return LinearExpression(
            self.lhs.get_output_dim(),
            decomposition_dict={
                k: np.squeeze(np.asarray(np.linalg.solve(self.lhs_mat, v.todense())))
                for k, v in self.rhs.decomposition_dict.items()
            },
        )


@dataclass(frozen=True, kw_only=True)
class Relu(Step):
    r"""
    Rectified Linear Unit (ReLU) step operator, defined as:

    $$
        (\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.
    $$

    It sets all the negative values to zero while keeping the positive ones unchanged.

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
    proj_ranges: list
        List of index intervals that specify which parts of the output should be
        projected in the ReLU step
    idx_with_binary_vars: set[int]
        Indices corresponding to binary decision variables.
    proj_indices: set[int]
        Indices of output coordinates subjected to ReLU projection.
    nonproj_indices: set[int]
        Set of output indices not subjected to ReLU.
    """

    proj_ranges: list
    idx_with_binary_vars: set[int] = field(default_factory=set)

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    @property
    def proj_indices(self):
        """Returns indices of output coordinates that are subjected to ReLU projections."""
        return {i for (min, max) in self.proj_ranges for i in range(min, max)}

    @property
    def nonproj_indices(self):
        """Returns indices of output coordinates that bypass the ReLU projections."""
        return {i for i in range(self.rhs.get_output_dim()) if i not in self.proj_indices}


@dataclass(frozen=True, kw_only=True)
class SaturatedLinear(Step):
    r"""
    Saturated Linear Unit step operator, defined as:

    $$
        (\phi(v, x))_i = S_{[b, c]}(a^T y) =
        \begin{cases}
            b & a^T y < b, \\\\
            a^T y & b \leq a^T y \leq c, \\\\
            c & a^T y > c,
        \end{cases}
        \quad \text{with } y = (v, x).
    $$

    where $`b, c \in \mathbb{R}`$.

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
    l: int
        ...
    u: int
        ...
    equality_ranges: list
        List of index intervals that specify which parts of the output receive
        the SatLU projection.
    eq_indices: set[int]
        Indices of output coordinates subjected to Saturated Linear Unit projection.
    ineq_indices: set[int]
        Indices of output coordinates not subjected to Saturated Linear Unit projection.
    """

    # todo: complete the docstring
    l: int
    u: int
    equality_ranges: list = None

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    @cached_property
    def eq_indices(self):
        """Returns indices of output coordinates that are subjected to Saturated Linear Unit projections."""
        if self.equality_ranges is not None:
            return {i for (min, max) in self.equality_ranges for i in range(min, max)}
        return set()

    @cached_property
    def ineq_indices(self):
        """Returns indices of output coordinates that bypass the Saturated Linear Unit projections."""
        return {i for i in range(self.rhs.get_output_dim()) if i not in self.eq_indices}


@dataclass(frozen=True, kw_only=True)
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
        \quad \text{with } y=(v,x).
    $$

    where, $\mathcal{T}_{\lambda}$ is the soft-thresholding operator with constant $\lambda>0$,
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

    lambd: float
    idx_with_right_binary_vars: set[int] = field(default_factory=set)
    idx_with_left_binary_vars: set[int] = field(default_factory=set)

    def __hash__(self):
        return hash((self.rhs, self.lhs))
