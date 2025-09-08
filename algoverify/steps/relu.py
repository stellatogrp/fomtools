from dataclasses import dataclass, field

from algoverify.steps.step import Step


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
