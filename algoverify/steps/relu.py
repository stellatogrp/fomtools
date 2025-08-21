from algoverify.steps.step import Step
from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True, kw_only=True)
class Relu(Step):
    """
    Rectified Linear Unit (ReLU) step operator, defined as:

    .. math:: \[(\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.\]

    It sets all the negative values to zero while keeping the positive ones unchanged.

    Attributes:
    ----------
    proj_ranges is a list of index intervals that specify which parts of the output should be projected in the ReLU step

    """

    proj_ranges: list
    idx_with_binary_vars: set[int] = field(default_factory=set)

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    @cached_property
    def proj_indices(self):
        return {i for (min, max) in self.proj_ranges for i in range(min, max)}

    @cached_property
    def nonproj_indices(self):
        """Partitions indices into projected vs non-projected sets, based on proj_ranges."""
        return {i for i in range(self.rhs.get_output_dim()) if i not in self.proj_indices}
