from algoverify.steps.step import Step
import numpy as np


class Relu(Step):
    """
    Rectified Linear Unit (ReLU) step operator, defined as:

    .. math:: \[(\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.\]

    It sets all the negative values to zero while keeping the positive ones unchanged.

    Attributes:
    ----------
    proj_ranges is a list of index intervals that specify which parts of the output should be projected in the ReLU step

    """

    def __init__(
        self, lhs_expr, rhs_expr, lb=-np.inf, ub=np.inf, proj_ranges=None, relax_binary_vars=False
    ):
        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars, lb=lb, ub=ub)
        self.proj_indices = self.process_proj_ranges(proj_ranges)
        self.nonproj_indices = self._process_proj_ranges()
        self.idx_with_binary_vars = set([])

    def process_proj_ranges(self, proj_ranges):
        return {i for (min, max) in proj_ranges for i in range(min, max)}

    def _process_proj_ranges(self):
        """Partitions indices into projected vs non-projected sets, based on proj_ranges."""
        return {i for i in range(self.rhs.get_output_dim()) if i not in self.proj_indices}
