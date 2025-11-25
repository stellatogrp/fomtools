from dataclasses import dataclass
from functools import cached_property

from algoverify.steps.step import Step


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

    where $b, c \in \mathbb{R}$.

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
