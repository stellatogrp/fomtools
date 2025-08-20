from algoverify.steps.step import Step
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True, kw_only=True)
class SaturatedLinear(Step):
    """
    A Saturated Linear operator, defined as:

    .. math::
        \[
        (\phi(v,x))_i = \mathcal{S}_{[b,c]}(a^T y) =
        \begin{cases}
        b & a^T y < b, \\[6pt]
        a^T y & b \leq a^T y \leq c, \\[6pt]
        c & a^T y > c,
        \end{cases}
        \quad \text{with } y = (v,x).
        \]

    with two scalars :math:`$b, c \in \mathbf{R}$`.

    Attributes:
    -----------

    """

    l: int
    u: int
    equality_ranges: list = None

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    @cached_property
    def eq_indices(self):
        if self.equality_ranges is not None:
            return {i for (min, max) in self.equality_ranges for i in range(min, max)}
        return set()

    @cached_property
    def ineq_indices(self):
        return {i for i in range(self.rhs.get_output_dim()) if i not in self.eq_indices}
