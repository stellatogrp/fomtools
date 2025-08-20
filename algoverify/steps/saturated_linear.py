from algoverify.steps.step import Step


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

    def __init__(self, lhs_expr, rhs_expr, l, u, relax_binary_vars=False, equality_ranges=None):
        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars)
        self.l = l
        self.u = u

        self.eq_indices = self.process_equality_ranges(equality_ranges)
        self.eq_indices = self.process_inequality_ranges()

    def process_equality_ranges(self, equality_ranges):
        if equality_ranges is not None:
            return {i for (min, max) in equality_ranges for i in range(min, max)}
        return set()

    def process_inequality_ranges(self):
        return {i for i in range(self.rhs.get_output_dim()) if i not in self.eq_indices}
