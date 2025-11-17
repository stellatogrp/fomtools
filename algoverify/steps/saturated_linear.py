import numpy as np

from algoverify import LinearExpression
from algoverify.steps.step import Step


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
    l: np.ndarray[int]
        ...
    u: np.ndarray[int]
        ...
    equality_ranges: list
        List of index intervals that specify which parts of the output receive
        the SatLU projection.
    eq_indices: set[int]
        Indices of output coordinates subjected to Saturated Linear Unit projection.
    ineq_indices: set[int]
        Indices of output coordinates not subjected to Saturated Linear Unit projection.
    """

    # # todo: complete the docstring
    def __init__(
        self,
        lhs: LinearExpression,
        rhs: LinearExpression,
        l,
        u,
        relax_binary_vars: bool = False,
        lb: float = None,
        ub: float = None,
    ):
        super().__init__(lhs, rhs, relax_binary_vars)
        self.l = l
        self.u = u

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    def new_bound(self, x):
        try:
            return np.minimum(np.maximum(x, self.l.lb), self.u.ub)
        except AttributeError:
            return np.minimum(np.maximum(x, self.l), self.u)

    def get_constraints(self):
        constraints = []

        for l, u, lb, ub, lhs, rhs in zip(self.l, self.u, self.lb, self.ub, self.lhs, self.rhs):
            try:
                float(l), float(u)
            except TypeError:
                return constraints

            if lb >= u:
                constraints.append(lhs == u)
            elif ub <= l:
                constraints.append(lhs == l)
            elif lb >= l and ub <= u:
                constraints.append(lhs == rhs)
            else:
                if not self.relax_binary_vars:
                    w1 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(rhs.ndim, is_binary=True)

                    if lb < l:
                        constraints.extend(
                            [
                                lhs <= rhs + (ub - lb) * w2,
                                rhs >= l + (lb - l) * w2,
                                rhs <= l + (ub - l) * (1 - w2),
                            ]
                        )

                        if ub > u:
                            constraints.extend(
                                [
                                    lhs >= u - (u - l) * (1 - w1),
                                    lhs <= l + (u - l) * (1 - w2),
                                    lhs >= l + (lb - l) * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                    if ub > u:
                        constraints.extend(
                            [
                                lhs >= rhs - (ub - lb) * w1,
                                rhs >= u + (lb - u) * (1 - w1),
                                rhs <= u + (ub - u) * w1,
                            ]
                        )
                        if lb > l:
                            constraints.append(lhs >= u - (u - lb) * (1 - w1))

                if lb < l and ub > u:
                    expr = (u - l) / (u - lb) * (rhs - u) + u
                    constraints.extend([lhs <= expr, lhs >= expr])

                elif lb < l and ub < u:
                    constraints.extend(
                        [lhs >= rhs, lhs <= ((ub - l) / (ub - lb)) * (rhs - ub) + ub]
                    )

                elif lb > l and ub > u:
                    constraints.extend(
                        [lhs <= rhs, lhs >= ((u - lb) / (ub - lb) * (rhs - lb) + lb)]
                    )

        return constraints
