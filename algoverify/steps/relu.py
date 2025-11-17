from dataclasses import dataclass, field, InitVar
import numpy as np
from algoverify.steps.step import Step
from algoverify.linearexpression import LinearExpression
from algoverify.utils.steps_utils import linear_bound_propagation

@dataclass(frozen=False, kw_only=True)
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
    idx_with_binary_vars: set[int]
        Indices corresponding to binary decision variables.
    proj_indices: set[int]
        Indices of output coordinates subjected to ReLU projection.
    nonproj_indices: set[int]
        Set of output indices not subjected to ReLU.
    """

    proj_ranges: InitVar[list[tuple[int, int]]] = None
    idx_with_binary_vars: set[int] = field(default_factory=set)
    proj_indices: set = field(init=False)

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    def __post_init__(self, proj_ranges: list[tuple[int, int]]) -> list[int]:
        def _get_ranges(ranges):
            """
            Return the set of integers covered by one or more of half-open intervals.

            Parameters
            ----------
            ranges: tuple[int, int] or Iterable[tuple[int, int]]
                Either a single (start, end) pair or an iterable of such pair.
                Each pair is interpreted as a half-open interval ``[start, end)``.

            Returns
            -------
            set[int]
                The unique integers contained in the given half-open range(s)

            Examples
            --------
            >>> _get_ranges(((1, 2), (5, 9)))
            {1, 2, 5, 6, 7, 8}

            >>> _get_ranges((4, 7))
            {4, 5, 6}
            """
            try:
                ranges.append((0, 0))
                return {i for (min, max) in ranges for i in range(min, max)}
            except AttributeError:
                return set(range(ranges[0], ranges[1]))

        object.__setattr__(
            self,
            "proj_indices",
            set(range(0, self.rhs.ndim)) if proj_ranges is None else _get_ranges(proj_ranges),
        )
        self.lb, self.ub = linear_bound_propagation(expression=self.rhs, bounds=self.bounds)

    # TODO : out_lb and out_ub can have common function that they can call from
    def new_bound(self, b):
        return np.where(
            np.isin(np.arange(len(b)), list(self.proj_indices)),
            np.maximum(0, b),
            b,
        )

    def get_constraints(self):
        constraints = []
        for i, (_lhs, _rhs) in enumerate(zip(self.lhs, self.rhs)):
            if i not in self.proj_indices:
                constraints.append(_lhs == _rhs)
            else:
                if self.ub[i] <= 0:
                    constraints.append(_lhs == 0)
                elif self.lb[i] > 0:
                    constraints.append(_lhs == _rhs)
                elif self.ub[i] > 0 or self.lb[i] <= 0:
                    new_var = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                    self.idx_with_binary_vars.add(i)
                    constraints.append(_lhs <= self.ub[i] * new_var)

                    constraints.extend(
                        [
                            _lhs <= self.ub[i] / (self.ub[i] - self.lb[i]) * (_rhs - self.lb[i]),
                            _lhs >= _rhs,
                            _lhs <= _rhs - self.lb[i] * (1 - new_var),
                        ]
                    )
        return constraints
