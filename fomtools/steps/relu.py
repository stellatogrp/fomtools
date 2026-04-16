from typing import List, Tuple

import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.steps.step import Step


class Relu(Step):
    r"""
    Rectified Linear Unit (ReLU) step operator, defined as:

    $$
        (\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.
    $$

    It sets all the negative values to zero while keeping the positive ones unchanged.

    Attributes
    ----------
    rhs : LinearExpression
        Right-hand side (input) linear expression.
    lb : np.ndarray
        Lower bounds on the INPUT (rhs), before ReLU is applied.
    ub : np.ndarray
        Upper bounds on the INPUT (rhs), before ReLU is applied.
    relax_binary_vars : bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    idx_with_binary_vars : set[int]
        Indices corresponding to binary decision variables.
    proj_indices : set[int]
        Indices of output coordinates subjected to ReLU projection.
    """

    def __init__(
        self,
        rhs: LinearExpression,
        proj_ranges=None,
        relax_binary_vars=False,
    ):
        super().__init__(rhs, relax_binary_vars)
        self.proj_indices = (
            set(range(0, self.rhs.ndim)) if proj_ranges is None else self._get_ranges(proj_ranges)
        )
        self.idx_with_binary_vars = set()
        self.update_bounds(*self.rhs.bound_propagation())

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    @staticmethod
    def _get_ranges(ranges: Tuple[int, int] | List[Tuple[int, int]]) -> set[int]:
        """
        Return the set of integers covered by one or more of half-open intervals.

        Parameters
        ----------
        ranges: Tuple[int, int] or List[Tuple[int, int]]
            Either a single (start, end) pair or an iterable of such pair.
            Each pair is interpreted as a half-open interval ``[start, end)``.

        Returns
        -------
        set[int]
            The unique integers contained in the given half-open range(s)

        Examples
        --------
        >>> relu = Relu()
        >>> relu._get_ranges(((1, 2), (5, 9)))
        {1, 2, 5, 6, 7, 8}

        >>> relu._get_ranges((4, 7))
        {4, 5, 6}
        """
        try:
            ranges.append((0, 0))
            return {i for (min, max) in ranges for i in range(min, max)}
        except AttributeError:
            return set(range(ranges[0], ranges[1]))

    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ReLU activation selectively to specified indices.

        Parameters
        ----------
        x: np.ndarray
            A 1-D array of vlaues on which to apply the activation.

        Returns
        -------
        np.ndarray
            A new array where elements are positions specified in
            ``self.proj_indices`` are passed through ReLU, while
            all other elements are left unchanged.
        """
        proj_mask = np.isin(np.arange(len(x)), list(self.proj_indices))
        return np.where(proj_mask, np.maximum(0, x), x)

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update the bounds of the left-hand side after applying
        ReLU activation.

        Parameters
        ----------
        lb: float or np.ndarray
            The new lower bound(s) before applying ReLU.

        ub: float or np.ndarray
            The new upper bound(s) before applying ReLU.

        """
        self.lb, self.ub = lb, ub
        self.lhs.update_bounds(lb=self.relu(lb), ub=self.relu(ub))
