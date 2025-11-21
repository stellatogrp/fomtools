from typing import List, Tuple

import numpy as np

from algoverify.constraint import Constraint
from algoverify.linearexpression import LinearExpression
from algoverify.steps.step import Step


class Relu(Step):
    r"""
    Rectified Linear Unit (ReLU) step operator, defined as:

    $$
        (\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.
    $$

    It sets all the negative values to zero while keeping the positive ones unchanged.

    Attributes
    ----------
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
        Update the bounds of the left-land side after applying
        ReLU activation.

        Parameters
        ----------
        lb: float or np.ndarray
            The new lower bound(s) before applying ReLU.

        ub: float or np.ndarray
            The new upper bound(s) before applying ReLU.

        """
        self.lhs.update_bounds(lb=self.relu(lb), ub=self.relu(ub))

    def get_constraints(self) -> List[Constraint]:
        r"""
        Construct the convex-hull linear constraints corresponding
        to the `Rectified Linear Unit (ReLU)` operator.

        Given a ReLU step: $(\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.$

        This method derives the convex hull of the region defined by:

        $$
            \phi = \left\{ (y, w) \in \left [\underline{y}, \overline{y} \right ] \times
            \mathbb{R} | w = \textbf{ReLU}(a^{T}y) \right\},
        $$

        and expresses it as a set of Constraint objects. The convex hull
        of the region is:


        $$
            \operatorname{conv}(\Phi) =
            \left\{
            (y, w) \in [\underline{y}, \overline{y}] \times \mathbb{R}
            \;\middle|\;
            \begin{cases}
            w = a^{\mathsf{T}} y, & \ell_J \ge 0, \\[6pt]
            w = 0, & \ell_J < 0, \\[6pt]
            (y, w) \in Q, & \text{otherwise}
            \end{cases}
            \right\},
        $$

        with

        $$
            Q =
            \left\{
            (y, w) \in \mathbb{R}^{d + p + 1}
            \;\middle|\;
            \begin{aligned}
            & w \ge a^{\mathsf{T}} y, \quad y \ge 0, \\[4pt]
            & w \le \sum_{i \in I} a_i (y_i - \ell_i^{0})
               + \frac{\ell_I}{u_o^{0} - \ell_o^{0}} (y_o - \ell_o^{0}),
               \quad \forall (I, o) \in \mathcal{I}
            \end{aligned}
            \right\},
        $$

        where:

        - vector $l^{0} \in \mathbb{R}^{d}$ with entries $l_i^{0} = \underline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \overline{y}_i$ otherwise,
        - vector $u^{0} \in \mathbb{R}^{d}$ with entries $u_i^{0} = \overline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \underline{y}_i$ otherwise,
        - Given $\textbf{I} \subseteq \textbf{J} = \left\{1, ..., n\right\}$, the lower bound is
        defined as $l_I = \sum_{i \in I}a_il_i^{0} + \sum_{i \notin I}a_iu_i^{0}$


        Returns
        -------
        List[Constraint]
            A list of constraint objects describing the convex hull of the feasible region
            defined by the ReLU activation.
        """
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
