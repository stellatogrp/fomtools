from typing import List, Tuple

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.steps.step import Step
from fomtools.utils.dict_utils import prune_dict


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

    Notes
    -----
    Partial projection (``proj_ranges`` covers only a subset of the input
    coordinates) is encoded with a *composite* ``lhs``: only
    ``len(proj_indices)`` fresh leaf variables are introduced (stored on
    ``self._lhs_proj_leaf``), and the non-projected output coordinates
    share decomposition with ``rhs`` directly. This keeps the SDP variable
    count, MIP variable count, and simulator leaf count at the minimum.
    For the full-projection default, ``lhs`` remains a single leaf of size
    ``rhs.ndim`` (current behavior).
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

        rhs_lb, rhs_ub = self.rhs.bound_propagation()

        if self.proj_indices != set(range(self.rhs.ndim)):
            # Initialize the proj-leaf with array-valued bounds so the composite
            # lhs's bound_propagation has a consistent shape to work with.
            proj_list = sorted(self.proj_indices)
            proj_lb = np.maximum(0, np.asarray(rhs_lb)[proj_list])
            proj_ub = np.maximum(0, np.asarray(rhs_ub)[proj_list])
            self._lhs_proj_leaf = LinearExpression.from_int(
                len(self.proj_indices), lb=proj_lb, ub=proj_ub
            )
            # Override Step.lhs cached_property with a composite expression
            # whose non-proj rows share storage with rhs and whose proj rows
            # come from _lhs_proj_leaf.
            self.__dict__["lhs"] = self._build_composite_lhs()
        else:
            self._lhs_proj_leaf = None

        self.update_bounds(rhs_lb, rhs_ub)

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    @staticmethod
    def _get_ranges(ranges: Tuple[int, int] | List[Tuple[int, int]]) -> set[int]:
        """
        Return the set of integers covered by one or more of half-open intervals.

        Parameters
        ----------
        ranges: Tuple[int, int] or List[Tuple[int, int]]
            Either a single (start, end) pair or an iterable of such pairs.
            Each pair is interpreted as a half-open interval ``[start, end)``.

        Returns
        -------
        set[int]
            The unique integers contained in the given half-open range(s).

        Examples
        --------
        >>> Relu._get_ranges(((1, 2), (5, 9)))
        {1, 5, 6, 7, 8}

        >>> Relu._get_ranges((4, 7))
        {4, 5, 6}
        """
        # Try the iterable-of-pairs form first. If ``ranges`` is a single
        # ``(start, end)`` pair, unpacking its int elements raises TypeError
        # ("cannot unpack non-iterable int") — fall back to the single-range
        # form. This avoids the pre-existing ``ranges.append((0, 0))``
        # workaround which silently mutated caller-owned lists.
        try:
            return {i for (start, end) in ranges for i in range(start, end)}
        except TypeError:
            return set(range(ranges[0], ranges[1]))

    def _build_composite_lhs(self) -> LinearExpression:
        """Construct the composite lhs: proj rows from ``_lhs_proj_leaf``,
        non-proj rows passed through directly from ``rhs``.

        The composite has dimension ``rhs.ndim`` and its decomposition_dict
        references both ``_lhs_proj_leaf`` and the leaves of ``rhs``.
        """
        n = self.rhs.ndim
        proj_list = sorted(self.proj_indices)
        nonproj_list = sorted(set(range(n)) - self.proj_indices)
        k = len(proj_list)

        # Selection matrix placing _lhs_proj_leaf (size k) into proj rows of output.
        S_proj = spa.csc_matrix(
            (np.ones(k), (np.array(proj_list, dtype=int), np.arange(k))),
            shape=(n, k),
        )

        # Selection matrix masking out proj rows of rhs; non-proj rows of rhs
        # pass through to output unchanged.
        m = len(nonproj_list)
        if m > 0:
            nonproj_arr = np.array(nonproj_list, dtype=int)
            S_nonproj = spa.csc_matrix(
                (np.ones(m), (nonproj_arr, nonproj_arr)),
                shape=(n, n),
            )
        else:
            S_nonproj = spa.csc_matrix((n, n))

        decomposition_dict: dict = {self._lhs_proj_leaf: S_proj}
        for leaf, coeff in self.rhs.decomposition_dict.items():
            composed = S_nonproj @ coeff
            if leaf in decomposition_dict:
                decomposition_dict[leaf] = decomposition_dict[leaf] + composed
            else:
                decomposition_dict[leaf] = composed

        const = S_nonproj @ self.rhs.const

        # Drop entries whose composed coefficient is all-zero — happens when
        # an rhs leaf contributes only to projected rows, since ``S_nonproj``
        # zeros out those rows. Pruning prevents downstream canonicalizers
        # from looking up an irrelevant leaf in ``var_idx_map``.
        result = LinearExpression(
            decomposition_dict=prune_dict(decomposition_dict), const=const
        )
        # Initialize array-valued bounds so later update_bounds calls have a
        # consistent shape to maximize/minimize against.
        result.lb, result.ub = result.bound_propagation()
        return result

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

    @property
    def lhs_leaves(self) -> list[LinearExpression]:
        """For partial projection, only ``_lhs_proj_leaf`` is fresh; non-proj
        rows of the composite lhs reuse the rhs's leaves. For full projection,
        the lhs is itself a single fresh leaf.
        """
        if self._lhs_proj_leaf is not None:
            return [self._lhs_proj_leaf]
        return [self.lhs]

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
        if self._lhs_proj_leaf is not None:
            proj_list = sorted(self.proj_indices)
            proj_lb = np.maximum(0, np.asarray(lb)[proj_list])
            proj_ub = np.maximum(0, np.asarray(ub)[proj_list])
            self._lhs_proj_leaf.update_bounds(proj_lb, proj_ub)
