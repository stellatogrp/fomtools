"""Backend-aware iterate trajectory container.

An `IterateTrajectory` is a fixed-length, list-like holder of
`LinearExpression` iterates bound to a
`Verifier`. It exists to give users a single idiom
for building an algorithm's iterate sequence that automatically makes the right
representation choice per backend.

The choice being automated is whether to *collapse* an affine assignment.
Writing ``z[k] = M @ z[k - 1] + c`` produces a non-leaf ``LinearExpression``
that references existing leaf variables and mints no new solver variable --
this is the "collapsed" form and is what the MIP and simulator backends want
(fewer variables). For the SDP backend, however, keeping each iterate as its
own leaf variable -- linked to its defining expression by an equality
constraint -- leaves the moment matrix sparse, admitting a chordal
decomposition into many small PSD cones that is often much faster to solve.

``collapse_expr='auto'`` (the default) resolves this per backend: collapse on
MIP/simulator, keep a separate leaf on SDP. ``True`` / ``False`` force the
choice regardless of backend. The policy only *applies* to non-leaf values;
an assignment whose value is already a leaf (e.g. a ``relu`` output, or an
``iterate`` variable) is stored as-is on every backend.

``'auto'`` also watches for *accumulation*, which is the second situation where
collapsing is the wrong choice. Writing ``z[k] = z[k-1] + ut - u``, as
Douglas-Rachford and ADMM do, makes each slot reference every leaf the previous
one did plus the new step outputs, so the collapsed expression grows without
bound: leaf support runs 3, 5, 7, 9, ... down the trajectory. Every consumer then
pays for that growth -- code generation quadratically, since it expresses each
iterate in terms of leaves and earlier temporaries. A method that merely *refers*
to an earlier iterate holds its support flat instead (a Halpern anchor stays at 2
forever), which is what makes the two cases distinguishable without being told
which one this is. So ``'auto'`` collapses until support has strictly grown on two
consecutive assignments, then keeps a separate leaf for the rest of the
trajectory. The latch is deliberate: a materialized slot resets the support, so
re-deciding per slot would oscillate. ``collapse_expr=True`` forces collapsing
regardless, which is the escape hatch if the extra variables are unwanted.
"""

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression

#: Consecutive increases in leaf support before ``'auto'`` stops collapsing.
_GROWTH_TO_LATCH = 2


class IterateTrajectory:
    """Fixed-length, list-like sequence of iterates with backend-aware collapsing.

    Created via `fomtools.verifier.Verifier.iterate_trajectory`. Behaves
    like a fixed-length list: index with ``traj[k]`` (negative indices
    supported), assign with ``traj[k] = expr``, take ``len(traj)``, and iterate.

    Parameters
    ----------
    verifier : Verifier
        The owning verifier. Assignments that keep a separate leaf are routed
        through ``verifier.implicit_linear`` and the ``'auto'`` policy
        reads ``verifier.method``.
    length : int
        Number of slots (e.g. ``K + 1`` for ``K`` steps). Must be ``>= 1``.
    collapse_expr : {'auto', True, False}, default 'auto'
        Trajectory-level default collapse policy. ``'auto'`` collapses on
        MIP/simulator and keeps a separate leaf on SDP, and switches to keeping a
        separate leaf once the assigned expressions are seen to accumulate. May be
        overridden per slot via `assign`.
    """

    def __init__(self, verifier, length: int, collapse_expr="auto"):
        if length < 1:
            raise ValueError(f"IterateTrajectory length must be >= 1, got {length}.")
        _validate_policy(collapse_expr)
        self._verifier = verifier
        self._slots = [None] * length
        self._default_policy = collapse_expr
        # Accumulation tracking for the 'auto' policy; see the module docstring.
        self._prev_support = None
        self._consecutive_growth = 0
        self._accumulating = False

    def __len__(self) -> int:
        return len(self._slots)

    def _normalize_index(self, k: int) -> int:
        """Normalize (possibly negative) index ``k`` and range-check it."""
        try:
            idx = int(k)
        except (TypeError, ValueError):
            raise TypeError(f"Trajectory index must be an integer, got {k!r}.")
        n = len(self._slots)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Trajectory index {k} out of range for length {n}.")
        return idx

    def __getitem__(self, k):
        if isinstance(k, slice):
            raise TypeError("IterateTrajectory does not support slice access.")
        idx = self._normalize_index(k)
        value = self._slots[idx]
        if value is None:
            raise LookupError(f"Trajectory slot k={idx} is unset; assign it before reading.")
        return value

    def __setitem__(self, k, value):
        if isinstance(k, slice):
            raise TypeError("IterateTrajectory does not support slice assignment.")
        self.assign(k, value)

    def __iter__(self):
        for idx in range(len(self._slots)):
            yield self[idx]

    def assign(self, k, value, collapse_expr=None):
        """Assign ``value`` into slot ``k`` under an explicit collapse policy.

        Parameters
        ----------
        k : int
            Slot index (negative indices supported).
        value : LinearExpression
            The iterate expression to store.
        collapse_expr : {None, 'auto', True, False}, default None
            Per-slot collapse policy. ``None`` inherits the trajectory-level
            default; otherwise overrides it for this slot only.

        Notes
        -----
        The policy only has an effect when ``value`` is *not* already a leaf.
        A leaf value (e.g. a ``relu``/``iterate`` output) is stored as-is
        on every backend and policy, since it is already a single registered
        solver variable.
        """
        idx = self._normalize_index(k)
        if not isinstance(value, LinearExpression):
            raise TypeError(
                "IterateTrajectory slots must hold a LinearExpression, got "
                f"{type(value).__name__}. Build the iterate with a step such as verifier.relu "
                "or verifier.iterate before assigning."
            )

        policy = self._default_policy if collapse_expr is None else collapse_expr
        _validate_policy(policy)

        # Tracked for every assignment, leaf included: a leaf has support 1, so
        # storing one genuinely ends whatever growth preceded it.
        self._note_support(value)

        # A leaf is already a single registered variable; there is nothing to
        # collapse and no benefit to minting a redundant linking leaf.
        if value.is_leaf:
            self._slots[idx] = value
            return

        if self._resolve_collapse(policy):
            # Collapsed form: store the non-leaf expression as-is (current
            # behavior on MIP/simulator). Mints no variable.
            self._slots[idx] = value
        else:
            self._slots[idx] = self._make_linked_leaf(idx, value)

    def _note_support(self, value: LinearExpression) -> None:
        """Track leaf-support growth, latching once accumulation is evident.

        Two consecutive increases rather than one, so a recursion that widens once
        and then settles is not mistaken for unbounded growth. The cost is one
        extra iteration of growth before the latch, which stays bounded.
        """
        support = len(value.decomposition_dict)
        if self._prev_support is not None and support > self._prev_support:
            self._consecutive_growth += 1
        else:
            self._consecutive_growth = 0
        self._prev_support = support
        if self._consecutive_growth >= _GROWTH_TO_LATCH:
            self._accumulating = True

    def _resolve_collapse(self, policy) -> bool:
        """Resolve a policy to a concrete collapse decision (True = collapse)."""
        if policy is True or policy is False:
            return policy
        # 'auto': never collapse on SDP, nor once the assignments accumulate.
        if self._verifier.method == "sdp":
            return False
        return not self._accumulating

    def _make_linked_leaf(self, idx: int, value: LinearExpression) -> LinearExpression:
        """Mint a fresh leaf equal to ``value`` via an identity affine step."""
        # SDP requires finite bounds on every registered variable. The collapsed
        # baseline would never register this expression, so surface an
        # actionable, slot-scoped error rather than the deep SDP add_var failure.
        if self._verifier.method == "sdp":
            lb, ub = value.bound_propagation()
            finite = np.isfinite(lb) & np.isfinite(ub)
            if not np.all(finite):
                bad = int(np.argmin(finite))
                raise ValueError(
                    f"Trajectory slot k={idx} cannot be kept as a separate SDP "
                    f"variable because its bounds are unbounded (coordinate {bad} "
                    f"has lb={lb[bad]}, ub={ub[bad]}). Give the initial iterate / "
                    "params finite bounds, enable OBBT, or pass collapse_expr=True "
                    "for this slot."
                )
        return self._verifier.implicit_linear(value, output_mat=spa.eye(value.ndim))


def _validate_policy(collapse_expr):
    if collapse_expr not in (True, False, "auto"):
        raise ValueError(f"collapse_expr must be True, False, or 'auto', got {collapse_expr!r}.")
