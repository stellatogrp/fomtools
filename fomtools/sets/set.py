import numpy as np

from fomtools.proxable_functions.proximal_operator import ProxableFunction


def _fmt(value) -> str:
    """Compact repr for a constructor argument.

    Scalars/objects use their normal ``repr``; small arrays are shown by value and
    large arrays (> 8 elements) by shape, so reprs stay readable in interactive use
    and test-failure messages.
    """
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return np.array2string(value, separator=", ")
        return f"<ndarray shape={value.shape}>"
    return repr(value)


class Set:
    r"""
    Base class for convex sets onto which we can project.

    A :class:`Set` is the user-facing, math-named counterpart of an *indicator*
    :class:`~fomtools.proxable_functions.proximal_operator.ProxableFunction`: for a
    closed convex set $C$, the Euclidean projection onto $C$ is exactly the proximal
    operator of its indicator $I_C$,

    $$
        \Pi_C(x) = \operatorname{prox}_{I_C}(x)
                 = \arg\min_{u \in C} \tfrac12 \lVert u - x \rVert_2^2 .
    $$

    Each concrete set therefore only needs to know its indicator
    (:meth:`indicator`); projection and membership reuse the indicator's existing
    numerical ``prox``/``eval``. The verifier-level :meth:`fomtools.verifier.Verifier.project`
    is a thin wrapper that calls ``prox`` on this indicator, so :class:`Set` adds no
    new lowering logic.

    Subclasses form an inheritance tree mirroring the mathematics (e.g.
    :class:`~fomtools.sets.box.NonNegative` is a :class:`~fomtools.sets.polyhedron.Polyhedron`),
    but each concrete set overrides :meth:`indicator` to return its *most specific*
    indicator so the efficient backend lowering is preserved.
    """

    def indicator(self) -> ProxableFunction:
        """
        Return the indicator :class:`ProxableFunction` whose prox is the Euclidean
        projection onto this set.

        Raises
        ------
        NotImplementedError
            Raised by the base class; concrete sets must implement this.
        """
        raise NotImplementedError

    def project(self, x: np.ndarray) -> np.ndarray:
        r"""
        Euclidean projection of the point ``x`` onto this set, $\Pi_C(x)$.

        Delegates to the indicator's numerical ``prox`` (projection is
        ``prox`` of the indicator).

        Notes
        -----
        A fresh indicator object is constructed on each call (construction is cheap —
        it only stores the set's parameters). If you are projecting/checking membership
        in a tight loop, build the indicator once and reuse it:
        ``ind = my_set.indicator(); ind.prox(x1); ind.prox(x2); ...``.
        """
        return self.indicator().prox(np.asarray(x, dtype=float))

    def contains(self, x: np.ndarray, tol: float = 1e-6, rtol: float = 1e-5) -> bool:
        r"""
        Return ``True`` if ``x`` lies in the set (within tolerance).

        Tested via projection idempotence: for a closed convex set,
        $x \in C \iff \Pi_C(x) = x$. This relies only on the indicator's ``prox``
        (which every indicator implements), so it works uniformly across all sets.

        The comparison uses :func:`numpy.allclose` semantics,
        $\lVert \Pi_C(x) - x \rVert \le \text{tol} + \text{rtol}\,\lvert x \rvert$
        (elementwise): ``tol`` is the absolute tolerance and ``rtol`` adds a relative,
        **scale-aware** slack so large-magnitude points are not held to an unrealistic
        absolute bound. Pass ``rtol=0.0`` for a pure absolute check. For cvxpy-solved
        sets (Simplex, L1Ball, Polyhedron) the round-trip inherits the solver's
        precision (~``PROX_ATOL``), so ``tol`` may need to be loosened accordingly.

        Parameters
        ----------
        x : np.ndarray
            Point to test.
        tol : float, default 1e-6
            Absolute tolerance.
        rtol : float, default 1e-5
            Relative tolerance (scale-aware term).

        Notes
        -----
        Calls :meth:`project`, which constructs a fresh indicator each call; for tight
        membership-checking loops reuse a single indicator (see :meth:`project`).
        """
        x = np.asarray(x, dtype=float)
        return bool(np.allclose(self.project(x), x, atol=tol, rtol=rtol))

    def _repr_fields(self) -> str:
        """Constructor-argument string for :meth:`__repr__`; overridden per set."""
        return ""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._repr_fields()})"
