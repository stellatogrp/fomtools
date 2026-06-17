import numpy as np

from fomtools.proxable_functions.proximal_operator import ProxableFunction


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
        """
        return self.indicator().prox(np.asarray(x, dtype=float))

    def contains(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        r"""
        Return ``True`` if ``x`` lies in the set (within ``tol``).

        Tested via projection idempotence: for a closed convex set,
        $x \in C \iff \Pi_C(x) = x$. This relies only on the indicator's ``prox``
        (which every indicator implements), so it works uniformly across all sets.
        """
        x = np.asarray(x, dtype=float)
        return bool(np.allclose(self.project(x), x, atol=tol, rtol=0.0))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
