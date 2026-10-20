from fomtools.proxable_functions.ind_affine import IndAffine
from fomtools.sets.set import Set, _fmt


class AffineSubspace(Set):
    r"""
    The affine subspace $\{x : A x = b\}$.

    Backed by `IndAffine`, which
    projects onto the equality set in closed form. Named ``AffineSubspace`` to avoid
    colliding with the `Affine` *step*. Sits directly
    under `Set` (it is an equality, not an ``Ax <= b``
    inequality polyhedron).

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix ``(m, n)`` (full row rank).
    b : float, np.ndarray, or LinearExpression
        Right-hand side (dim ``m``).
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def indicator(self) -> IndAffine:
        return IndAffine(self.A, self.b)

    def _repr_fields(self) -> str:
        return f"A={_fmt(self.A)}, b={_fmt(self.b)}"
