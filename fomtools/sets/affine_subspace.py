from fomtools.proxable_functions.ind_affine import IndAffine
from fomtools.sets.set import Set


class AffineSubspace(Set):
    r"""
    The affine subspace $\{x : A x = b\}$.

    Backed by :class:`~fomtools.proxable_functions.ind_affine.IndAffine`, which
    projects onto the equality set in closed form. Named ``AffineSubspace`` to avoid
    colliding with the :class:`~fomtools.steps.affine.Affine` *step*. Sits directly
    under :class:`~fomtools.sets.set.Set` (it is an equality, not an ``Ax <= b``
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
