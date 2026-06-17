import numpy as np

from fomtools.proxable_functions.ind_half_space import IndHalfSpace
from fomtools.proxable_functions.ind_polyhedral import IndPolyhedral
from fomtools.proxable_functions.ind_simplex import IndSimplex
from fomtools.sets.set import Set, _fmt


class Polyhedron(Set):
    r"""
    The polyhedron $\{x : A x \le b\}$.

    Backed by :class:`~fomtools.proxable_functions.ind_polyhedral.IndPolyhedral`,
    whose prox solves the projection via a KKT system. ``dual_ub`` bounds the
    inequality multipliers and is required by the MIP/SDP lowering.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix ``(m, n)``.
    b : np.ndarray or LinearExpression
        Right-hand side (dim ``m``).
    dual_ub : float, optional
        Upper bound on the inequality dual variables (needed by the backend).
    """

    def __init__(self, A=None, b=None, dual_ub=None):
        self.A = A
        self.b = b
        self.dual_ub = dual_ub

    def indicator(self) -> IndPolyhedral:
        return IndPolyhedral(self.A, self.b, dual_ub=self.dual_ub)

    def _repr_fields(self) -> str:
        return f"A={_fmt(self.A)}, b={_fmt(self.b)}, dual_ub={self.dual_ub!r}"


class HalfSpace(Polyhedron):
    r"""
    The half-space $\{x : a^\top x \le b\}$ (a single linear inequality).

    A :class:`Polyhedron` with a single row: ``A = a`` (shape ``(1, n)``) and RHS vector
    ``b = [b]``. The inherited ``self.A``/``self.b`` therefore carry the usual polyhedral
    meaning (``self.b`` is the length-1 RHS vector, *not* a bare scalar); the scalar normal
    is kept on ``self.a``. Backed by
    :class:`~fomtools.proxable_functions.ind_half_space.IndHalfSpace`, which projects in
    closed form (no dual bound needed).

    Parameters
    ----------
    a : np.ndarray
        Normal vector (nonzero).
    b : float
        Offset (the scalar right-hand side of ``a^T x <= b``).
    """

    def __init__(self, a, b):
        a = np.asarray(a, dtype=float)
        # A half-space is a polyhedron with one row; store A/b with the inherited meaning.
        super().__init__(A=a.reshape(1, -1), b=np.array([float(b)]))
        self.a = a

    def indicator(self) -> IndHalfSpace:
        return IndHalfSpace(self.a, float(self.b[0]))

    def _repr_fields(self) -> str:
        return f"a={_fmt(self.a)}, b={float(self.b[0])!r}"


class Simplex(Polyhedron):
    r"""
    The scaled probability simplex $\{x : x \ge 0,\ \mathbf{1}^\top x = a\}$.

    Backed by :class:`~fomtools.proxable_functions.ind_simplex.IndSimplex`.

    Parameters
    ----------
    a : float or LinearExpression, default 1.0
        Total mass (``a > 0``).
    """

    def __init__(self, a=1.0):
        super().__init__()
        self.a = a

    def indicator(self) -> IndSimplex:
        return IndSimplex(self.a)

    def _repr_fields(self) -> str:
        return f"a={self.a!r}"
