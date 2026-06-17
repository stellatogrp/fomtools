from fomtools.proxable_functions.ind_half_space import IndHalfSpace
from fomtools.proxable_functions.ind_polyhedral import IndPolyhedral
from fomtools.proxable_functions.ind_simplex import IndSimplex
from fomtools.sets.set import Set


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


class HalfSpace(Polyhedron):
    r"""
    The half-space $\{x : a^\top x \le b\}$ (a single linear inequality).

    Backed by :class:`~fomtools.proxable_functions.ind_half_space.IndHalfSpace`,
    which projects in closed form (no dual bound needed).

    Parameters
    ----------
    a : np.ndarray
        Normal vector (nonzero).
    b : float
        Offset.
    """

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def indicator(self) -> IndHalfSpace:
        return IndHalfSpace(self.a, self.b)


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
