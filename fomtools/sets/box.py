import numpy as np

from fomtools.proxable_functions.ind_box import IndBox
from fomtools.proxable_functions.ind_nonnegative import IndNonNegative
from fomtools.proxable_functions.ind_nonpositive import IndNonPositive
from fomtools.proxable_functions.ind_point import IndPoint
from fomtools.sets.polyhedron import Polyhedron


class Box(Polyhedron):
    r"""
    The axis-aligned box $\{x : l \le x \le u\}$ (element-wise).

    A :class:`~fomtools.sets.polyhedron.Polyhedron` specialization that projects by
    element-wise clipping; backed by
    :class:`~fomtools.proxable_functions.ind_box.IndBox`.

    Parameters
    ----------
    l, u : float, np.ndarray, or LinearExpression
        Lower and upper bounds (scalars broadcast to the dimension of the input).
    """

    def __init__(self, l, u):
        super().__init__()
        self.l = l
        self.u = u

    def indicator(self) -> IndBox:
        return IndBox(self.l, self.u)


class NonNegative(Box):
    r"""
    The non-negative orthant $\{x : x \ge 0\}$, i.e. the box $[0, +\infty)$.

    A parameterless :class:`Box` specialization (``l = 0``, ``u = +\infty``); backed by
    :class:`~fomtools.proxable_functions.ind_nonnegative.IndNonNegative`, which projects
    via a single ReLU (more efficient than the generic box/polyhedron path). The box
    bounds are stored for the inherited :class:`Box`/:class:`Polyhedron` contract but are
    not used by :meth:`indicator`.
    """

    def __init__(self):
        super().__init__(0.0, np.inf)

    def indicator(self) -> IndNonNegative:
        return IndNonNegative()


class NonPositive(Box):
    r"""
    The non-positive orthant $\{x : x \le 0\}$, i.e. the box $(-\infty, 0]$.

    A parameterless :class:`Box` specialization (``l = -\infty``, ``u = 0``); backed by
    :class:`~fomtools.proxable_functions.ind_nonpositive.IndNonPositive`. The box bounds
    are stored for the inherited contract but are not used by :meth:`indicator`.
    """

    def __init__(self):
        super().__init__(-np.inf, 0.0)

    def indicator(self) -> IndNonPositive:
        return IndNonPositive()


class Point(Box):
    r"""
    The singleton $\{p\}$ (a degenerate box with $l = u = p$).

    Backed by :class:`~fomtools.proxable_functions.ind_point.IndPoint`; projection
    is the constant map $x \mapsto p$.

    Parameters
    ----------
    p : float or np.ndarray
        The single point (scalars broadcast to the dimension of the input).
    """

    def __init__(self, p):
        super().__init__(p, p)  # degenerate box l = u = p
        self.p = p

    def indicator(self) -> IndPoint:
        return IndPoint(self.p)
