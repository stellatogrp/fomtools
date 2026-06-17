"""Convex sets for the ``Verifier.project`` API.

Import concrete sets from this package, e.g.::

    from fomtools.sets import NonNegative, Box, Polyhedron, L1Ball

Only the base :class:`~fomtools.sets.set.Set` is re-exported at the top level
(``from fomtools import Set``), mirroring how only :class:`~fomtools.steps.step.Step`
is exposed there; the concrete sets live here in ``fomtools.sets`` (parallel to how the
concrete steps live in ``fomtools.steps``).

Available sets: :class:`Polyhedron`, :class:`HalfSpace`, :class:`Simplex`, :class:`Box`,
:class:`NonNegative`, :class:`NonPositive`, :class:`Point`, :class:`L1Ball`,
:class:`LinfBall`, :class:`AffineSubspace`.
"""

from fomtools.sets.set import Set
from fomtools.sets.polyhedron import Polyhedron, HalfSpace, Simplex
from fomtools.sets.box import Box, NonNegative, NonPositive, Point
from fomtools.sets.ball import L1Ball, LinfBall
from fomtools.sets.affine_subspace import AffineSubspace
