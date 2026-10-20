"""Convex sets for the ``Verifier.project`` API.

Import concrete sets from this package, e.g.::

    from fomtools.sets import NonNegative, Box, Polyhedron, L1Ball

Only the base `Set` is re-exported at the top level
(``from fomtools import Set``), mirroring how only `Step`
is exposed there; the concrete sets live here in ``fomtools.sets`` (parallel to how the
concrete steps live in ``fomtools.steps``).

Available sets: `Polyhedron`, `HalfSpace`, `Simplex`, `Box`,
`NonNegative`, `NonPositive`, `Point`, `L1Ball`,
`LinfBall`, `AffineSubspace`.
"""

from fomtools.sets.set import Set
from fomtools.sets.polyhedron import Polyhedron, HalfSpace, Simplex
from fomtools.sets.box import Box, NonNegative, NonPositive, Point
from fomtools.sets.ball import L1Ball, LinfBall
from fomtools.sets.affine_subspace import AffineSubspace
