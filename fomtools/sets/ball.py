from fomtools.proxable_functions.ind_ball_l1 import IndBallL1
from fomtools.proxable_functions.ind_ball_linf import IndBallLinf
from fomtools.sets.box import Box
from fomtools.sets.set import Set


class L1Ball(Set):
    r"""
    The $\ell_1$-ball $\{x : \lVert x \rVert_1 \le r\}$.

    Backed by `IndBallL1`, whose
    projection lifts auxiliary variables and solves a KKT system. ``dual_ub`` bounds
    the inequality multipliers for the MIP/SDP lowering. Sits directly under
    `Set` (the lifting does not fit the plain ``Ax <= b`` mold).

    Parameters
    ----------
    r : float or LinearExpression
        Radius (``r >= 0``).
    dual_ub : float, optional
        Upper bound on the inequality dual variables (needed by the backend).
    """

    def __init__(self, r, dual_ub=None):
        self.r = r
        self.dual_ub = dual_ub

    def indicator(self) -> IndBallL1:
        return IndBallL1(self.r, dual_ub=self.dual_ub)

    def _repr_fields(self) -> str:
        return f"r={self.r!r}, dual_ub={self.dual_ub!r}"


class LinfBall(Box):
    r"""
    The $\ell_\infty$-ball $\{x : \lVert x \rVert_\infty \le r\}$, i.e. the box
    $[-r, r]$.

    A `Box` specialization; backed by
    `IndBallLinf`, which projects
    by clipping to $[-r, r]$.

    Parameters
    ----------
    r : float
        Radius (``r >= 0``).
    """

    def __init__(self, r):
        super().__init__(-r, r)  # the box [-r, r]
        self.r = r

    def indicator(self) -> IndBallLinf:
        return IndBallLinf(self.r)

    def _repr_fields(self) -> str:
        return f"r={self.r!r}"
