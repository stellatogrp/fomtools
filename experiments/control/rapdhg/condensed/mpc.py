"""The condensed MPC program in the inequality form rAPDHG solves.

The program itself is assembled by ``common.condensed_qp``, shared with the
``osqp`` study, so the two solve the same control problem and any difference in
how they fly belongs to the algorithm.  What this module adds is the change of
form that rAPDHG needs, and the two scalars its step sizes are built from.

The paper poses

    minimize    (1/2) x' Q x + c' x
    subject to  A x <= b,

a one-sided inequality with a nonnegative dual, whereas the condensed program
carries the two-sided box ``lo <= z <= hi``.  Stacking

    G = [ I ; -I ],    h = [ hi ; -lo ]

turns one into the other exactly: ``G z <= h`` is ``z <= hi`` together with
``-z <= -lo``.  The signs are the whole content of the reformulation, and the
lower bound is the one that flips.  The paper's ``A`` and ``b`` are called ``G``
and ``h`` here because ``A`` is already the discrete-time state matrix of the
plant, inherited from :class:`~common.condensed_qp.CondensedQP`.

``G`` is stored densely and is never actually multiplied by: every product with
it is a sign flip and a copy.  It exists because the recording needs a matrix to
hand to code generation, which discovers the structure for itself -- the
emitted blocks come out sparse.

Two derived scalars complete the data.  ``norm_q`` and ``norm_g`` are the
spectral norms the step sizes of Theorem 1 are scaled by; both are properties of
the problem, computed once here rather than inside any loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.condensed_qp import CondensedQP, build_qp

#: Restart period, in inner iterations.  Chosen by ``tuning.py`` against how the
#: closed loop flies at a fixed iteration count, not against how fast one solve
#: converges; the two disagree, and the README has the sweep.
#:
#: The trade-off is real in both directions.  Too short and the method barely
#: accelerates -- at K = 8 the momentum coefficient never exceeds 5, and tracking
#: is 27% worse across the budget range.  Too long and a truncated solve stops
#: improving monotonically with its budget: at K = 64 the flight crashes at some
#: iteration counts and not at larger or smaller ones, which is unusable however
#: good the asymptotics.  Thirty-two accelerates properly and stays well behaved.
DEFAULT_RESTART = 32

#: Primal weight.  Multiplies the dual step and divides the primal one, which
#: leaves Lemma 1's condition (3) with slack ``(w - 1) ||Q|| / beta_k``, so any
#: value at or above one stays inside the paper's guarantee.
#:
#: It is what makes the chosen restart period safe: at K = 32 the worst
#: regression as the budget grows is 4.4x at w = 1 and 1.38x at w = 2.
DEFAULT_PRIMAL_WEIGHT = 2.0


@dataclass
class RapdhgProblem(CondensedQP):
    """A condensed MPC quadratic program in the form of the paper's problem (1)."""

    #: Restart period K: the inner loop runs this many steps between restarts.
    restart: int = DEFAULT_RESTART
    #: Primal weight w.
    primal_weight: float = DEFAULT_PRIMAL_WEIGHT

    G: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    h: np.ndarray = field(default_factory=lambda: np.zeros(0))

    norm_q: float = 0.0
    norm_g: float = 0.0

    @property
    def m(self) -> int:
        """Number of inequality rows, twice the number of variables."""
        return self.G.shape[0]

    def apply_g(self, v: np.ndarray) -> np.ndarray:
        """``G @ v``, using the structure rather than the stored matrix.

        ``G`` is ``[I; -I]``, so the product is a copy and a sign flip.  Forming
        it as a dense matrix-vector product would cost 3200 multiply-adds where
        80 are structurally nonzero, and code generation emits these blocks
        sparse -- so a NumPy arm that used the dense matrix would be losing to
        the generated C on a difference in what the two were asked to compute
        rather than on how fast they compute it.
        """
        return np.concatenate([v, -v])

    def apply_gt(self, y: np.ndarray) -> np.ndarray:
        """``G' @ y``, likewise: the difference of the two halves."""
        n = self.n
        return y[:n] - y[n:]


def build_mpc(
    dt: float = 0.01,
    lookahead: float = 1.0,
    block: int = 10,
    input_weight: float = 1.0,
    restart: int = DEFAULT_RESTART,
    primal_weight: float = DEFAULT_PRIMAL_WEIGHT,
) -> RapdhgProblem:
    """Assemble the condensed program and put it in one-sided inequality form.

    Parameters mirror ``common.condensed_qp.build_qp`` with two additions,
    ``restart`` and ``primal_weight``, which are the algorithm's two knobs.
    """
    qp = build_qp(dt=dt, lookahead=lookahead, block=block, input_weight=input_weight)
    n = qp.n
    identity = np.eye(n)
    G = np.vstack([identity, -identity])
    return RapdhgProblem(
        dt=qp.dt,
        horizon=qp.horizon,
        block=qp.block,
        A=qp.A,
        B=qp.B,
        P=qp.P,
        Gx=qp.Gx,
        Gr=qp.Gr,
        lo=qp.lo,
        hi=qp.hi,
        restart=restart,
        primal_weight=primal_weight,
        G=G,
        h=np.concatenate([qp.hi, -qp.lo]),
        norm_q=float(np.linalg.norm(qp.P, 2)),
        # ||[I; -I]|| = sqrt(2) exactly; computed rather than asserted so that a
        # future constraint matrix does not silently keep this value.
        norm_g=float(np.linalg.norm(G, 2)),
    )
