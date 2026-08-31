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

The tilt bound stacks underneath in the same way,

    G = [ I ; -I ; C ; -C ],    h(x) = [ hi ; -lo ; e - F x ; e + F x ],

and it is what makes the right-hand side a genuine parameter: the thrust rows
of ``h`` are constant, the tilt rows move with the measured state.

``G`` is stored densely, but the products with it are applied by structure:
the box rows are a copy and a sign flip, and only the tilt rows cost a real
matrix-vector product, one ``2 num_blocks`` by ``n`` block each way.  The
dense matrix exists because the recording needs one to hand to code
generation, which discovers the structure for itself and emits the box rows
sparse.

Two derived scalars complete the data.  ``norm_q`` and ``norm_g`` are the
spectral norms the step sizes of Theorem 1 are scaled by; both are properties of
the problem, computed once here rather than inside any loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import svdvals

from common.condensed_qp import CondensedQP, build_qp
from common.quadcopter import TILT_LIMIT

#: Restart period, in inner iterations.  Chosen by ``tuning.py`` against how the
#: closed loop flies at a fixed iteration count, not against how fast one solve
#: converges; the two disagree, and the README has the sweep.
#:
#: The trade-off is real in both directions.  Too short and the method barely
#: accelerates; too long and a truncated solve stops improving monotonically
#: with its budget.  On the tilt-bounded program, flown from the reference,
#: K = 128 is marginally the most accurate (0.027 m typical against 0.028) but
#: tracks worse at two budgets than at the one below them, by up to 1.35x;
#: K = 64 is as accurate to within four percent and never regresses.
DEFAULT_RESTART = 64

#: Primal weight.  Multiplies the dual step and divides the primal one, which
#: leaves Lemma 1's condition (3) with slack ``(w - 1) ||Q|| / beta_k``, so any
#: value at or above one stays inside the paper's guarantee.
#:
#: One is the floor of that range and is the best of the swept values here at
#: every restart period: the tilt rows raise ``||G||`` from ``sqrt(2)`` to about
#: nine, which already shrinks the primal step relative to the dual one, and a
#: larger weight slows the primal further for no gain in how the bound is kept.
DEFAULT_PRIMAL_WEIGHT = 1.0


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
        """Number of inequality rows: two per variable, plus two per tilt row."""
        return self.G.shape[0]

    def rhs(self, state: np.ndarray) -> np.ndarray:
        """The right-hand side ``h`` at the measured ``state``.

        The thrust rows never move; the tilt rows are the limit less the free
        response from ``state``, so this is evaluated once per sampling instant,
        alongside the linear cost, and handed to the solver as its second
        parameter.
        """
        upper, lower = self.state_rhs(state)
        return np.concatenate([self.hi, -self.lo, upper, lower])

    def apply_g(self, v: np.ndarray) -> np.ndarray:
        """``G @ v``, using the structure rather than the stored matrix.

        ``G`` is ``[I; -I]``, so the product is a copy and a sign flip.  Forming
        it as a dense matrix-vector product would cost 3200 multiply-adds where
        80 are structurally nonzero, and code generation emits these blocks
        sparse -- so a NumPy arm that used the dense matrix would be losing to
        the generated C on a difference in what the two were asked to compute
        rather than on how fast they compute it.
        """
        if self.num_state_rows == 0:
            return np.concatenate([v, -v])
        cv = self.C @ v
        return np.concatenate([v, -v, cv, -cv])

    def apply_gt(self, y: np.ndarray) -> np.ndarray:
        """``G' @ y``, likewise: the difference of the box halves, plus ``C'`` on
        the difference of the tilt halves."""
        n, k = self.n, self.num_state_rows
        out = y[:n] - y[n : 2 * n]
        if k:
            out = out + self.C.T @ (y[2 * n : 2 * n + k] - y[2 * n + k : 2 * n + 2 * k])
        return out


def build_mpc(
    dt: float = 0.01,
    lookahead: float = 1.0,
    block: int = 10,
    input_weight: float = 1.0,
    restart: int = DEFAULT_RESTART,
    primal_weight: float = DEFAULT_PRIMAL_WEIGHT,
    tilt_limit: float | None = TILT_LIMIT,
) -> RapdhgProblem:
    """Assemble the condensed program and put it in one-sided inequality form.

    Parameters mirror ``common.condensed_qp.build_qp`` with two additions,
    ``restart`` and ``primal_weight``, which are the algorithm's two knobs.  The
    tilt bound is on by default here, unlike in ``build_qp``: this study flies
    the bounded model, the ``osqp`` study the published unbounded one.
    """
    qp = build_qp(
        dt=dt, lookahead=lookahead, block=block, input_weight=input_weight, tilt_limit=tilt_limit
    )
    n = qp.n
    identity = np.eye(n)
    G = np.vstack([identity, -identity, qp.C, -qp.C])
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
        C=qp.C,
        F=qp.F,
        e=qp.e,
        restart=restart,
        primal_weight=primal_weight,
        G=G,
        # ``h`` at the hover state; the controller re-evaluates ``rhs`` per instant.
        h=np.concatenate([qp.hi, -qp.lo, qp.e, qp.e]),
        norm_q=float(np.linalg.norm(qp.P, 2)),
        # sqrt(2) for the box alone; the tilt rows raise it (about 9 here).
        norm_g=float(svdvals(G)[0]),
    )
