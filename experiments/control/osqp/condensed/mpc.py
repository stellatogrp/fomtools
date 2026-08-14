"""The condensed MPC program with the data OSQP's ADMM needs on top of it.

The program itself -- the Hessian, the linear-cost maps, and the thrust box --
is assembled by ``common.condensed_qp``, which the ``rapdhg`` study builds on
too, so the two studies provably solve the same control problem.  What this
module adds is the one thing that belongs to the algorithm rather than to the
problem: the factorization of ``P + sigma I + rho I``, the matrix of the affine
step.  The constraint matrix of the condensed formulation is the identity, so
that is the whole linear system, and it is fixed by the problem data, so it is
factorized once here and only applied inside the loop.  That is the honest
counterpart of the generated code, which applies a factorization computed at
generation time; refactorizing inside the loop would overstate what code
generation contributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from common.condensed_qp import CondensedQP, build_qp


@dataclass
class MpcProblem(CondensedQP):
    """A condensed MPC quadratic program together with its ADMM data."""

    rho: float = 0.1
    sigma: float = 1e-4

    #: Cholesky factor of P + sigma I + rho I, the matrix of the affine step.
    _chol: tuple = field(default=(), repr=False)

    def solve_system(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the precomputed factorization of ``P + sigma I + rho I``."""
        return sla.cho_solve(self._chol, rhs)


def build_mpc(
    dt: float = 0.01,
    lookahead: float = 1.0,
    block: int = 10,
    rho: float = 0.1,
    sigma: float = 1e-4,
    input_weight: float = 1.0,
) -> MpcProblem:
    """Assemble the condensed MPC program and factorize its ADMM system.

    Parameters mirror ``common.condensed_qp.build_qp`` with two additions.

    Parameters
    ----------
    rho, sigma:
        The two ADMM parameters.  The defaults are the OSQP default penalty and
        the regularization used in the verification experiments; neither is
        tuned to this instance.
    """
    qp = build_qp(dt=dt, lookahead=lookahead, block=block, input_weight=input_weight)
    lhs = qp.P + (sigma + rho) * np.eye(qp.n)
    return MpcProblem(
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
        rho=rho,
        sigma=sigma,
        _chol=sla.cho_factor(lhs),
    )
