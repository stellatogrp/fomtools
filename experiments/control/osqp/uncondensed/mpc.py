"""Uncondensed model predictive control problem for the quadcopter.

The condensed formulation eliminates the states, which buys a small program at
the cost of a dense one: every input move influences every later state, so the
Hessian fills in completely.  This module keeps the states as variables and the
dynamics as constraints,

    minimize    (1/2) zeta' P zeta + q(r)' zeta
    subject to  l(x) <= M zeta <= u(x),

over ``zeta = (xi_1, ..., xi_N, z_1, ..., z_nb)``.  Nothing is eliminated, so
the program is roughly thirty times larger than the condensed one, but every
matrix in it is banded: ``P`` is block diagonal, and each row of ``M`` is either
one dynamics relation, which touches two consecutive states and one input move,
or one bound on an input move.

Both formulations describe the same control problem and have the same
minimizing first input, so the trajectories they fly are comparable and any
difference between them is a property of the algorithm rather than of the
controller.  The input is held over the same blocks, which here means the
dynamics rows reference the move covering their time step rather than a move of
their own.

There is one consequence that has nothing to do with size.  The measured state
enters through the constraint bounds, since ``xi_1 = x`` is an equality and an
equality is a bound with its two sides equal.  In the condensed formulation the
measured state enters the linear cost instead.  Code generation accepts the
second and rejects the first: its intermediate representation stores the clip
bounds of a saturated-linear step as constant arrays, so a bound that moves with
the parameter has nowhere to live.  See ``check_against_fomtools.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from common.quadcopter import (
    N_INPUT,
    N_STATE,
    POS,
    Q_STAGE,
    R_INPUT,
    TERMINAL_FACTOR,
    U_MAX,
    U_MIN,
    VEL,
    discretize,
)


@dataclass
class UncondensedProblem:
    """A full-space MPC quadratic program together with its ADMM data."""

    dt: float
    horizon: int  # number of predicted states, xi_1 ... xi_horizon
    block: int  # sampling instants over which the input is held constant
    sigma: float

    A: np.ndarray
    B: np.ndarray

    P: sp.csc_matrix  # block-diagonal cost
    M: sp.csc_matrix  # initial state, dynamics, and input bounds
    Qr: sp.csc_matrix  # maps a reference window to the state part of the cost
    rho: np.ndarray  # one penalty per constraint row

    lo: np.ndarray  # lower bounds; the first N_STATE entries are the parameter
    hi: np.ndarray  # upper bounds; likewise

    _solve: object = field(repr=False)  # factorization of P + sigma I + M' R M
    _system_nnz: int = 0
    _factor_nnz: int = 0

    @property
    def n(self) -> int:
        """Number of optimization variables."""
        return self.P.shape[0]

    @property
    def m(self) -> int:
        """Number of constraint rows."""
        return self.M.shape[0]

    @property
    def num_blocks(self) -> int:
        return (self.n - self.horizon * N_STATE) // N_INPUT

    @property
    def input_offset(self) -> int:
        """Index at which the input moves begin in ``zeta``."""
        return self.horizon * N_STATE

    @property
    def lookahead(self) -> float:
        return self.horizon * self.dt

    @property
    def system_nnz(self) -> int:
        """Stored entries in the system matrix ``P + sigma I + M' R M``."""
        return self._system_nnz

    @property
    def factor_nnz(self) -> int:
        """Stored entries in its factorization."""
        return self._factor_nnz

    def bounds(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Constraint bounds at a measured state.

        Only the first ``N_STATE`` rows move: they pin the initial state, and an
        equality is a bound whose two sides coincide.
        """
        lo = self.lo.copy()
        hi = self.hi.copy()
        lo[:N_STATE] = state
        hi[:N_STATE] = state
        return lo, hi

    def linear_cost(self, ref_window: np.ndarray) -> np.ndarray:
        """The cost vector at one sampling instant.

        Unlike the condensed formulation this depends on the reference only; the
        measured state has gone into the bounds.
        """
        q = np.zeros(self.n)
        q[: self.input_offset] = -(self.Qr @ ref_window)
        return q

    def reference_window(self, ref, t: float) -> np.ndarray:
        """Stack the reference over the prediction horizon starting at ``t``."""
        window = np.zeros(self.horizon * N_STATE)
        for i in range(self.horizon):
            pos, vel = ref(t + i * self.dt)
            base = i * N_STATE
            window[base + POS.start : base + POS.stop] = pos
            window[base + VEL.start : base + VEL.stop] = vel
        return window

    def solve_system(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the precomputed sparse factorization."""
        return self._solve(rhs)

    def first_input(self, zeta: np.ndarray) -> np.ndarray:
        """The rotor thrusts to apply now, saturated at the actuator limits."""
        start = self.input_offset
        return np.clip(zeta[start : start + N_INPUT], U_MIN, U_MAX)


def _cost(horizon: int, block: int, input_weight: float) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Block-diagonal cost, and the map from a reference window to its linear part.

    Each blocked move is charged for every step it covers, so the cost is the
    same function of the input sequence that the condensed formulation minimizes.
    """
    q_diag = np.tile(np.diag(Q_STAGE), horizon)
    q_diag[(horizon - 1) * N_STATE :] *= TERMINAL_FACTOR

    steps_per_block = np.bincount(np.arange(horizon - 1) // block)
    r_diag = np.concatenate([n * input_weight * np.diag(R_INPUT) for n in steps_per_block])

    P = sp.diags(np.concatenate([q_diag, r_diag]), format="csc")
    Qr = sp.diags(q_diag, format="csc")
    return P, Qr


def _constraints(A: np.ndarray, B: np.ndarray, horizon: int, block: int):
    """The constraint matrix and the constant parts of its bounds.

    Three groups of rows, in order: the initial state, the dynamics, and the
    input box.  The first group's bounds are filled in per sampling instant.
    """
    num_blocks = int(np.ceil((horizon - 1) / block))
    n_state_vars = horizon * N_STATE
    n = n_state_vars + num_blocks * N_INPUT

    rows = []
    # xi_1 = x.
    initial = sp.hstack(
        [sp.eye(N_STATE, format="csc"), sp.csc_matrix((N_STATE, n - N_STATE))], format="csc"
    )
    rows.append(initial)

    # xi_{t+1} - A xi_t - B z_{block(t)} = 0.
    dyn = sp.lil_matrix(((horizon - 1) * N_STATE, n))
    for t in range(horizon - 1):
        r = t * N_STATE
        dyn[r : r + N_STATE, (t + 1) * N_STATE : (t + 2) * N_STATE] = np.eye(N_STATE)
        dyn[r : r + N_STATE, t * N_STATE : (t + 1) * N_STATE] = -A
        j = t // block
        col = n_state_vars + j * N_INPUT
        dyn[r : r + N_STATE, col : col + N_INPUT] = -B
    rows.append(dyn.tocsc())

    # The input box.
    box = sp.hstack(
        [
            sp.csc_matrix((num_blocks * N_INPUT, n_state_vars)),
            sp.eye(num_blocks * N_INPUT, format="csc"),
        ],
        format="csc",
    )
    rows.append(box)

    M = sp.vstack(rows, format="csc")
    lo = np.concatenate(
        [np.zeros(N_STATE), np.zeros((horizon - 1) * N_STATE), np.tile(U_MIN, num_blocks)]
    )
    hi = np.concatenate(
        [np.zeros(N_STATE), np.zeros((horizon - 1) * N_STATE), np.tile(U_MAX, num_blocks)]
    )
    return M, lo, hi, num_blocks


def build_mpc(
    dt: float = 0.01,
    lookahead: float = 1.0,
    block: int = 10,
    rho: float = 0.1,
    sigma: float = 1e-4,
    input_weight: float = 1.0,
    rho_equality: float = 1000.0,
) -> UncondensedProblem:
    """Assemble the uncondensed MPC program.

    Parameters mirror the condensed builder, so the two describe the same
    control problem, with one addition.

    Parameters
    ----------
    rho_equality:
        Multiplier applied to the penalty on the equality rows, the initial
        state and the dynamics.  The default of 1000 is OSQP's own convention
        for equality constraints and is not tuned here; it is also not optional.
        At the same penalty as the inequality rows the iterate buys objective by
        violating the dynamics, and after five thousand iterations the predicted
        trajectory still breaks them by a tenth of a metre, which is to say the
        controller is optimizing over motions the quadcopter cannot perform.
        The condensed formulation has no equality rows for this to apply to: its
        dynamics are substituted away and every row of its constraint matrix is
        a thrust bound.
    """
    horizon = int(round(lookahead / dt))
    if horizon < 2:
        raise ValueError(f"lookahead {lookahead} s at dt {dt} s gives fewer than two steps")

    A, B = discretize(dt)
    P, Qr = _cost(horizon, block, input_weight)
    M, lo, hi, num_blocks = _constraints(A, B, horizon, block)

    n_eq = horizon * N_STATE
    rho_vec = np.full(M.shape[0], rho)
    rho_vec[:n_eq] *= rho_equality

    lhs = (P + sigma * sp.eye(P.shape[0], format="csc") + M.T @ sp.diags(rho_vec) @ M).tocsc()
    lu = spla.splu(lhs, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})

    return UncondensedProblem(
        dt=dt,
        horizon=horizon,
        block=block,
        sigma=sigma,
        A=A,
        B=B,
        P=P,
        M=M,
        Qr=Qr,
        rho=rho_vec,
        lo=lo,
        hi=hi,
        _solve=lu.solve,
        _system_nnz=int(lhs.nnz),
        _factor_nnz=int(lu.L.nnz + lu.U.nnz),
    )
