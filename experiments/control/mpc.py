"""Condensed model predictive control problem for the quadcopter.

Eliminating the states through the dynamics turns the tracking problem into a
box-constrained quadratic program in the input sequence alone,

    minimize    (1/2) z' P z + q(x, r)' z
    subject to  l <= z <= u,

where ``z`` collects the (blocked) input moves, the bounds are the constant
rotor-thrust box, and the parameter enters only through the linear cost.  That
last property is what makes the family suitable for code generation: the
saturated-linear step of the algorithm then has constant bounds, and the matrix
of the linear system is fixed at generation time and factorized once.

Two devices keep the program small without shortening the horizon.  The states
are eliminated, and the input is held constant over blocks of consecutive
sampling instants (move blocking).  A one second lookahead at a 10 ms sampling
period is 100 steps, which would give 396 variables; holding the input over
blocks of ten gives 40.  The distinction matters for what this experiment
measures: at 40 variables an interpreted iteration is dominated by per-operation
overhead, which is exactly the cost that generated code removes, whereas at 396
variables both implementations spend their time inside the same dense triangular
solve.

The linear cost is
    q(x, r) = G_x x - G_r r,
affine in the measured state ``x`` and in the reference window ``r``.  The
generated solver takes ``q`` itself as its parameter and this map is applied
outside the iteration loop, once per sampling instant, so the timings compare
like with like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
from quadcopter import (
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
class MpcProblem:
    """A condensed MPC quadratic program together with its ADMM data."""

    dt: float
    horizon: int  # number of predicted states, x_1 ... x_horizon
    block: int  # sampling instants over which the input is held constant
    rho: float
    sigma: float

    A: np.ndarray  # discrete-time state matrix at this dt
    B: np.ndarray  # discrete-time input matrix at this dt

    P: np.ndarray  # Hessian of the condensed program
    Gx: np.ndarray  # linear-cost map from the measured state
    Gr: np.ndarray  # linear-cost map from the reference window
    lo: np.ndarray  # lower thrust bound, stacked over blocks
    hi: np.ndarray  # upper thrust bound, stacked over blocks

    #: Cholesky factor of P + sigma I + rho I, the matrix of the affine step.
    #: The constraint matrix is the identity, so this is the whole system.
    _chol: tuple = field(repr=False)

    @property
    def n(self) -> int:
        """Number of optimization variables."""
        return self.P.shape[0]

    @property
    def num_blocks(self) -> int:
        """Number of independent input moves."""
        return self.n // N_INPUT

    @property
    def lookahead(self) -> float:
        """Prediction horizon in seconds."""
        return self.horizon * self.dt

    def linear_cost(self, state: np.ndarray, ref_window: np.ndarray) -> np.ndarray:
        """The parameter ``q`` of the quadratic program at one sampling instant."""
        return self.Gx @ state - self.Gr @ ref_window

    def reference_window(self, ref, t: float) -> np.ndarray:
        """Stack the reference over the prediction horizon starting at time ``t``.

        Only the coordinates the stage cost weights are filled: the position and
        the linear velocity.  Everything else is referenced to zero, which for
        the attitude and the angular rates is the hover trim.
        """
        window = np.zeros(self.horizon * N_STATE)
        for i in range(self.horizon):
            pos, vel = ref(t + i * self.dt)
            base = i * N_STATE
            window[base + POS.start : base + POS.stop] = pos
            window[base + VEL.start : base + VEL.stop] = vel
        return window

    def solve_system(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the precomputed factorization of ``P + sigma I + rho I``."""
        return sla.cho_solve(self._chol, rhs)

    def first_input(self, z: np.ndarray) -> np.ndarray:
        """The rotor thrusts to apply now, saturated at the actuator limits.

        The iterate ``z`` need not be feasible after a finite number of
        iterations, so the actuator saturates it.  A physical actuator does the
        same, and it keeps a badly converged solve from being reported as a
        constraint violation that the hardware would never have committed.
        """
        return np.clip(z[:N_INPUT], U_MIN, U_MAX)


def _blocking_matrix(num_moves: int, block: int) -> np.ndarray:
    """Map blocked moves to the per-step input sequence.

    Column block ``j`` is the identity on every step assigned to move ``j``, so
    the product with a stacked move vector repeats each move ``block`` times.
    """
    num_blocks = int(np.ceil(num_moves / block))
    mat = np.zeros((num_moves * N_INPUT, num_blocks * N_INPUT))
    for step in range(num_moves):
        j = step // block
        mat[step * N_INPUT : (step + 1) * N_INPUT, j * N_INPUT : (j + 1) * N_INPUT] = np.eye(
            N_INPUT
        )
    return mat


def _prediction_matrices(A: np.ndarray, B: np.ndarray, horizon: int) -> tuple:
    """Stacked free and forced responses over the horizon.

    Returns ``(S_x, S_v)`` with ``X = S_x x_1 + S_v z``, where ``X`` stacks
    ``x_1 ... x_horizon`` and ``z`` stacks ``u_1 ... u_{horizon-1}``.
    """
    powers = [np.eye(N_STATE)]
    for _ in range(horizon - 1):
        powers.append(A @ powers[-1])

    s_x = np.vstack(powers)
    s_v = np.zeros((horizon * N_STATE, (horizon - 1) * N_INPUT))
    for t in range(1, horizon):
        for j in range(t):
            s_v[t * N_STATE : (t + 1) * N_STATE, j * N_INPUT : (j + 1) * N_INPUT] = (
                powers[t - 1 - j] @ B
            )
    return s_x, s_v


def build_mpc(
    dt: float = 0.01,
    lookahead: float = 1.0,
    block: int = 10,
    rho: float = 0.1,
    sigma: float = 1e-4,
    input_weight: float = 1.0,
) -> MpcProblem:
    """Assemble the condensed MPC program.

    Parameters
    ----------
    dt:
        Sampling period in seconds.  The controller re-solves at this rate.
    lookahead:
        Prediction horizon in seconds.  The horizon in steps is ``lookahead / dt``.
    block:
        Number of consecutive sampling instants over which the input is held.
    rho, sigma:
        The two ADMM parameters.  The defaults are the OSQP default penalty and
        the regularization used in the verification experiments; neither is
        tuned to this instance.
    input_weight:
        Multiplier on ``R_INPUT``, provided as a knob and left at one, so the
        cost is the published one.  Tracking accuracy turns out to be almost
        insensitive to it over two decades, because the thrust box rather than
        the input penalty is what limits the flight; lowering it only worsens
        the conditioning of ``P``.
    """
    horizon = int(round(lookahead / dt))
    if horizon < 2:
        raise ValueError(f"lookahead {lookahead} s at dt {dt} s gives fewer than two steps")

    A, B = discretize(dt)
    s_x, s_v = _prediction_matrices(A, B, horizon)

    E = _blocking_matrix(horizon - 1, block)
    s_v = s_v @ E

    # The stage cost is diagonal, so the stacked weight is a vector and the
    # products below are scalings rather than dense multiplications.
    q_diag = np.tile(np.diag(Q_STAGE), horizon)
    q_diag[(horizon - 1) * N_STATE :] *= TERMINAL_FACTOR
    r_block = input_weight * (E.T @ np.kron(np.eye(horizon - 1), R_INPUT) @ E)

    weighted = s_v.T * q_diag  # S_v' Qbar, reused three times below
    P = weighted @ s_v + r_block
    P = 0.5 * (P + P.T)  # symmetrize away the round-off of the products
    Gx = weighted @ s_x
    Gr = weighted

    n = P.shape[0]
    lhs = P + (sigma + rho) * np.eye(n)
    chol = sla.cho_factor(lhs)

    num_blocks = n // N_INPUT
    return MpcProblem(
        dt=dt,
        horizon=horizon,
        block=block,
        rho=rho,
        sigma=sigma,
        A=A,
        B=B,
        P=P,
        Gx=Gx,
        Gr=Gr,
        lo=np.tile(U_MIN, num_blocks),
        hi=np.tile(U_MAX, num_blocks),
        _chol=chol,
    )
