"""Linearized quadcopter model and its cost / constraint data.

The discrete-time matrices below are the OSQP documentation's quadcopter example
(a hover linearization of Bouffard's Crazyflie model), sampled at 0.1 s.  They
are copied verbatim from ``mip_algo_verify/experiments/MPC/quadcopter_compact.py``
so that this experiment and the verification experiments analyze the same plant.

The state is

    xi = (phi, theta, psi,  x, y, z,  omega_phi, omega_theta, omega_psi,  vx, vy, vz),

i.e. attitude, position, angular rates, linear velocities.  The ordering is not
documented upstream but is unambiguous from the structure of the matrices:
``B_DYN`` drives coordinate 5 equally from all four rotors (collective thrust
-> altitude), coordinate 2 antisymmetrically (differential torque -> yaw), and
``A_DYN`` integrates coordinates 9:12 into 3:6 (velocity -> position) while
coupling 0:2 into 3:6 (tilt -> horizontal motion).  Within that block,
coordinate 0 drives ``x`` (``A_DYN[3, 0] > 0``) and coordinate 1 drives ``y``
with the opposite sign (``A_DYN[4, 1] < 0``), so coordinate 0 is the
pitch-like angle (rotation about the body ``y`` axis) and coordinate 1 the
roll-like angle, whatever the letters above suggest; the renderer relies on
this.

The input is the vector of four rotor thrusts expressed as deviations from the
hover equilibrium ``U_HOVER``, so the origin of the state-input space is the
hover trim and the model carries no affine offset.

Since the sampling period of this experiment is not 0.1 s, we recover the
underlying continuous-time model and re-discretize.  Every eigenvalue of
``A_DYN`` is real and positive, so its principal matrix logarithm is real and
the recovery is exact to machine precision.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.linalg as sla

#: Discrete-time state matrix at the native 0.1 s sampling period.
A_DYN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0992, 0.0, 0.0],
        [0.0, -0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, -0.0016, 0.0, 0.0, 0.0992, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0992],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0488, 0.0, 0.0, 0.9846, 0.0, 0.0],
        [0.0, -0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0488, 0.0, 0.0, 0.9846, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9846],
    ]
)

#: Discrete-time input matrix at the native 0.1 s sampling period.
B_DYN = np.array(
    [
        [0.0, -0.0726, 0.0, 0.0726],
        [-0.0726, 0.0, 0.0726, 0.0],
        [-0.0152, 0.0152, -0.0152, 0.0152],
        [-0.0, -0.0006, -0.0, 0.0006],
        [0.0006, 0.0, -0.0006, 0.0],
        [0.0106, 0.0106, 0.0106, 0.0106],
        [0.0, -1.4512, 0.0, 1.4512],
        [-1.4512, 0.0, 1.4512, 0.0],
        [-0.3049, 0.3049, -0.3049, 0.3049],
        [-0.0, -0.0236, 0.0, 0.0236],
        [0.0236, 0.0, -0.0236, 0.0],
        [0.2107, 0.2107, 0.2107, 0.2107],
    ]
)

#: Sampling period the matrices above were discretized at.
DT_NATIVE = 0.1

N_STATE, N_INPUT = B_DYN.shape

#: Slices naming the physically meaningful blocks of the state.
ATT = slice(0, 3)
POS = slice(3, 6)
RATE = slice(6, 9)
VEL = slice(9, 12)

#: Hover thrust, in the same units as the rotor commands.  Inputs are deviations.
U_HOVER = 10.5916
U_MIN = np.full(N_INPUT, 9.6 - U_HOVER)
U_MAX = np.full(N_INPUT, 13.0 - U_HOVER)

#: Stage cost.  Penalizes yaw, position, and linear velocity; attitude and
#: angular rates are left free.  Verbatim from the verification experiments.
Q_STAGE = np.diag([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
R_INPUT = 0.1 * np.eye(N_INPUT)

#: Multiplier applied to ``Q_STAGE`` at the end of the horizon.
TERMINAL_FACTOR = 10.0

#: Bound on the two tilt angles, in radians.  The OSQP example's state box puts
#: ``+-pi/6`` on the first two coordinates (``xmin``/``xmax``) and nothing on
#: the rest.  A hover linearization is only meaningful near it: left free, the
#: tracking controller flies the linear model at 60 to 70 degrees of tilt.
TILT_LIMIT = np.pi / 6


def continuous_model() -> tuple[np.ndarray, np.ndarray]:
    """Recover the continuous-time model ``(A_c, B_c)`` behind ``A_DYN``, ``B_DYN``.

    The state matrix follows from the principal matrix logarithm.  The input
    matrix follows from the zero-order-hold relation
    ``B_DYN = (int_0^dt exp(A_c s) ds) B_c``, whose integral is read off the
    exponential of the augmented matrix ``[[A_c dt, I dt], [0, 0]]``.
    """
    log_a = sla.logm(A_DYN)
    imag = float(np.max(np.abs(log_a.imag))) if np.iscomplexobj(log_a) else 0.0
    if imag > 1e-12:
        raise ValueError(
            f"the matrix logarithm of A_DYN is not real (largest imaginary part {imag:.2e}); "
            "the continuous-time model cannot be recovered this way."
        )
    a_cont = np.asarray(log_a.real) / DT_NATIVE
    b_cont = np.linalg.solve(_zoh_integral(a_cont, DT_NATIVE), B_DYN)
    return a_cont, b_cont


def _zoh_integral(a_cont: np.ndarray, dt: float) -> np.ndarray:
    """The zero-order-hold integral ``int_0^dt exp(A_c s) ds``."""
    n = a_cont.shape[0]
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = a_cont * dt
    block[:n, n:] = np.eye(n) * dt
    return sla.expm(block)[:n, n:]


@lru_cache(maxsize=None)
def discretize(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretization of the quadcopter at sampling period ``dt``.

    Returns ``A_DYN``, ``B_DYN`` unchanged when ``dt`` is the native period, so
    that the default configuration uses the published matrices bit for bit.
    """
    if dt == DT_NATIVE:
        return A_DYN.copy(), B_DYN.copy()
    a_cont, b_cont = continuous_model()
    return sla.expm(a_cont * dt), _zoh_integral(a_cont, dt) @ b_cont
