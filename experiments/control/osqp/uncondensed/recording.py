"""The uncondensed ADMM iteration, recorded on a ``Verifier``.

One definition of the algorithm, shared by the equivalence check and by code
generation, so the thing that is generated is the thing that was checked.

The formulation puts the measured state in the constraint bounds: ``xi_1 = x`` is
an equality, and an equality is a bound whose two sides coincide.  Code
generation stores the clip bounds of a saturated-linear step as constant arrays
(``ir.ActivationOp.lower`` names an entry of ``FomIR.matrices``), so a bound that
moves with the parameter has nowhere to live and ``extract._satlin_arrays``
rejects it.  The identity

    sat_[l, u](y) = l + sat_[0, u - l](y - l)

moves the parameter out of the bounds and into the linear expression the step is
applied to, which code generation accepts.  It is exact, and it is cheap here:
the width ``u - l`` is constant everywhere, zero on the equality rows and the
thrust range on the input box, and only the twelve rows that pin the initial
state move at all.

The recording therefore carries two parameters, the linear cost and the measured
state, and no frozen data.  What is generated from it is a solver that flies.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from common.quadcopter import N_STATE
from common.recording import ParameterBox, ParameterBoxes, Recording, box_around
from uncondensed.mpc import UncondensedProblem


def state_selector(m: int) -> sp.csc_matrix:
    """Place a measured state on the rows of the bounds that pin ``xi_1``."""
    return sp.vstack(
        [sp.eye(N_STATE, format="csc"), sp.csc_matrix((m - N_STATE, N_STATE))], format="csc"
    )


def system_matrix(problem: UncondensedProblem) -> sp.csc_matrix:
    """The matrix the affine step solves against, ``P + sigma I + M\' R M``."""
    M = problem.M.tocsc()
    R = sp.diags(problem.rho)
    return (problem.P + problem.sigma * sp.eye(problem.n, format="csc") + M.T @ R @ M).tocsc()


def parameter_boxes(
    problem: UncondensedProblem,
    ref,
    duration: float = 8.0,
    iterations: int = 1000,
    margin: float = 0.05,
) -> ParameterBoxes:
    """Fly once, well converged, and take the boxes that flight visits.

    Two parameters here: the linear cost, which depends on the reference window
    alone, and the measured state, which reaches the bounds.
    """
    from common.simulate import closed_loop
    from uncondensed.controller import UncondensedController

    roll = closed_loop(UncondensedController(problem, ref), ref, iterations, duration=duration)
    costs = np.vstack(
        [
            problem.linear_cost(problem.reference_window(ref, k * problem.dt))
            for k in range(roll.steps)
        ]
    )
    states = roll.states[: roll.steps]
    mid = roll.steps // 2

    q_lb, q_ub = box_around(costs, margin)
    x_lb, x_ub = box_around(states, margin)
    return ParameterBoxes(
        boxes=[
            ParameterBox(name="p0", lb=q_lb, ub=q_ub, mid=costs[mid]),
            ParameterBox(name="p1", lb=x_lb, ub=x_ub, mid=states[mid]),
        ],
        state=states[mid],
    )


def record(problem: UncondensedProblem, boxes: ParameterBoxes, iterations: int = 3) -> Recording:
    """Record ``iterations`` steps of the iteration in :mod:`uncondensed.admm`.

    Only the first is needed to generate from; the rest let
    :func:`fomtools.codegen.oracle.check_stationary` confirm that repeating one
    iteration reproduces what was recorded, which is the assumption extraction
    makes and the only thing that can catch it being false.
    """
    from fomtools.verifier import Verifier

    n, m = problem.n, problem.m
    M = problem.M.tocsc()
    MtR = (M.T @ sp.diags(problem.rho)).tocsc()
    lhs = system_matrix(problem)

    # Constant width of the box, and the parameter-dependent lower bound.  The
    # width is zero wherever the row is an equality, so the step returns the
    # bound itself there, which is what an equality asks for.
    width = problem.hi - problem.lo
    selector = state_selector(m)

    verifier = Verifier(method="simulator", obbt_solver=None)
    cost = verifier.add_param(n, lb=boxes["p0"].lb, ub=boxes["p0"].ub)
    x = verifier.add_param(N_STATE, lb=boxes["p1"].lb, ub=boxes["p1"].ub)
    zeta = verifier.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    nu = verifier.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

    lower = selector @ x + problem.lo
    state = [zeta, nu]
    iterates = []
    for _ in range(iterations):
        w = verifier.step.satlin(nu - lower, l=np.zeros(m), u=width) + lower
        zeta = verifier.step.implicit_linear(
            problem.sigma * zeta - cost + MtR @ (2.0 * w - nu), output_mat=lhs
        )
        nu = nu + M @ zeta - w
        iterates.append([zeta, nu])

    return Recording(
        verifier=verifier,
        next_state=iterates[0],
        state=state,
        params=[cost, x],
        iterates=iterates,
    )
