"""The condensed ADMM iteration, recorded on a ``Verifier``.

One definition of the algorithm, shared by the equivalence check and by code
generation, so the thing that is generated is the thing that was checked.

This formulation is the easy one to generate.  The states are eliminated, so the
constraint matrix is the identity and the saturated-linear bounds are the
constant rotor-thrust box; the parameter enters only through the linear cost.
That is exactly the shape ``fomtools.codegen`` accepts, with no rewriting.  The
uncondensed formulation needs an exact reformulation to get there, and
``uncondensed/recording.py`` explains why.

The recording therefore carries a single parameter, the 40-long cost vector.  The
map from a measured state and a reference window to that vector is affine and is
applied once per sampling instant, outside the loop, by both implementations
alike.
"""

from __future__ import annotations

import numpy as np

from common.recording import ParameterBox, ParameterBoxes, Recording, box_around
from condensed.mpc import MpcProblem


def parameter_boxes(
    problem: MpcProblem,
    ref,
    duration: float = 8.0,
    iterations: int = 1000,
    margin: float = 0.05,
) -> ParameterBoxes:
    """Fly once, well converged, and take the box of costs that flight visits."""
    from common.simulate import closed_loop
    from condensed.controller import CondensedController

    roll = closed_loop(CondensedController(problem, ref), ref, iterations, duration=duration)
    costs = np.vstack(
        [
            problem.linear_cost(roll.states[k], problem.reference_window(ref, k * problem.dt))
            for k in range(roll.steps)
        ]
    )
    lb, ub = box_around(costs, margin)
    mid = roll.steps // 2
    return ParameterBoxes(
        boxes=[ParameterBox(name="p0", lb=lb, ub=ub, mid=costs[mid])],
        state=roll.states[mid],
    )


def record(problem: MpcProblem, boxes: ParameterBoxes, iterations: int = 3) -> Recording:
    """Record ``iterations`` steps of the iteration in :mod:`condensed.admm`.

    Only the first is needed to generate from; the rest let
    :func:`fomtools.codegen.oracle.check_stationary` confirm that repeating one
    iteration reproduces what was recorded, which is the assumption extraction
    makes and the only thing that can catch it being false.
    """
    from fomtools.verifier import Verifier

    n = problem.n
    rho, sigma = problem.rho, problem.sigma
    lhs = problem.P + (sigma + rho) * np.eye(n)

    verifier = Verifier(method="simulator", obbt_solver=None)
    cost = verifier.add_param(n, lb=boxes["p0"].lb, ub=boxes["p0"].ub)
    z = verifier.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    v = verifier.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))

    state = [z, v]
    iterates = []
    for _ in range(iterations):
        w = verifier.step.satlin(v, l=problem.lo, u=problem.hi)
        z = verifier.step.implicit_linear(sigma * z - cost + rho * (2.0 * w - v), output_mat=lhs)
        v = v + z - w
        iterates.append([z, v])

    return Recording(
        verifier=verifier,
        next_state=iterates[0],
        state=state,
        params=[cost],
        iterates=iterates,
    )
