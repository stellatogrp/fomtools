"""One rAPDHG iteration, recorded on a ``Verifier`` for code generation.

The obstacle this module works around is that the recording language has no
restart.  There is no nested loop and no branch: extraction reads a straight
line of steps and rolls it into one repeating body.  Unrolling the K inner steps
is not an option either -- the term count grows like K squared, and
``extract_unrolled`` rejects it outright.

What makes the method expressible is that its coefficients are *tables*.  The
inner index ``k`` runs ``0 .. K-1`` and resets, so every coefficient is periodic
in the global iteration counter, and ``fomtools.codegen.schedule`` accepts any
function of that counter.  So one recorded iteration is one *inner* step, and
the restart is a coefficient that is nonzero on every K-th iteration:

    x_next = (1 - r) xhat + r xbar,     r = 1 when k = K-1, else 0.

K never appears in the size of the emitted code.  It only sets numbers in the
tables, which is why a restart period of tens or hundreds costs the same as one.

Two properties of rAPDHG make the scheduled path apply where it would not for
ADMM.  There is no factorization, so the check that a scheduled system matrix
would fail cannot fire; and the projection is a ``relu``, which carries no
scalar constants, so the check that a scheduled activation threshold would fail
cannot fire either.

Two details are load bearing.

*The gradient step is materialized.*  ``xhat`` is wrapped in a saturated-linear
step with bounds so wide they never bind.  Extraction folds every expression
that is not a step output into each place it is used, and ``xhat`` is used by
three state slots; without materializing it, the one expensive product in the
iteration, ``Q @ xmd``, would be emitted three times.  With it, the emitted
iteration applies ``Q`` exactly once.  ``check_against_fomtools.py`` asserts the
saturation never binds, so it changes the arithmetic not at all.

*The right-hand side is a parameter.*  The dual step needs ``tau_k * h``, and a
scheduled constant offset is rejected -- only matrix coefficients can carry a
schedule -- so ``h`` is declared as a parameter, which turns it into a scheduled
coefficient on a parameter, which is allowed.  With the tilt bound it is a
genuine parameter anyway: the thrust rows of ``h`` are constant, but the tilt
rows are the limit less the free response from the measured state, and they
move every sampling instant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from common.recording import ParameterBox, ParameterBoxes, box_around
from condensed import rapdhg
from condensed.mpc import RapdhgProblem

#: Bound of the materializing saturation.  Large enough never to bind -- the
#: iterates of this program live inside a thrust box of half-width 2.4 -- and
#: finite so the step is an ordinary activation with constant arrays.
MATERIALIZE_BOUND = 1e6


def parameter_boxes(
    problem: RapdhgProblem,
    ref,
    duration: float = 8.0,
    iterations: int = 4096,
    margin: float = 0.05,
) -> ParameterBoxes:
    """Fly once, well converged, and take the boxes of parameters that flight visits.

    Two boxes: the linear cost ``q`` and the right-hand side ``h``.  The thrust
    rows of ``h`` never move, so that box is degenerate on them; the tilt rows
    follow the measured state, so on those it has width.
    """
    from common.simulate import closed_loop
    from condensed.controller import RapdhgController

    controller = RapdhgController(problem, ref, max_iterations=iterations)
    roll = closed_loop(controller, ref, iterations, duration=duration)
    costs = np.vstack(
        [
            problem.linear_cost(roll.states[k], problem.reference_window(ref, k * problem.dt))
            for k in range(roll.steps)
        ]
    )
    rhs = np.vstack([problem.rhs(roll.states[k]) for k in range(roll.steps)])
    lb, ub = box_around(costs, margin)
    rhs_lb, rhs_ub = box_around(rhs, margin)
    mid = roll.steps // 2
    return ParameterBoxes(
        boxes=[
            ParameterBox(name="p0", lb=lb, ub=ub, mid=costs[mid]),
            ParameterBox(name="p1", lb=rhs_lb, ub=rhs_ub, mid=rhs[mid]),
        ],
        state=roll.states[mid],
    )


def make_build(problem: RapdhgProblem, boxes: ParameterBoxes) -> Callable[..., Any]:
    """Return the build function ``extract_scheduled_ir`` probes.

    It is called once per declared schedule plus twice more, at generic probe
    values, and every coefficient block is fitted as an affine function of those
    scalars.  So each scalar has to enter the expressions below *linearly*: any
    product of two of the paper's coefficients is a separate table, computed in
    :func:`condensed.rapdhg.schedule_tables`, never a product written here.
    """
    from fomtools.codegen.extract_scheduled import Recording
    from fomtools.verifier import Verifier

    n, m = problem.n, problem.m
    Q = np.asarray(problem.P)
    G = np.asarray(problem.G)
    Gt = G.T
    big_lo = np.full(n, -MATERIALIZE_BOUND)
    big_hi = np.full(n, MATERIALIZE_BOUND)
    cost_box, rhs_box = boxes["p0"], boxes["p1"]

    def build(taux, tth, tau, eta, g1, g2, d1, d2, m1, m2) -> Recording:
        v = Verifier(method="simulator", obbt_solver=None)
        cost = v.add_param(n, lb=cost_box.lb, ub=cost_box.ub)
        rhs = v.add_param(m, lb=rhs_box.lb, ub=rhs_box.ub)

        x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
        y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
        xbar = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
        ybar = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
        x_prev = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
        xmd = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))

        # Dual step.  tau G (theta (x - x_prev) + x) - tau h, with the product
        # tau * theta carried by its own table.
        y_new = v.step.relu(y + (taux * G) @ x - (tth * G) @ x_prev - tau * rhs)

        # Primal step, materialized so Q is applied once; see the module docstring.
        x_new = v.step.satlin(
            x - (eta * Q) @ xmd - eta * cost - (eta * Gt) @ y_new,
            l=big_lo,
            u=big_hi,
        )

        return Recording(
            verifier=v,
            next_state=[
                g1 * x_new + g2 * xbar,  # restart mux
                g1 * y_new + g2 * ybar,
                d1 * xbar + d2 * x_new,  # running averages
                d1 * ybar + d2 * y_new,
                1.0 * x,  # x_prev
                m1 * xbar + m2 * x_new,  # xmd, built for the next inner index
            ],
            state=[x, y, xbar, ybar, x_prev, xmd],
            params=[cost, rhs],
        )

    return build


def schedules(problem: RapdhgProblem, max_iters: int) -> dict[str, np.ndarray]:
    """The tables the generated loop reads, one value per iteration.

    The same tables :func:`condensed.rapdhg.solve` runs on, so the NumPy arm and
    the generated arms are the same recurrence by construction rather than by
    inspection.
    """
    return rapdhg.schedule_tables(problem, max_iters)
