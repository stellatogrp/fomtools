"""The rAPDHG solver as a controller the shared simulator can fly."""

from __future__ import annotations

import numpy as np

from common.quadcopter import N_INPUT
from common.reference import Reference
from common.simulate import Command
from condensed import rapdhg
from condensed.mpc import RapdhgProblem, build_mpc


class RapdhgController:
    """Solves the condensed program with rAPDHG and applies the first input move.

    The measured state and the reference window are folded into the linear cost
    ``q`` and the constraint right-hand side ``h`` before the iteration starts,
    so the parameter reaches the solver only as those two vectors.  That is the
    property the generated solver relies on, and it is why the condensed
    formulation is the one this study generates from.

    The schedule tables are built once, for the largest iteration count the
    controller will be asked for, and sliced per solve.  Rebuilding them inside
    ``command`` would charge the iteration for work the generated solver does at
    compile time.
    """

    def __init__(self, problem: RapdhgProblem, ref: Reference, max_iterations: int = 8192):
        self.problem = problem
        self.ref = ref
        self.dt = problem.dt
        self.A = problem.A
        self.B = problem.B
        self.n = problem.n
        self.max_iterations = max_iterations
        self.tables = rapdhg.schedule_tables(problem, max_iterations)

    def describe(self) -> str:
        p = self.problem
        return (
            f"condensed program: {p.n} variables, {p.m} inequality rows "
            f"({p.num_blocks} input moves over a {p.lookahead:.2f} s horizon, "
            f"{p.horizon} steps of {1000 * p.dt:.0f} ms, held {p.block} at a time)\n"
            f"iteration:         matrix free; one {p.n} x {p.n} product per step, "
            f"no linear solve\n"
            f"rAPDHG parameters: K = {p.restart}, w = {p.primal_weight}, "
            f"||Q|| = {p.norm_q:.4g}, ||G|| = {p.norm_g:.4g}\n"
            + (
                f"state constraints: |tilt| <= {np.degrees(p.e[0]):.0f} deg at the end of "
                f"each block, {2 * p.num_state_rows} rows"
                if p.num_state_rows
                else "state constraints: none"
            )
        )

    def command(self, state: np.ndarray, t: float, iterations: int) -> Command:
        if iterations > self.max_iterations:
            raise ValueError(
                f"{iterations} iterations requested but the schedule tables only cover "
                f"{self.max_iterations}; raise max_iterations"
            )
        q = self.problem.linear_cost(state, self.problem.reference_window(self.ref, t))
        h = self.problem.rhs(state)
        result = rapdhg.solve(self.problem, q, iterations, tables=self.tables, h=h)
        u = self.problem.first_input(result.x)
        return Command(
            u=u,
            residual=rapdhg.relkkt(self.problem, q, result.x, result.y, h=h),
            saturated=bool(np.any(np.abs(result.x[:N_INPUT] - u) > 1e-9)),
        )


def build(ref: Reference, **kwargs) -> RapdhgController:
    """A controller on the default condensed program."""
    max_iterations = kwargs.pop("max_iterations", 8192)
    return RapdhgController(build_mpc(**kwargs), ref, max_iterations=max_iterations)
