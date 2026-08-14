"""The condensed formulation as a controller the shared simulator can fly."""

from __future__ import annotations

import numpy as np

from common.quadcopter import N_INPUT
from common.reference import Reference
from common.simulate import Command
from condensed import admm
from condensed.mpc import MpcProblem, build_mpc


class CondensedController:
    """Solves the condensed program and applies the first input move.

    The measured state and the reference window are folded into the linear cost
    before the iteration starts, so the parameter never reaches the solver as
    anything but ``q``.  That is the property the generated solver relies on.
    """

    def __init__(self, problem: MpcProblem, ref: Reference):
        self.problem = problem
        self.ref = ref
        self.dt = problem.dt
        self.A = problem.A
        self.B = problem.B
        self.n = problem.n

    def describe(self) -> str:
        p = self.problem
        return (
            f"condensed program: {p.n} variables, {p.n} constraint rows "
            f"({p.num_blocks} input moves over a {p.lookahead:.2f} s horizon, "
            f"{p.horizon} steps of {1000 * p.dt:.0f} ms, held {p.block} at a time)\n"
            f"system matrix:     {p.n} x {p.n} dense, "
            f"{100 * np.count_nonzero(np.abs(p.P) > 1e-12) / p.P.size:.0f}% filled, "
            f"dense Cholesky\n"
            f"ADMM parameters:   rho = {p.rho}, sigma = {p.sigma}"
        )

    def command(self, state: np.ndarray, t: float, iterations: int) -> Command:
        q = self.problem.linear_cost(state, self.problem.reference_window(self.ref, t))
        result = admm.solve(self.problem, q, iterations)
        u = self.problem.first_input(result.z)
        return Command(
            u=u,
            residual=result.residual if result.residual is not None else np.nan,
            saturated=bool(np.any(np.abs(result.z[:N_INPUT] - u) > 1e-9)),
        )


def build(ref: Reference, **kwargs) -> CondensedController:
    """A controller on the default condensed program."""
    return CondensedController(build_mpc(**kwargs), ref)
