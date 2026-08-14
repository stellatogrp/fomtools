"""The uncondensed formulation as a controller the shared simulator can fly."""

from __future__ import annotations

import numpy as np

from common.quadcopter import N_INPUT
from common.reference import Reference
from common.simulate import Command
from uncondensed import admm
from uncondensed.mpc import UncondensedProblem, build_mpc


class UncondensedController:
    """Solves the full-space program and applies the first input move."""

    def __init__(self, problem: UncondensedProblem, ref: Reference):
        self.problem = problem
        self.ref = ref
        self.dt = problem.dt
        self.A = problem.A
        self.B = problem.B
        self.n = problem.n

    def describe(self) -> str:
        p = self.problem
        return (
            f"uncondensed program: {p.n} variables, {p.m} constraint rows "
            f"({p.num_blocks} input moves over a {p.lookahead:.2f} s horizon, "
            f"{p.horizon} steps of {1000 * p.dt:.0f} ms, held {p.block} at a time)\n"
            f"system matrix:       {p.n} x {p.n} sparse, {p.system_nnz} stored entries "
            f"({100 * p.system_nnz / p.n**2:.2f}% filled), factor holds {p.factor_nnz}\n"
            f"ADMM parameters:     rho = {p.rho[-1]} on the input box, "
            f"{p.rho[0]} on the equalities, sigma = {p.sigma}"
        )

    def command(self, state: np.ndarray, t: float, iterations: int) -> Command:
        q = self.problem.linear_cost(self.problem.reference_window(self.ref, t))
        lo, hi = self.problem.bounds(state)
        result = admm.solve(self.problem, q, lo, hi, iterations)
        u = self.problem.first_input(result.zeta)
        start = self.problem.input_offset
        raw = result.zeta[start : start + N_INPUT]
        return Command(
            u=u,
            residual=result.residual if result.residual is not None else np.nan,
            saturated=bool(np.any(np.abs(raw - u) > 1e-9)),
        )


def build(ref: Reference, **kwargs) -> UncondensedController:
    """A controller on the default uncondensed program."""
    return UncondensedController(build_mpc(**kwargs), ref)
