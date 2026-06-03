"""Reference numerical execution of a FOM, used as ground truth.

Two evaluators live here:

* :class:`ReferenceOracle` runs the **rolled** loop directly from a
  :class:`~fomtools.codegen.ir.FomIR`. This is the truth that emitted C code is
  compared against by the verify harness.
* :func:`run_faithful` runs the same loop straight from the ``Verifier``'s step
  objects (solving the affine system rather than reusing its folded inverse).
  Agreement between the two validates that IR extraction preserved the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fomtools.codegen.config import TerminationKind
from fomtools.codegen.extract import (
    _eval_step_fresh,
    eval_expr,
    resolve_structure,
)
from fomtools.codegen.ir import ActivationKind, ActivationOp, FomIR, LinearOp


@dataclass
class OracleResult:
    """Outcome of a reference solve."""

    solution: np.ndarray
    iters: int
    residual: float
    trajectory: list[np.ndarray]
    residuals: list[float]


def _apply_activation(act: ActivationOp, x: np.ndarray, ir: FomIR) -> np.ndarray:
    if act.kind == ActivationKind.IDENTITY:
        return x
    if act.kind == ActivationKind.SOFT_THRESHOLD:
        return np.sign(x) * np.maximum(np.abs(x) - act.lambd, 0.0)
    if act.kind == ActivationKind.RELU:
        return np.maximum(0.0, x)
    if act.kind == ActivationKind.SATURATED_LINEAR:
        lower = act.lower_scalar if act.lower is None else ir.matrices[act.lower].array
        upper = act.upper_scalar if act.upper is None else ir.matrices[act.upper].array
        return np.minimum(np.maximum(x, lower), upper)
    raise ValueError(f"Unknown activation kind: {act.kind}")


class ReferenceOracle:
    """Rolled NumPy executor of a :class:`FomIR`."""

    def __init__(self, ir: FomIR):
        self.ir = ir

    def _eval_linop(self, lin: LinearOp, sources: dict[str, np.ndarray]) -> np.ndarray:
        if lin.const is not None:
            out = self.ir.matrices[lin.const].array.astype(float).reshape(-1).copy()
        else:
            out = np.zeros(lin.out_dim)
        for term in lin.terms:
            mat = self.ir.matrices[term.matrix].array
            out = out + np.asarray(mat @ sources[term.source]).reshape(-1)
        return out

    def solve(
        self,
        param_values: dict[str, np.ndarray] | None = None,
        x0: np.ndarray | None = None,
    ) -> OracleResult:
        """Run the rolled loop.

        Parameters
        ----------
        param_values:
            Mapping from parameter name (``"p0"`` …) to its value vector.
        x0:
            Optional initial state; defaults to the IR's stored init.
        """
        ir = self.ir
        param_values = param_values or {}
        state = (
            np.array(ir.state.init, dtype=float)
            if x0 is None
            else np.asarray(x0, dtype=float).copy()
        )

        sources: dict[str, np.ndarray] = {"state": state}
        for p in ir.params:
            if p.name not in param_values:
                raise ValueError(f"Missing value for parameter {p.name!r}")
            sources[f"param:{p.name}"] = np.asarray(param_values[p.name], dtype=float).reshape(-1)

        trajectory = [state.copy()]
        residuals: list[float] = []
        term = ir.termination
        iters = 0
        for _ in range(term.max_iters):
            for step in ir.body:
                lin_val = self._eval_linop(step.linear_op, sources)
                if step.solve is not None:
                    # Affine step: solve M x = rhs via the factorization (the
                    # inverse is never formed), matching the emitted C exactly.
                    val = ir.factorizations[step.solve].solve(lin_val)
                else:
                    val = _apply_activation(step.activation, lin_val, ir)
                sources[f"tmp:{step.name}"] = val
            new_state = self._eval_linop(ir.state_update, sources)
            residual = float(np.max(np.abs(new_state - state))) if new_state.size else 0.0
            state = new_state
            sources["state"] = state
            trajectory.append(state.copy())
            residuals.append(residual)
            iters += 1
            if term.kind == TerminationKind.INF_NORM_CHANGE and residual <= term.tol:
                break

        return OracleResult(
            solution=state,
            iters=iters,
            residual=residuals[-1] if residuals else 0.0,
            trajectory=trajectory,
            residuals=residuals,
        )


def run_faithful(
    verifier,
    next_state,
    param_values: dict,
    n_iters: int,
    state=None,
    params=None,
    x0: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Run ``n_iters`` of the FOM straight from the verifier's step objects.

    ``param_values`` maps each parameter *leaf* to its value vector. Returns the
    state trajectory (including the initial state). Used to cross-check the IR.
    """
    next_state_exprs, body_steps, body_fresh, state_leaves, param_leaves = resolve_structure(
        verifier, next_state, state, params
    )

    # initial state vector
    slots = []
    offset = 0
    for leaf in state_leaves:
        slots.append((leaf, offset, leaf.ndim))
        offset += leaf.ndim
    state_dim = offset
    if x0 is None:
        state_vec = np.zeros(state_dim)
        for leaf, off, dim in slots:
            lb = np.asarray(leaf.lb, dtype=float).reshape(-1)
            ub = np.asarray(leaf.ub, dtype=float).reshape(-1)
            fixed = np.isfinite(lb) & np.isfinite(ub) & np.isclose(lb, ub)
            state_vec[off : off + dim] = np.where(fixed, lb, 0.0)
    else:
        state_vec = np.asarray(x0, dtype=float).copy()

    trajectory = [state_vec.copy()]
    for _ in range(n_iters):
        values: dict = {}
        for leaf, off, dim in slots:
            values[leaf] = state_vec[off : off + dim]
        for leaf in param_leaves:
            values[leaf] = np.asarray(param_values[leaf], dtype=float).reshape(-1)
        for step, fresh in zip(body_steps, body_fresh):
            values[fresh] = _eval_step_fresh(step, values)
        pieces = [eval_expr(e, values) for e in next_state_exprs]
        state_vec = np.concatenate(pieces) if pieces else np.zeros(0)
        trajectory.append(state_vec.copy())

    return trajectory
