"""Reference numerical execution of a FOM, used as ground truth.

Three evaluators live here:

* :class:`ReferenceOracle` runs the **rolled** loop directly from a
  :class:`~fomtools.codegen.ir.FomIR`. This is the truth that emitted C code is
  compared against by the verify harness.
* :func:`run_faithful` runs the same loop straight from the ``Verifier``'s step
  objects (solving the affine system rather than reusing its folded inverse).
  Agreement between the two validates that IR extraction preserved the algorithm.
* :func:`run_recorded_tape` replays the verifier's **unrolled** tape once, in
  recorded order. Unlike the other two it is not rolled, so it is the only thing
  here that can disagree with a wrongly-rolled IR — which makes it the ground
  truth behind :func:`check_stationary`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from fomtools.codegen.config import TerminationKind, TerminationSpec
from fomtools.codegen.extract import (
    _as_list,
    _eval_step_fresh,
    eval_expr,
    fresh_leaf_of,
    normalize_iterates,
    resolve_structure,
)
from fomtools.codegen.ir import (
    ActivationKind,
    ActivationOp,
    FomIR,
    LinearOp,
    NonStationaryFOMError,
    UnrolledFidelityError,
)


@dataclass
class OracleResult:
    """Outcome of a reference solve.

    ``residual`` is what the generated solver reports in ``solution.residual``:
    the infinity-norm change of the last iteration, or, under
    ``TerminationKind.OBJECTIVE``, the verification objective at the returned
    iterate. ``residuals`` always holds the per-iteration infinity-norm change,
    and ``objectives`` the ``(k, r^k)`` pairs at every iteration the objective
    was tested. ``solved`` mirrors ``FOM_SOLVED``: the loop exited on its
    tolerance rather than at ``max_iter``.
    """

    solution: np.ndarray
    iters: int
    residual: float
    trajectory: list[np.ndarray]
    residuals: list[float]
    objectives: list[tuple[int, float]] = field(default_factory=list)
    solved: bool = False


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


def objective_norm(values: np.ndarray, norm: str) -> float:
    """The verifier's norm of an objective vector (``"2"`` is the sum of squares)."""
    if values.size == 0:
        return 0.0
    if norm == "inf":
        return float(np.max(np.abs(values)))
    if norm == "1":
        return float(np.sum(np.abs(values)))
    return float(np.sum(values * values))


class ReferenceOracle:
    """Rolled NumPy executor of a :class:`FomIR`."""

    def __init__(self, ir: FomIR):
        self.ir = ir

    def _eval_linop(self, lin: LinearOp, sources: dict[str, np.ndarray], it: int = 0) -> np.ndarray:
        if lin.const is not None:
            out = self.ir.matrices[lin.const].array.astype(float).reshape(-1).copy()
        else:
            out = np.zeros(lin.out_dim)
        for term in lin.terms:
            mat = self.ir.matrices[term.matrix].array
            contrib = np.asarray(mat @ sources[term.source]).reshape(-1)
            if term.scale is not None:
                # Scheduled term: the emitted C reads the same table at the same
                # 0-based loop counter.
                contrib = self.ir.schedules[term.scale][it] * contrib
            out = out + contrib
        return out

    def objective_value(self, sources: dict[str, np.ndarray], it: int) -> float:
        """The verification objective ``r^k`` at the end of 0-based iteration ``it``.

        ``sources`` must hold ``state`` (``w^{k-1}``), ``new_state`` (``w^k``), the
        parameters and that iteration's temporaries.
        """
        obj = self.ir.objective
        rows = [
            _apply_activation(
                block.activation, self._eval_linop(block.linear_op, sources, it), self.ir
            )
            for block in obj.blocks
        ]
        return objective_norm(np.concatenate(rows) if rows else np.zeros(0), obj.norm)

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
        objectives: list[tuple[int, float]] = []
        solved = False
        term = ir.termination
        # Early exit only on "check" iterations: every `check_termination`
        # iterations (0 disables checking), mirroring the generated C exactly.
        # The objective is also tested at the last iteration.
        check = term.check_termination
        use_tol = term.uses_tolerance
        use_objective = term.uses_objective
        iters = 0
        for it in range(term.max_iters):
            for step in ir.body:
                lin_val = self._eval_linop(step.linear_op, sources, it)
                if step.solve is not None:
                    # Affine step: solve M x = rhs via the factorization (the
                    # inverse is never formed), matching the emitted C exactly.
                    val = ir.factorizations[step.solve].solve(lin_val)
                else:
                    val = _apply_activation(step.activation, lin_val, ir)
                sources[f"tmp:{step.name}"] = val
            new_state = self._eval_linop(ir.state_update, sources, it)
            residual = float(np.max(np.abs(new_state - state))) if new_state.size else 0.0
            objective = None
            if use_objective and term.is_check_iteration(it + 1):
                # Before the state advances: the objective reads w^{k-1} and w^k.
                sources["new_state"] = new_state
                objective = self.objective_value(sources, it)
                objectives.append((it + 1, objective))
            state = new_state
            sources["state"] = state
            trajectory.append(state.copy())
            residuals.append(residual)
            iters += 1
            if use_tol and (it + 1) % check == 0 and residual <= term.tol:
                solved = True
                break
            if objective is not None and objective <= term.tol:
                solved = True
                break

        if use_objective:
            reported = objectives[-1][1] if objectives else 0.0
        else:
            reported = residuals[-1] if residuals else 0.0
        return OracleResult(
            solution=state,
            iters=iters,
            residual=reported,
            trajectory=trajectory,
            residuals=residuals,
            objectives=objectives,
            solved=solved,
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


def _state_vector(state_leaves, x0=None) -> tuple[np.ndarray, list]:
    """Initial state vector and ``(leaf, offset, dim)`` slots for a leaf list."""
    slots = []
    offset = 0
    for leaf in state_leaves:
        slots.append((leaf, offset, leaf.ndim))
        offset += leaf.ndim
    if x0 is not None:
        return np.asarray(x0, dtype=float).copy(), slots
    vec = np.zeros(offset)
    for leaf, off, dim in slots:
        lb = np.asarray(leaf.lb, dtype=float).reshape(-1)
        ub = np.asarray(leaf.ub, dtype=float).reshape(-1)
        fixed = np.isfinite(lb) & np.isfinite(ub) & np.isclose(lb, ub)
        vec[off : off + dim] = np.where(fixed, lb, 0.0)
    return vec, slots


def run_recorded_tape(
    verifier,
    iterates,
    param_values: dict,
    state=None,
    params=None,
    x0: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Replay the verifier's whole recorded tape once, in recorded order.

    This is the algorithm the user actually wrote. Every other evaluator here is
    rolled — it repeats a single extracted iteration — so all of them agree with
    a wrongly-rolled IR by construction. This one does not, which is what makes
    it usable as ground truth.

    Parameters
    ----------
    verifier:
        The ``Verifier`` holding the recorded steps.
    iterates:
        The successive iterate expressions, one per recorded iteration, in order.
        Each entry has the same shape as ``next_state``: an expression or a list
        of them.
    param_values:
        Maps each parameter *leaf* to its value vector.
    state, params:
        Optional explicit leaf lists, as for :func:`run_faithful`.
    x0:
        Optional initial state; defaults to the fixed values of the state leaves.

    Returns
    -------
    list of ndarray
        The state trajectory, including the initial state, of length
        ``len(iterates) + 1``.
    """
    iterate_groups = [_as_list(g) for g in iterates]
    if not iterate_groups:
        raise ValueError("iterates must contain at least one recorded iteration")

    _, _, _, state_leaves, param_leaves = resolve_structure(
        verifier, iterate_groups[0], state, params
    )
    state_vec, slots = _state_vector(state_leaves, x0)

    values: dict = {}
    for leaf, off, dim in slots:
        values[leaf] = state_vec[off : off + dim]
    for leaf in param_leaves:
        values[leaf] = np.asarray(param_values[leaf], dtype=float).reshape(-1)

    # Evaluate the tape once, in recorded order. Each step's inputs are either
    # external leaves or fresh leaves of earlier steps, so one forward pass is
    # enough.
    for step in verifier.steps:
        values[fresh_leaf_of(step)] = _eval_step_fresh(step, values)

    trajectory = [state_vec.copy()]
    for group in iterate_groups:
        pieces = [eval_expr(e, values) for e in group]
        trajectory.append(np.concatenate(pieces) if pieces else np.zeros(0))
    return trajectory


def check_stationary(
    ir: FomIR,
    verifier,
    iterates,
    param_values: dict,
    state=None,
    params=None,
    tol: float = 1e-9,
) -> None:
    """Verify that the rolled IR reproduces the recorded tape.

    Extraction reads one iteration and repeats it. That is only faithful if the
    algorithm really is stationary, or if whatever varies was declared as a
    schedule. This compares the rolled trajectory against
    :func:`run_recorded_tape` over the recorded horizon and raises if they part
    company.

    Raises
    ------
    NonStationaryFOMError
        If the two trajectories differ by more than ``tol`` at any recorded
        iteration.
    """
    truth = run_recorded_tape(verifier, iterates, param_values, state=state, params=params)
    n_iters = len(truth) - 1

    named = {}
    _, _, _, _, param_leaves = resolve_structure(verifier, _as_list(iterates[0]), state, params)
    for i, leaf in enumerate(param_leaves):
        named[f"p{i}"] = np.asarray(param_values[leaf], dtype=float).reshape(-1)

    # Run exactly the recorded horizon with early exit disabled, so the
    # comparison is iteration-for-iteration.
    probe_ir = replace(
        ir,
        termination=TerminationSpec(
            kind=TerminationKind.MAX_ITERS,
            max_iters=n_iters,
            check_termination=ir.termination.check_termination,
        ),
    )
    rolled = ReferenceOracle(probe_ir).solve(param_values=named, x0=truth[0]).trajectory

    for k in range(1, n_iters + 1):
        err = float(np.max(np.abs(rolled[k] - truth[k]))) if truth[k].size else 0.0
        if err > tol:
            raise NonStationaryFOMError(
                f"the extracted iteration does not reproduce the recorded algorithm: "
                f"they first diverge at iteration {k} of {n_iters}, by {err:.3e} "
                f"(tolerance {tol:.1e}).\n"
                "The recorded method is not stationary — a coefficient changes with the "
                "iteration index. Declare it as a schedule and generate through "
                "extract_scheduled_ir()/generate_code(build, schedules=...), which "
                "tabulates it and reads the table at the loop counter."
            )


def sample_param_values(verifier, iterates, state=None, params=None, seed: int = 0) -> dict:
    """A deterministic parameter draw keyed by parameter *leaf*.

    Mirrors what the emitted verify harness does: uniform inside the declared box,
    with an infinite bound standing in as -1 / +1. A draw rather than the box
    midpoint, because the midpoint of a symmetric box is zero, which would make a
    soft-threshold step trivially zero and hide exactly the errors worth catching.
    """
    groups = normalize_iterates(verifier, iterates)
    _, _, _, _, param_leaves = resolve_structure(verifier, groups[-1], state, params)
    rng = np.random.default_rng(seed)
    values = {}
    for leaf in param_leaves:
        lb = np.asarray(leaf.lb, dtype=float).reshape(-1)
        ub = np.asarray(leaf.ub, dtype=float).reshape(-1)
        values[leaf] = rng.uniform(
            np.where(np.isfinite(lb), lb, -1.0), np.where(np.isfinite(ub), ub, 1.0)
        )
    return values


def check_unrolled(
    ir: FomIR,
    verifier,
    iterates,
    param_values: dict | None = None,
    state=None,
    params=None,
    tol: float = 1e-9,
) -> None:
    """Verify that an unrolled IR reproduces the recorded tape.

    The unrolled body is the whole run, so only its *final* state is comparable
    with the tape; the intermediate iterates are temporaries rather than states.

    Unlike :func:`check_stationary`, this needs nothing from the caller:
    ``param_values`` defaults to :func:`sample_param_values`. That is why the
    unrolled path can gate itself unconditionally where the rolled path has to
    warn when it was given no data to check with.

    Raises
    ------
    UnrolledFidelityError
        If the final states differ by more than ``tol``.
    """
    groups = normalize_iterates(verifier, iterates)
    if param_values is None:
        param_values = sample_param_values(verifier, groups, state=state, params=params)

    truth = run_recorded_tape(verifier, groups, param_values, state=state, params=params)

    _, _, _, _, param_leaves = resolve_structure(verifier, groups[-1], state, params)
    named = {
        f"p{i}": np.asarray(param_values[leaf], dtype=float).reshape(-1)
        for i, leaf in enumerate(param_leaves)
    }

    # Execute the body exactly once, whatever termination the caller configured:
    # an unrolled body is a single pass by construction.
    probe_ir = replace(ir, termination=TerminationSpec(kind=TerminationKind.MAX_ITERS, max_iters=1))
    got = ReferenceOracle(probe_ir).solve(param_values=named, x0=truth[0]).solution

    err = float(np.max(np.abs(got - truth[-1]))) if truth[-1].size else 0.0
    if err > tol:
        raise UnrolledFidelityError(
            f"the unrolled body does not reproduce the recorded tape: the final state "
            f"differs by {err:.3e} over {len(truth) - 1} recorded iteration(s) "
            f"(tolerance {tol:.1e}).\n"
            "An unrolled body gives every iteration its own coefficients, so this is "
            "not a limitation of the recording. Please report it with the recording."
        )
