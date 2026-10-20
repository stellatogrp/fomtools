"""Extraction for methods whose coefficients vary with the iteration index.

`extract_ir` reads one recorded iteration and
bakes its coefficients in as constants. That is correct for a stationary method
and silently wrong for anything else: FISTA recorded this way collapses to plain
ISTA, because its first momentum coefficient is zero.

The fix rests on a structural fact. A schedule scalar enters the recorded
``LinearExpression`` linearly, so every extracted coefficient block is *affine*
in it:

$$
M(\\gamma) = M_{0} + \\gamma M_{1}.
$$

So we do not need the scalar to stay symbolic through the recording. We ask the
user for the recording as a *function* of its schedule scalars, evaluate it at a
few probe values, and recover the base and slope blocks by finite differences.
One extra probe is held out to validate the fit, which is what catches a
coefficient that varies in some way a table cannot express.

The recovered slopes are emitted as extra coefficient blocks paired with a
scaled `Term`, and the schedule itself becomes a
table read at the loop counter. Nothing about the stationary path changes.

The tables are `Schedule` objects, coerced from
whatever the caller supplied by
`as_schedule`; that module is the type, this one
is the pass over it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import (
    FomIR,
    LinearOp,
    MatrixData,
    NonStationaryFOMError,
    ScheduleFitError,
    Term,
    max_abs,
)
from fomtools.codegen.schedule import as_schedule

__all__ = ["Recording", "extract_scheduled_ir"]

# Probe values for the schedule scalars. Deliberately generic: never 0 or 1, and
# not simple fractions. A zero probe would delete a term from the recorded
# decomposition dictionary entirely (this is exactly how FISTA's gamma_1 = 0
# collapses its two-slot state down to one), and a probe that happens to cancel
# against the problem data would make a varying block look constant.
_PROBE_BASE = 0.3141592653589793
_PROBE_STEP = 0.5772156649015329
_PROBE_CHECK = 0.8414709848078965

# A block counts as varying when a probe moves it by more than this, relative to
# the block's own scale. Well clear of the ~1e-16 round-off seen on blocks that
# are genuinely constant.
_VARY_RTOL = 1e-12

# The held-out probe must be predicted at least this well, relative to the
# block's scale. Measured error on a true affine dependence is ~1e-16.
_FIT_RTOL = 1e-10


@dataclass
class Recording:
    """One recorded iteration of a FOM, as returned by a build function.

    Attributes
    ----------
    verifier:
        The ``Verifier`` the iteration was recorded on.
    next_state:
        The iterate expression (or list of them) after exactly one iteration.
    state, params:
        Optional explicit leaf lists, forwarded to
        `extract_ir`. Passing ``state``
        explicitly is the usual way to pin a multi-slot layout, such as the
        ``(y^k, y^{k-1})`` pair an accelerated method carries.
    """

    verifier: Any
    next_state: Any
    state: Any = None
    params: Any = None


def _scale_of(arr) -> float:
    """A magnitude for relative comparisons that never returns zero."""
    m = max_abs(arr)
    return m if m > 0.0 else 1.0


def _sum(blocks) -> np.ndarray:
    """Sum of dense or sparse blocks (``sum`` would start from the integer 0)."""
    blocks = list(blocks)
    total = blocks[0]
    for block in blocks[1:]:
        total = total + block
    return total


def _extract_at(build: Callable, point: dict[str, float], config, name: str) -> FomIR:
    result = build(**point)
    if not isinstance(result, Recording):
        raise TypeError(
            f"the build function must return a codegen.Recording, got {type(result).__name__}"
        )
    return extract_ir(
        result.verifier,
        result.next_state,
        state=result.state,
        params=result.params,
        config=config,
        name=name,
    )


def _linops(ir: FomIR) -> list[LinearOp]:
    return ir.linops()


def _structure(ir: FomIR) -> dict:
    """Everything about an IR that a schedule must not change."""
    return {
        "state_dim": ir.state.dim,
        "state_slots": tuple(ir.state.slots),
        "params": tuple((p.name, p.dim) for p in ir.params),
        "body": tuple((s.name, s.out_dim, s.activation.kind, s.solve) for s in ir.body),
        "matrices": tuple(sorted((n, tuple(m.array.shape)) for n, m in ir.matrices.items())),
        "linops": tuple(
            (lin.out_dim, lin.const, tuple((t.source, t.matrix) for t in lin.terms))
            for lin in _linops(ir)
        ),
        "factorizations": tuple(sorted(ir.factorizations)),
    }


def _check_same_structure(irs: list[FomIR], points: list[dict[str, float]]) -> None:
    ref = _structure(irs[0])
    for ir, point in zip(irs[1:], points[1:]):
        other = _structure(ir)
        for key in ref:
            if ref[key] != other[key]:
                raise NonStationaryFOMError(
                    f"the recording changes shape with its schedule scalars: {key} is "
                    f"{ref[key]!r} at {points[0]} but {other[key]!r} at {point}. "
                    "A schedule may scale coefficients; it may not add, remove, or "
                    "resize steps, state slots, or parameters. A state slot that "
                    "disappears usually means an iterate is only referenced through "
                    "the schedule, so it drops out when the scalar is small — declare "
                    "it with an explicit state= list."
                )


def _check_scalars_fixed(irs: list[FomIR]) -> None:
    """Activation scalars cannot be scheduled yet; fail loudly rather than freeze one."""
    ref = irs[0]
    for other in irs[1:]:
        for a, b in zip(ref.body, other.body):
            for attr in ("lambd", "lower_scalar", "upper_scalar"):
                x, y = getattr(a.activation, attr), getattr(b.activation, attr)
                if x is None and y is None:
                    continue
                if x is None or y is None or not np.isclose(x, y, rtol=0, atol=0):
                    raise ScheduleFitError(
                        f"step {a.name!r} has an iteration-varying {attr} "
                        f"({x} vs {y}). Scheduled activation scalars (a soft-threshold "
                        "level or a saturated-linear bound that changes with k) are not "
                        "supported yet; only linear-operator coefficients can be scheduled."
                    )


def _check_factorizations_fixed(irs: list[FomIR]) -> None:
    """A factorization is computed once at generation time, so its matrix must be fixed.

    Compared at ``_VARY_RTOL``, the same tolerance that decides whether a
    coefficient block varies, rather than at ``np.allclose``'s default ``rtol`` of
    1e-5. A system matrix that moves only weakly with the schedule scalar would
    clear the loose default and then be silently frozen at the base
    factorization, which is the single outcome this check exists to prevent.
    """
    ref = irs[0]
    for fname, fac in ref.factorizations.items():
        rng = np.random.default_rng(0)
        probe = rng.normal(size=fac.dim)
        base = fac.solve(probe)
        scale = _scale_of(base)
        for other in irs[1:]:
            moved = other.factorizations[fname].solve(probe)
            if float(np.max(np.abs(moved - base))) > _VARY_RTOL * scale:
                raise ScheduleFitError(
                    f"the linear system solved by step {fname!r} varies with the "
                    "iteration index. Its factorization is computed once at generation "
                    "time and cannot be recomputed per iteration, so this method cannot "
                    "be lowered. Keep the system matrix fixed and put the varying "
                    "coefficient outside the solve."
                )


def _recover_slopes(
    irs: list[FomIR], names: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Split every coefficient block into a constant base and per-schedule slopes.

    Returns ``(bases, slopes)`` where ``M(g) = bases[m] + sum_n g[n] * slopes[m][n]``.
    Blocks that do not vary appear in ``bases`` only.
    """
    base_ir, check_ir = irs[0], irs[-1]
    bases: dict[str, np.ndarray] = {}
    slopes: dict[str, dict[str, np.ndarray]] = {}

    for mname, md in base_ir.matrices.items():
        m_at_base = md.array
        scale = _scale_of(m_at_base)
        per_schedule: dict[str, np.ndarray] = {}
        for i, sched in enumerate(names):
            moved = irs[1 + i].matrices[mname].array
            delta = moved - m_at_base
            if max_abs(delta) > _VARY_RTOL * scale:
                per_schedule[sched] = delta / (_PROBE_STEP - _PROBE_BASE)

        if not per_schedule:
            bases[mname] = m_at_base
            continue

        # M(g) = base + sum_n g_n S_n, with m_at_base = M(PROBE_BASE, ..., PROBE_BASE).
        const_block = m_at_base - _PROBE_BASE * _sum(per_schedule.values())
        predicted = const_block + _PROBE_CHECK * _sum(per_schedule.values())
        err = max_abs(predicted - check_ir.matrices[mname].array)
        if err > _FIT_RTOL * scale:
            raise ScheduleFitError(
                f"coefficient block {mname!r} does not depend affinely on its schedule "
                f"scalars: predicting the held-out probe was off by {err:.3e} "
                f"(tolerance {_FIT_RTOL * scale:.3e}). A schedule scalar must enter the "
                "recorded expression linearly; a product of two schedules, or a scalar "
                "used inside a nonlinearity, cannot be tabulated this way."
            )
        bases[mname] = const_block
        slopes[mname] = per_schedule

    return bases, slopes


def _rewrite_linop(lin: LinearOp, slopes: dict[str, dict[str, np.ndarray]]) -> LinearOp:
    """Split each term over a varying block into an unscaled base and scaled slopes."""
    if lin.const is not None and lin.const in slopes:
        raise ScheduleFitError(
            f"the constant vector {lin.const!r} varies with the iteration index. "
            "Scheduled constant offsets are not supported yet; only matrix "
            "coefficients can be scheduled."
        )
    terms: list[Term] = []
    for term in lin.terms:
        terms.append(term)
        for sched in sorted(slopes.get(term.matrix, {})):
            terms.append(
                Term(
                    source=term.source,
                    matrix=f"{term.matrix}__d_{sched}",
                    scale=sched,
                )
            )
    return replace(lin, terms=tuple(terms))


def extract_scheduled_ir(
    build: Callable[..., Recording],
    schedules: dict,
    config: CodegenConfig | None = None,
    name: str = "fom",
) -> FomIR:
    """Extract an IR whose coefficients vary with the iteration index.

    Parameters
    ----------
    build:
        A function that records **one** iteration and returns a
        `Recording`. It is called with one keyword argument per entry of
        ``schedules``, and must be free of side effects: it is called several
        times, at different values of those scalars, and must build the same
        algorithm each time.
    schedules:
        Maps a schedule name to a callable of the **0-based** iteration index, a
        sequence of values, or a `Schedule`. A
        textbook formula written in 1-based ``k`` must be shifted by the caller;
        the coefficient belongs to the method, not to the generator.
    config:
        Code-generation configuration; its ``termination.max_iters`` fixes the
        length of every emitted table.
    name:
        Human-friendly FOM name used for file naming.

    Returns
    -------
    FomIR
        A single-iteration IR carrying the schedule tables and scaled terms.

    Raises
    ------
    NonStationaryFOMError
        If the recording changes shape between probes.
    ScheduleFitError
        If a coefficient's dependence on the schedule is not affine, or the
        varying quantity reaches somewhere a table cannot (a factorized system
        matrix, an activation scalar, a constant offset).
    """
    config = config or CodegenConfig()
    if not schedules:
        raise ValueError(
            "extract_scheduled_ir() needs at least one schedule; use extract_ir() "
            "for a stationary method."
        )
    names = sorted(schedules)

    tables = {n: as_schedule(n, schedules[n], config.termination.max_iters) for n in names}

    # One probe at the base point, one per schedule with that scalar moved, and a
    # final all-moved point held out to validate the recovered fit.
    base_point = {n: _PROBE_BASE for n in names}
    points = [base_point]
    for n in names:
        moved = dict(base_point)
        moved[n] = _PROBE_STEP
        points.append(moved)
    points.append({n: _PROBE_CHECK for n in names})

    irs = [_extract_at(build, point, config, name) for point in points]

    _check_same_structure(irs, points)
    _check_scalars_fixed(irs)
    _check_factorizations_fixed(irs)

    bases, slopes = _recover_slopes(irs, names)

    ref = irs[0]
    bound_names = set()
    for step in ref.body:
        for attr in ("lower", "upper"):
            bound = getattr(step.activation, attr)
            if bound is not None:
                bound_names.add(bound)
    for mname in slopes:
        if mname in bound_names:
            raise ScheduleFitError(
                f"the saturated-linear bound {mname!r} varies with the iteration index. "
                "Scheduled clip bounds are not supported yet; only linear-operator "
                "coefficients can be scheduled."
            )

    matrices: dict[str, MatrixData] = {
        mname: MatrixData(
            name=mname,
            array=bases[mname],
            sparse_pattern=ref.matrices[mname].sparse_pattern if mname not in slopes else None,
        )
        for mname in ref.matrices
    }
    for mname, per_schedule in slopes.items():
        for sched, slope in per_schedule.items():
            dname = f"{mname}__d_{sched}"
            matrices[dname] = MatrixData(name=dname, array=slope)

    body = tuple(
        replace(step, linear_op=_rewrite_linop(step.linear_op, slopes)) for step in ref.body
    )
    state_update = _rewrite_linop(ref.state_update, slopes)
    # The objective is evaluated on the iterate an iteration produced, so a term
    # that varies with the schedule reads the same table entry as that iteration.
    objective = None
    if ref.objective is not None:
        objective = replace(
            ref.objective,
            blocks=tuple(
                replace(block, linear_op=_rewrite_linop(block.linear_op, slopes))
                for block in ref.objective.blocks
            ),
        )

    used = {
        t.scale
        for lin in [*(s.linear_op for s in body), state_update]
        + (objective.linops() if objective is not None else [])
        for t in lin.terms
    }
    unused = [n for n in names if n not in used]
    if unused:
        raise ScheduleFitError(
            f"schedule(s) {unused!r} do not affect any coefficient. The build function "
            "must actually use every scalar it is given; an unused schedule usually "
            "means a typo in its keyword name."
        )

    return FomIR(
        params=ref.params,
        state=ref.state,
        body=body,
        state_update=state_update,
        termination=ref.termination,
        matrices=matrices,
        factorizations=ref.factorizations,
        schedules={n: tables[n].values for n in names},
        dtype=ref.dtype,
        name=name,
        objective=objective,
    )
