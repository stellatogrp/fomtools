"""Extract a whole recorded run as one unrolled straight-line program.

The other two extraction paths need the method restructured before it can be
generated. `extract_ir` wants one *stationary*
iteration that a generated loop repeats, and
`extract_scheduled_ir` wants the recording as a
*build function* of the coefficients that vary, whose dependence must be affine.
Neither accepts the program the user already wrote to verify with: a plain Python
loop over ``verifier.step.*``.

This path does. It reads the finished tape and emits the run as a single pass, so
every iteration carries its own coefficient blocks. That buys three things the
rolled paths cannot express: coefficients varying with ``k`` in any way at all
rather than affinely, the step *type* changing between iterations, and a
reference to an earlier iterate held across the whole run -- a Halpern anchor
``z_{k+1} = a_k z_0 + (1 - a_k) T(z_k)``, where ``z_0`` simply resolves as a state
slot that nothing updates.

The cost is size, so a recording that did not need unrolling is rolled back up.
Before unrolling anything we extract iteration one on its own and ask
`check_stationary` whether repeating it reproduces
the recorded tape. When it does, that compact IR is what gets emitted, so writing
a stationary method as a loop costs nothing. **Rolling changes the artifact and
never the computation**: the rolled IR keeps the recorded horizon, and the same
check that decides to roll is the proof it was safe. Using the gate as the
decision rather than comparing block structure means a wrong roll is not merely
unlikely; it cannot be returned.

What remains unrolled is what genuinely varies. There the size depends on how the
loop carries its iterate. Extraction expresses each iterate in terms of leaves and
earlier temporaries, which is linear in the iteration count when the loop carries
a *leaf* -- a step output fed into the next step, as in ISTA, FISTA and Halpern --
and quadratic when it carries a non-leaf accumulation like Douglas-Rachford's
``z = z + ut - u``. `fomtools.verifier.Verifier.iterate_trajectory`
materializes an accumulating iterate as its own variable and restores linear
growth; a recording that would blow up raises
`UnrolledBlowupError` naming it. Repeated coefficient
blocks are then merged by content, so what is unrolled stores its data once
rather than once per iteration.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec
from fomtools.codegen.dedup import dedup_matrices
from fomtools.codegen.extract import extract_ir, normalize_iterates
from fomtools.codegen.ir import (
    FomIR,
    NonStationaryFOMError,
    ScheduleFitError,
    UnrolledBlowupError,
    UnrolledFidelityError,
)
from fomtools.codegen.objective import ObjectiveExtractionError
from fomtools.codegen.oracle import check_stationary, check_unrolled, sample_param_values

__all__ = ["extract_unrolled_ir"]

log = logging.getLogger(__name__)

#: Subclasses of ``NonStationaryFOMError`` that mean a genuine failure rather
#: than "this recording is not stationary". ``NonStationaryFOMError`` is caught to
#: decide against rolling, so these must be re-raised past that handler; none is
#: raised by the rolled path today, and this keeps that from becoming a trap if
#: one ever is.
_REAL_FAILURES = (ScheduleFitError, UnrolledBlowupError, UnrolledFidelityError)

#: Parameter draws used to decide whether the recording rolls. A wrong roll
#: silently changes the algorithm, so one coincidental agreement must not be
#: enough; the check also compares every iteration, not just the answer.
_ROLL_CONFIRM_SEEDS = (0, 1, 2)


#: A body is judged to accumulate when its late steps carry this many times more
#: terms than its early steps. Growth, not an absolute count, is the signature:
#: a leaf-carrying loop holds its term count flat however long it runs, while an
#: accumulating one grows without bound.
_GROWTH_RATIO = 3.0

#: Below this many terms the artifact is small whatever its shape, so growth is
#: not worth complaining about.
_BLOWUP_MIN_TERMS = 200

#: Too few steps to read a trend from.
_BLOWUP_MIN_STEPS = 10


def _require_supported_termination(config: CodegenConfig) -> None:
    """Reject the infinity-norm change criterion, whether or not the recording rolls.

    Checked up front rather than per branch: generating from a tape means
    generating the run that was recorded, and a rolled result keeps that same
    horizon. The verification objective is the exception: it is the metric the
    recording was verified on, so stopping on it within the recorded horizon is
    the certified stopping time. A rolled result tests it at the check interval
    and at the horizon; an unrolled single pass evaluates it once, at the end,
    and reports it.
    """
    if config.termination.kind == TerminationKind.INF_NORM_CHANGE:
        raise ValueError(
            f"termination kind {config.termination.kind.value!r} cannot be combined with "
            "code generation from a recorded tape: the generated solver runs the run that "
            "was recorded, so there is no criterion to evaluate against a horizon the "
            "recording does not have. Use TerminationKind.MAX_ITERS, TerminationKind."
            "OBJECTIVE (the verification objective, within the recorded horizon), or "
            "generate a rolled solver directly from one iteration, where the criterion is "
            "evaluated every iteration."
        )


def _horizon(config: CodegenConfig, max_iters: int) -> TerminationSpec:
    """The termination of an IR generated from a tape: the recorded horizon.

    Fixed-count by default; an ``OBJECTIVE`` criterion keeps its tolerance and
    check interval and gets the horizon as its ``max_iters``.
    """
    term = config.termination
    if term.uses_objective:
        return term.model_copy(update={"max_iters": max_iters})
    return TerminationSpec(kind=TerminationKind.MAX_ITERS, max_iters=max_iters)


def _term_counts(ir: FomIR) -> list[int]:
    return [len(step.linear_op.terms) for step in ir.body]


def _check_no_blowup(ir: FomIR, n_iterations: int) -> None:
    """Reject a recording whose operation count grows with the iteration count."""
    counts = _term_counts(ir)
    total = sum(counts) + len(ir.state_update.terms)
    if len(counts) < _BLOWUP_MIN_STEPS or total <= _BLOWUP_MIN_TERMS:
        return

    window = max(1, len(counts) // 5)
    early = sum(counts[:window]) / window
    late = sum(counts[-window:]) / window
    if late <= _GROWTH_RATIO * max(early, 1.0):
        return

    raise UnrolledBlowupError(
        f"the recorded loop carries its iterate through a non-leaf expression, so the "
        f"unrolled body grows with the square of the iteration count: its steps start at "
        f"{early:.0f} term(s) and end at {late:.0f}, for {total} matrix-vector products "
        f"over {n_iterations} iteration(s).\n"
        "Build the iterate sequence with a trajectory and the growth becomes linear: "
        "its default 'auto' policy notices the accumulation and keeps each iterate as "
        "its own variable.\n\n"
        "    z = verifier.iterate_trajectory(K + 1)\n"
        "    z[0] = verifier.iterate(n, lb=z0, ub=z0)\n"
        "    for k in range(1, K + 1):\n"
        "        ...\n"
        "        z[k] = <the iterate>\n\n"
        "then pass that trajectory as iterates=. If this recording already used a "
        "trajectory, it was built with collapse_expr=True, which overrides that."
    )


def _try_roll(
    verifier, groups, config: CodegenConfig, name: str, state, params, param_values
) -> FomIR | None:
    """The compact rolled IR if repeating iteration one reproduces the tape.

    Returns ``None`` when it does not, which is the signal to unroll. The rolled
    IR keeps the *recorded* horizon rather than any configured loop count: rolling
    is a size optimization, and running a different number of iterations would
    make it a change of behavior.
    """
    rolled_config = config.model_copy(update={"termination": _horizon(config, len(groups))})
    try:
        rolled = extract_ir(
            verifier,
            groups[0],
            state=state,
            params=params,
            config=rolled_config,
            name=name,
            iterates=groups,
        )
    except _REAL_FAILURES:
        # Subclasses of NonStationaryFOMError that report something wrong rather
        # than "this does not roll". Listed first so they propagate: swallowing
        # one would turn a real error into a silent decision to unroll.
        raise
    except NonStationaryFOMError:
        # Iteration one is not extractable on its own; nothing to roll.
        return None
    except ObjectiveExtractionError as err:
        # The objective reads something only the whole run carries (say, a step
        # output of the last iteration). The unrolled body keeps every
        # iteration's outputs, so it can still evaluate it.
        log.info("%s: not rolling, the objective needs the unrolled run: %s", name, err)
        return None

    # The multi-seed confirmation runs whether or not the caller supplied values.
    # A single draw is not enough to license a roll even when it is the caller's
    # own: one degenerate choice of parameters can make a genuinely varying
    # method look stationary, which is the original FISTA-collapses-to-ISTA
    # failure. A caller's draw is checked too, never instead.
    draws = [
        sample_param_values(verifier, groups, state=state, params=params, seed=seed)
        for seed in _ROLL_CONFIRM_SEEDS
    ]
    if param_values is not None:
        draws.insert(0, param_values)
    for values in draws:
        try:
            check_stationary(rolled, verifier, groups, values, state=state, params=params)
        except _REAL_FAILURES:
            raise
        except NonStationaryFOMError:
            return None
    return rolled


def extract_unrolled_ir(
    verifier,
    iterates,
    config: CodegenConfig | None = None,
    name: str = "fom",
    state=None,
    params=None,
    dedup: bool = True,
    check: bool = True,
    roll: bool = True,
    param_values: dict | None = None,
) -> FomIR:
    """Extract the whole recorded run as one unrolled body.

    Parameters
    ----------
    verifier:
        The ``Verifier`` holding the recorded steps.
    iterates:
        The iterate sequence, either an
        `IterateTrajectory` or a plain
        sequence of expressions. A leading entry that is the *initial* iterate is
        recognized and dropped, so both conventions work.
    config:
        Code-generation configuration. Its termination is replaced by the
        recorded horizon (a single pass when unrolled). ``OBJECTIVE`` keeps its
        tolerance and check interval: a rolled result stops on the verification
        objective within the horizon, an unrolled one evaluates it once at the
        end. ``INF_NORM_CHANGE`` is rejected.
    state, params:
        Optional explicit leaf lists, as for
        `extract_ir`.
    dedup:
        Merge coefficient blocks holding identical numbers.
    check:
        Verify the *unrolled* result against the recorded tape before returning.
        Does not apply to a rolled result: there the check is what decides to roll
        in the first place, so it cannot be skipped without changing the answer.
        Combine with ``roll=False`` to return an IR that has not been verified.
    roll:
        Emit the compact rolled IR when repeating one iteration is verified to
        reproduce the recording. Pass ``False`` to unroll regardless.
    param_values:
        Parameter values for the checks, keyed by parameter leaf. Defaults to
        deterministic draws inside the declared box. Supplying them *adds* a
        draw to the roll decision rather than replacing the default ones: no
        single draw, the caller's included, is enough to license a roll.

    Returns
    -------
    FomIR
        Either a rolled IR (``unrolled`` false, the recorded horizon in
        ``termination.max_iters``), or an unrolled one whose body holds the whole
        run in dependency order with ``state_update`` producing the answer.
        `unrolled` says which.
    """
    config = config or CodegenConfig()
    _require_supported_termination(config)

    groups = normalize_iterates(verifier, iterates)
    if not groups:
        raise ValueError(
            "iterates must contain at least one recorded iteration. Pass the iterate "
            "sequence the loop produced, or an IterateTrajectory."
        )

    # Try rolling before unrolling: for a stationary recording this is both the
    # better artifact and the cheaper path, since it never builds the big body.
    if roll:
        rolled = _try_roll(verifier, groups, config, name, state, params, param_values)
        if rolled is not None:
            log.info(
                "%s: repeating one iteration reproduces the %d recorded iteration(s); "
                "emitting a rolled body of %d step(s) instead of unrolling.",
                name,
                len(groups),
                len(rolled.body),
            )
            return rolled

    ir = extract_ir(
        verifier,
        groups[-1],
        state=state,
        params=params,
        config=config.model_copy(update={"termination": _horizon(config, 1)}),
        name=name,
        iterates=groups,
    )
    _check_no_blowup(ir, len(groups))
    log.info(
        "%s: the recording does not roll; emitting %d unrolled step(s) over "
        "%d recorded iteration(s).",
        name,
        len(ir.body),
        len(groups),
    )

    ir = replace(ir, unrolled=True)
    if dedup:
        ir = dedup_matrices(ir)
    if check:
        check_unrolled(ir, verifier, groups, param_values=param_values, state=state, params=params)
    return ir
