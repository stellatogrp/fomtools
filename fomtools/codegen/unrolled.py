"""Extract a whole recorded run as one unrolled straight-line program.

The other two extraction paths need the method restructured before it can be
generated. :func:`~fomtools.codegen.extract.extract_ir` wants one *stationary*
iteration that a generated loop repeats, and
:func:`~fomtools.codegen.scheduled.extract_scheduled_ir` wants the recording as a
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

The cost is size. Extraction expresses each iterate in terms of leaves and
earlier temporaries, which is linear in the iteration count when the loop carries
a *leaf* -- a step output fed into the next step, as in ISTA, FISTA and Halpern --
and quadratic when it carries a non-leaf accumulation like Douglas-Rachford's
``z = z + ut - u``. :func:`fomtools.verifier.Verifier.iterate_trajectory`
materializes an accumulating iterate as its own variable and restores linear
growth; a recording that would blow up raises
:class:`~fomtools.codegen.ir.UnrolledBlowupError` naming it. Repeated coefficient
blocks are then merged by content, so a stationary method unrolled stores its
data once rather than once per iteration.
"""

from __future__ import annotations

from dataclasses import replace

from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec
from fomtools.codegen.dedup import dedup_matrices
from fomtools.codegen.extract import extract_ir, normalize_iterates
from fomtools.codegen.ir import FomIR, UnrolledBlowupError
from fomtools.codegen.oracle import check_unrolled

__all__ = ["extract_unrolled_ir"]


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


def _one_pass(config: CodegenConfig) -> TerminationSpec:
    """The termination an unrolled body must carry: execute once, no early exit."""
    if config.termination.kind != TerminationKind.MAX_ITERS:
        raise ValueError(
            f"termination kind {config.termination.kind.value!r} cannot be combined with "
            "unrolled code generation: the body is the whole run executed once, so there "
            "is no repeated iteration to measure a change across. Use "
            "TerminationKind.MAX_ITERS, or generate a rolled solver instead, where the "
            "criterion is evaluated every iteration."
        )
    return TerminationSpec(kind=TerminationKind.MAX_ITERS, max_iters=1)


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
        "Materialize each iterate as its own variable and the growth becomes linear:\n\n"
        "    z = verifier.iterate_trajectory(K + 1, collapse_expr=False)\n"
        "    z[0] = verifier.add_iterate(n, lb=z0, ub=z0)\n"
        "    for k in range(1, K + 1):\n"
        "        ...\n"
        "        z[k] = <the iterate>\n\n"
        "then pass that trajectory as iterates=."
    )


def extract_unrolled_ir(
    verifier,
    iterates,
    config: CodegenConfig | None = None,
    name: str = "fom",
    state=None,
    params=None,
    dedup: bool = True,
    check: bool = True,
    param_values: dict | None = None,
) -> FomIR:
    """Extract the whole recorded run as one unrolled body.

    Parameters
    ----------
    verifier:
        The ``Verifier`` holding the recorded steps.
    iterates:
        The iterate sequence, either an
        :class:`~fomtools.iteratetrajectory.IterateTrajectory` or a plain
        sequence of expressions. A leading entry that is the *initial* iterate is
        recognized and dropped, so both conventions work.
    config:
        Code-generation configuration. Its termination is replaced by a single
        pass; a tolerance-based kind is rejected.
    state, params:
        Optional explicit leaf lists, as for
        :func:`~fomtools.codegen.extract.extract_ir`.
    dedup:
        Merge coefficient blocks holding identical numbers.
    check:
        Verify the result against the recorded tape before returning.
    param_values:
        Parameter values for that check, keyed by parameter leaf. Defaults to a
        deterministic draw inside the declared box.

    Returns
    -------
    FomIR
        With :attr:`~fomtools.codegen.ir.FomIR.unrolled` set, a body holding the
        whole run in dependency order, and ``state_update`` producing the answer.
    """
    config = config or CodegenConfig()

    groups = normalize_iterates(verifier, iterates)
    if not groups:
        raise ValueError(
            "iterates must contain at least one recorded iteration. Pass the iterate "
            "sequence the loop produced, or an IterateTrajectory."
        )

    ir = extract_ir(
        verifier,
        groups[-1],
        state=state,
        params=params,
        config=config.model_copy(update={"termination": _one_pass(config)}),
        name=name,
    )
    _check_no_blowup(ir, len(groups))

    ir = replace(ir, unrolled=True)
    if dedup:
        ir = dedup_matrices(ir)
    if check:
        check_unrolled(ir, verifier, groups, param_values=param_values, state=state, params=params)
    return ir
