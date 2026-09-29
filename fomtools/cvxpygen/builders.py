"""First-order-method builders over the canonical QP form.

A *builder* takes a :class:`~fomtools.cvxpygen.qpdata.QPData` snapshot and
records a first-order method in a fomtools :class:`~fomtools.verifier.Verifier`,
returning a spec that the code generator and the C bridge consume. Built-in
builders:

``admm``
    ADMM (KKT solve + projection). Default; handles any QP.
``relaxed_admm``
    The same, with an over-relaxation that varies with the iteration index —
    the built-in that goes down the *scheduled* path.
``pdhg``
    PDHG / Chambolle-Pock (matrix-vector products only, no linear solve).
    Requires a zero quadratic term (LP).

Users can pass their own builder callable ``f(qp, **fom_options) -> spec`` to
``generate_solver(fom=...)`` — this is the "define your FOM in fomtools"
extension point.

Three generation paths
----------------------
A builder returns one of two spec types and declares which path it wants with
:func:`fom_builder`, mirroring the dispatch in
:func:`fomtools.codegen.generate.generate_code`:

``"rolled"``
    A :class:`FomSpec` with ``next_state``: one stationary iteration, which the
    generated loop repeats ``config.termination.max_iters`` times. Pass
    ``iterates`` too and generation *checks* that the method really is
    stationary.
``"unrolled"``
    A :class:`FomSpec` with ``next_state=None`` and ``iterates``: the whole
    recorded run, which accepts arbitrary per-iteration coefficients, a step
    type that changes with the iteration, and anchored methods. Generation rolls
    it back up when repeating one iteration is verified to reproduce it. The
    horizon is baked in, so this needs ``allow_unrolled=True`` at the
    ``generate_solver`` call.
``"scheduled"``
    A :class:`ScheduledFomSpec`: a build function whose varying coefficients are
    declared as schedules and tabulated, giving one emitted block plus a table
    whatever the horizon.

The declared kind is what the cvxpygen interface reads to decide which runtime
settings the generated solver can honestly expose, and it has to be known before
the builder can run (cvxpygen fixes the settings struct before canonicalization
reaches us) — hence a declaration rather than an inference from the produced IR.

Canonical structure exploited throughout: the first ``n_eq`` constraint rows
are equalities (projection onto the point ``b``) and the remaining rows are
one-sided inequalities ``Ax <= u`` (projection ``w -> w - relu(w - u)``), so
parameter-dependent projection bounds need only ``relu`` steps — the embedded
``satlin`` step requires numeric bounds and is never needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as spa

from fomtools.codegen.extract_scheduled import Recording
from fomtools.codegen.oracle import sample_param_values
from fomtools.cvxpygen.qpdata import QPData
from fomtools.linearexpression import LinearExpression, stack
from fomtools.verifier import Verifier

#: The generation paths a builder may declare, as named by
#: :func:`fomtools.codegen.generate.generate_code`.
FOM_KINDS = ("rolled", "unrolled", "scheduled")


@dataclass(frozen=True, kw_only=True)
class BaseFomSpec:
    """Where the QP lives inside the FOM state — everything the C bridge needs.

    ``param_copy`` lists, in ``Verifier.add_param`` order (i.e. FomParams field
    order ``p0, p1, ...``), which canonical piece feeds each runtime parameter:
    ``'q'`` (length ``n``), ``'b'`` (first ``n_eq`` entries of canonical ``l``)
    or ``'u_ineq'`` (last ``n_ineq`` entries of canonical ``u``). The primal /
    dual slices locate the QP solution inside the stacked FOM state.

    Shared by both spec types so :func:`fomtools.cvxpygen.glue.emit_bridge` can
    take either: the bridge cares only about the layout, never about which path
    produced the solver.
    """

    name: str
    primal_offset: int
    primal_dim: int
    dual_offset: int
    dual_dim: int
    param_copy: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class FomSpec(BaseFomSpec):
    """A recorded FOM, as a tape: the ``rolled`` and ``unrolled`` paths.

    ``state`` and ``params`` are the explicit leaf lists passed to
    ``extract_ir`` so that the state layout matches ``next_state`` (extraction
    would otherwise infer the ordering from leaf-discovery order).

    ``next_state`` is the iterate after exactly one iteration, which the
    generated loop repeats. Leave it ``None`` and generation reads ``iterates``
    as the whole run to emit — the unrolled path, which takes coefficients that
    vary with the iteration index in any way at all.

    ``iterates`` and ``param_values`` are what turn on code generation's
    stationarity check when ``next_state`` is given: the recorded tape to replay
    the extracted iteration against, and the parameter draw to replay it at.
    Extraction reads *one* iteration and repeats it, so without them nothing
    would catch a builder whose coefficients vary with the iteration index — the
    generated solver would run a different algorithm than the one recorded. Both
    default to empty for builders that supply no tape; generation then warns, as
    it does for any unchecked extraction.

    On the unrolled path ``iterates`` is the run itself, and either convention
    works: the recorded iterations alone, or those with the *initial* iterate
    kept in slot 0, as an
    :class:`~fomtools.iteratetrajectory.IterateTrajectory` holds them.
    :func:`~fomtools.codegen.extract.normalize_iterates` recognizes a leading
    initial iterate structurally — an entry referencing no recorded step output
    — and drops it, so the emitted run and the iteration count reported through
    ``solver_stats`` both follow the recorded iterations either way. The rolled
    stationarity check is stricter: it replays the tape through
    :func:`~fomtools.codegen.oracle.run_recorded_tape`, which reads every entry
    as an iteration, so a tape passed alongside ``next_state`` must hold the
    recorded iterations only.
    """

    verifier: Verifier
    next_state: LinearExpression | None
    state: tuple[LinearExpression, ...]
    params: tuple[LinearExpression, ...]
    iterates: tuple = ()
    param_values: dict | None = None


@dataclass(frozen=True, kw_only=True)
class ScheduledFomSpec(BaseFomSpec):
    """A FOM whose varying coefficients are declared: the ``scheduled`` path.

    ``build`` records **one** iteration as a function of its schedule scalars
    and returns a :class:`fomtools.codegen.extract_scheduled.Recording`; it is
    called several times, at different scalar values, so it must be free of side
    effects and must build the same algorithm each time.

    ``schedules`` maps each scalar's name to a callable of the **0-based**
    iteration index, a sequence, or a
    :class:`~fomtools.codegen.schedule.Schedule`. The tables are emitted
    alongside the solver and read at the loop counter, so the solver stays one
    block long however far it runs — but only as far as the tables reach, which
    is why the interface refuses a horizon longer than they cover.

    The state/parameter leaf lists live inside the ``Recording`` the build
    function returns, so unlike :class:`FomSpec` this carries none.
    """

    build: Callable
    schedules: dict


def fom_builder(kind: str) -> Callable:
    """Declare which generation path a builder's spec goes down.

    Used as a decorator on a builder function::

        @fom_builder("scheduled")
        def my_fom(qp, **options) -> ScheduledFomSpec:
            ...

    An undecorated callable is read as ``"rolled"``, which is what a builder
    returning a plain ``FomSpec`` with a ``next_state`` wants and what every
    builder written against the older API is.
    """
    if kind not in FOM_KINDS:
        raise ValueError(f"unknown FOM kind {kind!r}; choices: {list(FOM_KINDS)}")

    def decorate(fn: Callable) -> Callable:
        fn.fom_kind = kind
        return fn

    return decorate


def fom_kind(fom: str | Callable) -> str:
    """The generation path declared by ``fom`` (a built-in name or a callable)."""
    kind = getattr(resolve_builder(fom), "fom_kind", "rolled")
    if kind not in FOM_KINDS:
        raise ValueError(
            f"builder for {fom!r} declares unknown fom_kind {kind!r}; "
            f"choices: {list(FOM_KINDS)}. Declare it with @fom_builder(...)."
        )
    return kind


#: The canonical vector pieces a FOM consumes, in FomParams field order.
PIECES = ("q", "b", "u_ineq")

#: Iterations each built-in builder records so generation can check the method
#: is stationary. One is not enough: a coefficient that varies with the
#: iteration index can agree with a stationary method on the first iteration
#: alone — that is exactly how FISTA's ``gamma_1 = 0`` makes a single recorded
#: iteration indistinguishable from plain ISTA.
_GATE_ITERS = 3


def _param_copy(qp: QPData) -> tuple[str, ...]:
    """Which canonical pieces become runtime parameters, in FomParams order.

    Available without recording anything, which the scheduled path needs: its
    spec is built before the build function ever runs.
    """
    return tuple(piece for piece in PIECES if piece in qp.param_dependent)


def _add_params(v: Verifier, qp: QPData) -> tuple[dict, tuple[str, ...]]:
    """Add runtime parameters for the parametric pieces; bake constants.

    Returns ``(exprs, param_copy)`` where ``exprs`` maps each piece in
    :data:`PIECES` to a ``LinearExpression`` parameter or a constant numpy
    vector, and ``param_copy`` lists the parametric ones in add order.
    """
    values = {"q": qp.q, "b": qp.b, "u_ineq": qp.u_ineq}
    exprs: dict[str, LinearExpression | np.ndarray] = {}
    for piece in PIECES:
        if piece in qp.param_dependent:
            lb, ub = qp.bounds[piece]
            exprs[piece] = v.add_param(values[piece].size, lb=lb, ub=ub)
        else:
            exprs[piece] = values[piece]
    # The add order above is PIECES order, which is what _param_copy reports;
    # sharing it keeps the FomParams field order and the bridge's copies from
    # drifting apart.
    return exprs, _param_copy(qp)


def _const_like(template: LinearExpression, value: np.ndarray) -> LinearExpression:
    """A constant vector as a LinearExpression (zero coefficients + offset)."""
    return 0.0 * template + value


def _record_iterations(one_iteration: Callable, state: tuple) -> tuple:
    """Record :data:`_GATE_ITERS` successive iterations of ``one_iteration``.

    Returns ``(next_state, iterates)``: the iterate after exactly one iteration,
    which is what the rolled extraction reads, and the tape it is checked
    against. The tape holds the recorded iterations *only* — no leading initial
    iterate. The rolled check reads it through
    :func:`~fomtools.codegen.oracle.run_recorded_tape`, which takes every entry
    as an iteration rather than recognizing the leading-initial-iterate
    convention that :func:`~fomtools.codegen.extract.normalize_iterates`
    implements for the unrolled path.

    Recording past the first iteration costs a little simulator time at
    generation and nothing in the emitted solver: extraction takes the
    dependency cone of ``next_state``, so the later steps fall outside it.
    """
    groups: list[list[LinearExpression]] = []
    current = tuple(state)
    for _ in range(_GATE_ITERS):
        current = tuple(one_iteration(*current))
        groups.append(list(current))
    return stack(groups[0]), tuple(groups)


def _gate_param_values(verifier: Verifier, iterates: tuple, state: tuple, params: tuple) -> dict:
    """The parameter draw the stationarity check replays the tape at.

    A draw inside each declared box rather than its midpoint: the boxes here are
    often symmetric, and a zero parameter makes a soft-threshold or projection
    step trivial, hiding exactly the disagreement the check exists to find (see
    :func:`~fomtools.codegen.oracle.sample_param_values`).
    """
    return sample_param_values(
        verifier, list(iterates), state=list(state), params=list(params), seed=0
    )


def _admm_constants(qp: QPData, rho: float, sigma: float, rho_eq_scale: float) -> tuple:
    """The fixed matrices one ADMM iteration needs: ``(R, Rinv, kkt)``.

    ``rho`` is boosted by ``rho_eq_scale`` on the equality rows (the defaults
    follow OSQP). The KKT matrix does not depend on the CVXPY parameters, so its
    factorization is computed once at code-generation time.
    """
    n, m, n_eq = qp.n, qp.m, qp.n_eq
    if m == 0:
        raise ValueError("ADMM builder needs at least one constraint row")

    rho_vec = np.full(m, float(rho))
    rho_vec[:n_eq] *= float(rho_eq_scale)
    Rinv = spa.diags(1.0 / rho_vec)
    kkt = spa.bmat([[qp.P + sigma * spa.eye(n), qp.A.T], [qp.A, -Rinv]], format="csc")
    return spa.diags(rho_vec), Rinv, kkt


def _admm_step(v: Verifier, qp: QPData, exprs: dict, consts: tuple, sigma: float, alpha: float):
    """The ADMM operator ``T: (x, z, y) -> (x, z, y)``, recorded on ``v``.

    Factored out so the plain and anchored builders drive exactly the same
    operator — the anchored one is this ``T`` under a Halpern average, and any
    difference between them would otherwise be a difference in the method rather
    than in the acceleration.
    """
    n, m, n_eq = qp.n, qp.m, qp.n_eq
    R, Rinv, kkt = consts
    q_e, b_e, u_e = exprs["q"], exprs["b"], exprs["u_ineq"]

    def one_iteration(x, z, y):
        rhs = stack([sigma * x - q_e, z - Rinv @ y])
        sol = v.step.implicit_linear(rhs, output_mat=kkt)
        x_tilde = sol[:n]
        nu = sol[n : n + m]

        z_tilde = z + Rinv @ (nu - y)
        proj_arg = alpha * z_tilde + (1.0 - alpha) * z + Rinv @ y

        # Projection onto the constraint set, row group by row group: equality
        # rows project onto the point b; inequality rows onto (-inf, u].
        parts: list[LinearExpression] = []
        if n_eq > 0:
            b_expr = b_e if isinstance(b_e, LinearExpression) else _const_like(proj_arg[:n_eq], b_e)
            parts.append(b_expr)
        if m - n_eq > 0:
            w = proj_arg[n_eq:m]
            parts.append(w - v.step.relu(w - u_e))
        z_next = stack(parts)

        x_next = alpha * x_tilde + (1.0 - alpha) * x
        y_next = y + R @ (alpha * z_tilde + (1.0 - alpha) * z - z_next)
        return x_next, z_next, y_next

    return one_iteration


@fom_builder("rolled")
def build_admm_qp(
    qp: QPData,
    *,
    rho: float = 0.1,
    sigma: float = 1e-6,
    alpha: float = 1.6,
    rho_eq_scale: float = 1e3,
) -> FomSpec:
    """One iteration of ADMM for the canonical QP (KKT solve + projection).

    State is ``(x, z, y)`` (primal, auxiliary, dual). The KKT matrix
    ``[[P + sigma I, A'], [A, -diag(1/rho_vec)]]`` is constant, so its
    factorization is computed at code-generation time; ``rho`` is boosted by
    ``rho_eq_scale`` on equality rows (the parameter defaults follow OSQP).
    The dual iterate ``y`` is exactly the dual of the canonical form, so
    CVXPY's duals are recovered from it directly.
    """
    n, m = qp.n, qp.m
    consts = _admm_constants(qp, rho, sigma, rho_eq_scale)

    v = Verifier(method="simulator", obbt_solver=None)
    exprs, param_copy = _add_params(v, qp)

    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    z = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
    y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

    one_iteration = _admm_step(v, qp, exprs, consts, sigma, alpha)
    state = (x, z, y)
    next_state, iterates = _record_iterations(one_iteration, state)
    params = tuple(exprs[piece] for piece in param_copy)
    return FomSpec(
        verifier=v,
        next_state=next_state,
        name="cvxpygen_admm",
        primal_offset=0,
        primal_dim=n,
        dual_offset=n + m,
        dual_dim=m,
        param_copy=param_copy,
        state=state,
        params=params,
        iterates=iterates,
        param_values=_gate_param_values(v, iterates, state, params),
    )


def _relaxation_ramp(start: float, final: float, ramp_iters: int) -> Callable[[int], float]:
    """A relaxation that moves linearly from ``start`` to ``final``, then holds.

    Indexed by the **0-based** loop counter, as the emitted accessor is.
    """

    def relaxation(k: int) -> float:
        progress = 1.0 if ramp_iters <= 0 else min(1.0, k / float(ramp_iters))
        return start + (final - start) * progress

    return relaxation


@fom_builder("scheduled")
def build_relaxed_admm_qp(
    qp: QPData,
    *,
    rho: float = 0.1,
    sigma: float = 1e-6,
    rho_eq_scale: float = 1e3,
    alpha_start: float = 1.0,
    alpha_final: float = 1.9,
    ramp_iters: int = 50,
) -> ScheduledFomSpec:
    r"""ADMM whose over-relaxation ``alpha_k`` varies with the iteration index.

    The operator is exactly :func:`build_admm_qp`'s; the only difference is that
    its relaxation is a schedule instead of a constant, ramping linearly from
    ``alpha_start`` to ``alpha_final`` over ``ramp_iters`` iterations and holding
    there. Douglas-Rachford converges for any relaxation sequence inside
    ``(0, 2)`` that stays away from the endpoints (Krasnoselskii-Mann with
    ``sum_k alpha_k (2 - alpha_k) = inf``), which the defaults satisfy, so this
    is safe on any QP the plain builder handles.

    This is the built-in that exercises the *scheduled* path, and it is the
    honest reason to reach for it: ``alpha_k`` enters every recorded expression
    linearly, so extraction recovers each coefficient block as an affine
    function of it and emits one block plus a table read at the loop counter.
    The solver is the same size and carries the same three state slots as the
    plain builder however long it runs, and the relaxation is *not* frozen at
    whatever value the first iteration happened to use — which is what would
    happen if a varying coefficient were recorded down the rolled path.

    Ramping is a heuristic, not an acceleration guarantee: being conservative
    while the active set is still moving and aggressive afterwards helps on some
    problems and not others. Tune it, or pass ``alpha_start == alpha_final`` for
    a constant relaxation.
    """
    for name, value in (("alpha_start", alpha_start), ("alpha_final", alpha_final)):
        if not 0.0 < value < 2.0:
            raise ValueError(
                f"{name}={value} is outside (0, 2), where the relaxed ADMM iteration "
                "is no longer guaranteed to converge"
            )

    n, m = qp.n, qp.m
    consts = _admm_constants(qp, rho, sigma, rho_eq_scale)
    param_copy = _param_copy(qp)

    def build(relaxation):
        v = Verifier(method="simulator", obbt_solver=None)
        exprs, _ = _add_params(v, qp)

        x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
        z = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
        y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

        # The recorded relaxation is the probe scalar rather than a constant;
        # everything else is the plain builder's iteration, unchanged.
        tx, tz, ty = _admm_step(v, qp, exprs, consts, sigma, relaxation)(x, z, y)
        return Recording(
            v,
            next_state=[tx, tz, ty],
            state=[x, z, y],
            params=[exprs[piece] for piece in param_copy],
        )

    return ScheduledFomSpec(
        build=build,
        schedules={"relaxation": _relaxation_ramp(alpha_start, alpha_final, ramp_iters)},
        name="cvxpygen_relaxed_admm",
        primal_offset=0,
        primal_dim=n,
        dual_offset=n + m,
        dual_dim=m,
        param_copy=param_copy,
    )


@fom_builder("rolled")
def build_pdhg_lp(qp: QPData, *, step_scale: float = 0.9) -> FomSpec:
    """One iteration of PDHG (Chambolle-Pock) for the canonical LP.

    Requires ``P == 0``. Pure first-order: matrix-vector products and a relu
    projection, no linear-system solve — suited to large or factorization-free
    embedded targets. Step sizes ``tau = sigma_step = step_scale / ||A||_2``
    are computed offline.
    """
    if qp.P.nnz != 0:
        raise ValueError(
            "the PDHG builder requires a zero quadratic term (LP); use fom='admm' for QPs"
        )
    n, m, n_eq = qp.n, qp.m, qp.n_eq
    if m == 0:
        raise ValueError("PDHG builder needs at least one constraint row")

    norm_A = float(spa.linalg.norm(qp.A, 2)) if min(qp.A.shape) > 1 else float(abs(qp.A).max())
    if norm_A <= 0:
        raise ValueError("canonical A has zero norm")
    tau = sig = step_scale / norm_A

    v = Verifier(method="simulator", obbt_solver=None)
    exprs, param_copy = _add_params(v, qp)
    q_e, b_e, u_e = exprs["q"], exprs["b"], exprs["u_ineq"]

    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))

    def one_iteration(x, y):
        # Primal step: the prox of a linear objective is a shift.
        x_next = x - tau * (q_e + qp.A.T @ y)

        # Dual step via Moreau: y+ = v - sig * proj_[l,u](v / sig) with
        # v = y + sig * A (2 x+ - x), split into equality / inequality rows.
        v_arg = y + sig * (qp.A @ (2.0 * x_next - x))
        parts: list[LinearExpression] = []
        if n_eq > 0:
            parts.append(v_arg[:n_eq] - sig * b_e)
        if m - n_eq > 0:
            v_in = v_arg[n_eq:m]
            # proj onto (-inf, u]: w - relu(w - u) => y+ = sig * relu(v/sig - u).
            parts.append(sig * v.step.relu((1.0 / sig) * v_in - u_e))
        return x_next, stack(parts)

    state = (x, y)
    next_state, iterates = _record_iterations(one_iteration, state)
    params = tuple(exprs[piece] for piece in param_copy)
    return FomSpec(
        verifier=v,
        next_state=next_state,
        name="cvxpygen_pdhg",
        primal_offset=0,
        primal_dim=n,
        dual_offset=n,
        dual_dim=m,
        param_copy=param_copy,
        state=state,
        params=params,
        iterates=iterates,
        param_values=_gate_param_values(v, iterates, state, params),
    )


FomBuilder = Callable[..., BaseFomSpec]

BUILDERS: dict[str, FomBuilder] = {
    "admm": build_admm_qp,
    "relaxed_admm": build_relaxed_admm_qp,
    "pdhg": build_pdhg_lp,
}


def resolve_builder(fom: str | FomBuilder) -> FomBuilder:
    """Resolve the ``fom=`` argument: a built-in name or a user callable."""
    if callable(fom):
        return fom
    try:
        return BUILDERS[fom]
    except KeyError:
        raise ValueError(
            f"unknown FOM {fom!r}; built-in choices: {sorted(BUILDERS)} "
            "(or pass a callable qp -> FomSpec for the rolled/unrolled paths, "
            "qp -> ScheduledFomSpec for the scheduled one)"
        ) from None
