import logging
import warnings
from typing import Callable, ClassVar

import numpy as np
import scipy.sparse as spa

from fomtools.constraint import complementarity
from fomtools.linearexpression import LinearExpression, solve_linexpr, stack
from fomtools.proxable_functions import (
    ElasticNet,
    HingeLoss,
    HuberLoss,
    IndAffine,
    IndBallL1,
    IndBallLinf,
    IndBox,
    IndHalfSpace,
    IndNonNegative,
    IndNonPositive,
    IndPoint,
    IndPolyhedral,
    IndSimplex,
    LabeledHingeLoss,
    Linear,
    NormInf,
    NormL1,
    Quadratic,
    SoftNonNegative,
    SqL2Norm,
    SumNegative,
    SumPositive,
    TotalVariation1D,
)
from fomtools.proxable_functions.total_variation_1d import _SHORT_SIGNAL_MSG, diff_matrix
from fomtools.steps import Relu, SaturatedLinear, SoftThreshold
from fomtools.steps.step import Step

log = logging.getLogger(__name__)


def _is_identity_matrix(P) -> bool:
    """True if ``P`` is ``None`` or (numerically) the identity (dense or sparse)."""
    if P is None:
        return True
    n = P.shape[0]
    if spa.issparse(P):
        diff = P - spa.eye(n, format="csr")
        return diff.nnz == 0 or float(np.abs(diff).max()) <= 1e-12
    return np.allclose(np.asarray(P, dtype=float), np.eye(n))


def _finite_bounds(expr: LinearExpression) -> tuple[np.ndarray, np.ndarray]:
    """Interval-propagate ``expr`` bounds, mapping non-finite entries to +/- inf.

    Unbounded duals (e.g. a free ``nu`` with +/- inf bounds) make interval
    arithmetic yield ``inf - inf = NaN``; map those back to +/- inf so the leaf
    still registers and any downstream complementarity raises a clean finite-bound
    error rather than propagating NaN.
    """
    with np.errstate(invalid="ignore"):
        lb, ub = expr.bound_propagation()
    lb = np.where(np.isfinite(lb), lb, -np.inf)
    ub = np.where(np.isfinite(ub), ub, np.inf)
    return lb, ub


class StepDecomposer:
    """
    Base class for step decomposition and prox lowering.

    Decomposes composite steps into primitive steps that the canonicalizer
    can handle directly, and lowers proxable functions into steps + aux
    variables. Returns LinearExpression (leaf or composite).

    Subclasses declare two class-level registry dicts:

    - ``_DECOMPOSERS`` maps the lowercase step class name
      (e.g. ``"softthreshold"``) to the function that decomposes it.
      Steps absent from the registry are treated as primitives and
      processed directly via the canonicalizer.
    - ``_PROX_DECOMPOSERS`` maps the lowercase proxable function class
      name (e.g. ``"norml1"``) to the function that lowers it into
      steps. Proxes absent from the registry raise ``NotImplementedError``
      — useful for backends that cannot encode a given prox (e.g.
      ``TotalVariation1D`` on MIP/SDP).

    Subclasses that need to *extend* (rather than replace) an inherited
    registry should spread the parent dict explicitly, for example::

        class SimulatorStepDecomposer(StepDecomposer):
            _PROX_DECOMPOSERS: ClassVar[dict] = {
                **StepDecomposer._PROX_DECOMPOSERS,
                "totalvariation1d": prox_totalvariation1d,
            }

    Parameters
    ----------
    canonicalizer : Canonicalizer
        The canonicalizer instance to use for encoding constraints.
    register_var : Callable[..., None]
        Callback function to register new leaf variables with both the
        canonicalizer and OBBT handler.
    add_iterate : Callable[..., LinearExpression]
        Callback to introduce a fresh iterate leaf (the new iterate
        produced by a prox step). Used by KKT-style prox lowerings
        that define their output implicitly via an equality constraint
        rather than via a ``Step`` activation. Must track the new leaf
        in ``Verifier.iterates``.
    add_constraint : Callable[..., None]
        Callback to post an arbitrary constraint, routed through both
        the main canonicalizer and (when enabled) the OBBT handler.
        Used by KKT-style prox lowerings.

    Attributes
    ----------
    canonicalizer : Canonicalizer
        The canonicalizer instance used for constraint encoding.
    register_var : Callable[..., None]
        Callback for registering new leaf variables.
    add_iterate : Callable[..., LinearExpression]
        Callback for introducing a fresh iterate leaf.
    add_constraint : Callable[..., None]
        Callback for posting an OBBT-aware constraint.
    """

    _DECOMPOSERS: ClassVar[dict] = {}

    def __init__(
        self,
        canonicalizer,
        register_var: Callable[..., None],
        add_iterate: Callable[..., LinearExpression],
        add_constraint: Callable[..., None],
    ):
        self.canonicalizer = canonicalizer
        self.register_var = register_var
        self.add_iterate = add_iterate
        self.add_constraint = add_constraint

    def _lookup_decomposer(self, step: Step):
        """
        Return the first registered decomposer along the step's MRO, or ``None``.

        Walking ``type(step).__mro__`` (rather than matching the exact class name)
        lets a subclass inherit its parent's decomposer — e.g.
        ``StandardFormLinearProgram`` resolves to the ``"linearprogram"`` entry — without
        a dedicated registry key. Behaviour is unchanged for registered classes: each is
        first in its own MRO, so it still matches itself.
        """
        for cls in type(step).__mro__:
            decomposer = self._DECOMPOSERS.get(cls.__name__.lower())
            if decomposer is not None:
                return decomposer
        return None

    def is_primitive(self, step: Step) -> bool:
        """
        Return True if ``step`` has no registered decomposition for this
        decomposer and should be processed directly as a primitive.
        """
        return self._lookup_decomposer(step) is None

    def process_step(self, step: Step) -> LinearExpression:
        """
        Process a step, decomposing if needed.

        For primitive steps: hands off to ``_process_primitive``, which
        handles OBBT input tightening, canonicalization, and OBBT
        relaxation registration in one place.
        For composite steps: dispatches to the registered decomposer,
        which builds sub-steps and recurses through ``process_step``.

        Parameters
        ----------
        step : Step
            The step to process.

        Returns
        -------
        LinearExpression
            The output expression. For primitive steps, this is step.output.
            For composite steps, this is a LinearExpression composed from
            the outputs of the decomposed primitive steps.
        """
        decomposer = self._lookup_decomposer(step)
        if decomposer is None:
            return self._process_primitive(step)
        return decomposer(self, step)

    def process_prox(
        self,
        prox_function,
        arg: LinearExpression,
        gamma: float = 1.0,
    ) -> LinearExpression:
        """
        Lower a proxable function into steps + aux variables.

        Looks up the lowercase prox class name (e.g. ``"norml1"``) in
        this decomposer's ``_PROX_DECOMPOSERS`` registry and invokes the
        registered lowering function. Raises ``NotImplementedError`` if
        the prox is not supported on this backend.

        Parameters
        ----------
        prox_function : ProxableFunction
            A pre-configured proxable function instance carrying problem
            data (e.g. the ``lambd`` of a weighted L1 norm).
        arg : LinearExpression
            The input expression to which the prox is applied.
        gamma : float, default 1.0
            The proximal step size.

        Returns
        -------
        LinearExpression
            The output expression ``prox_{gamma * f}(arg)``.

        Raises
        ------
        NotImplementedError
            If this decomposer's ``_PROX_DECOMPOSERS`` registry has no
            entry for the concrete type of ``prox_function``.
        """
        proxer = self._PROX_DECOMPOSERS.get(type(prox_function).__name__.lower())
        if proxer is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support {type(prox_function).__name__}"
            )
        return proxer(self, prox_function, arg, gamma)

    def _decompose_saturatedlinear_to_relus(self, step: SaturatedLinear) -> LinearExpression:
        """
        Decompose saturated linear into ReLU operations.

        Uses the identity:
            satlin(x, l, u) = relu(x - l) - relu(x - u) + l

        This works regardless of whether l and u are constant arrays or
        LinearExpression parameters — the LinearExpression arithmetic
        handles both types transparently. The only type-dependent part
        is extracting bounds for ReLU input interval propagation.

        Parameters
        ----------
        step : SaturatedLinear
            The saturated linear step to decompose.

        Returns
        -------
        LinearExpression
            A composite expression representing the saturated linear
            output as relu(x - l) - relu(x - u) + l.
        """
        x_lb, x_ub = step.lb, step.ub

        # Extract bounds for l and u. bound_propagation() returns numpy arrays
        # for any LinearExpression (leaf or composite). Constants don't have
        # this method, so we fall back to using the value directly.
        try:
            l_lb, l_ub = step.l.bound_propagation()
        except AttributeError:
            l_lb = l_ub = np.atleast_1d(step.l)

        try:
            u_lb, u_ub = step.u.bound_propagation()
        except AttributeError:
            u_lb = u_ub = np.atleast_1d(step.u)

        # Create ReLU steps using step.l/step.u directly (type-agnostic arithmetic)
        relu_lower = Relu(arg=step.arg - step.l)
        relu_upper = Relu(arg=step.arg - step.u)

        # Bounds on (arg - l): [x_lb - l_ub, x_ub - l_lb]
        relu_lower.update_bounds(x_lb - l_ub, x_ub - l_lb)
        # Bounds on (arg - u): [x_lb - u_ub, x_ub - u_lb]
        relu_upper.update_bounds(x_lb - u_ub, x_ub - u_lb)

        output_lower = self.process_step(relu_lower)
        output_upper = self.process_step(relu_upper)

        # satlin(x, l, u) = relu(x - l) - relu(x - u) + l
        return output_lower - output_upper + step.l

    def _process_primitive(self, step: Step) -> LinearExpression:
        """
        Default handling for primitive steps.

        When an OBBT handler is attached to the canonicalizer, this first
        tightens the step's *input* bounds against the OBBT LP (built from the
        relaxations of all previously processed primitives), then registers the
        output variable, canonicalizes into the main backend, and finally adds
        this primitive's own binary-free linear relaxation to the OBBT LP so
        that subsequent steps can be tightened against it.

        Because this hook runs once per *finalized primitive* — including each
        ReLU that the SDP/MIP decomposers produce — the OBBT relaxation always
        lands on the same leaf objects the backend uses, so every leaf
        (including decomposition auxiliaries) is tightened.

        Parameters
        ----------
        step : Step
            A primitive step to process directly.

        Returns
        -------
        LinearExpression
            The step's left-hand side (output) variable.
        """
        obbt = getattr(self.canonicalizer, "obbt_handler", None)

        # Tighten the input bounds BEFORE the output box and envelope
        # coefficients (which bake in lb/ub) are computed.
        if obbt is not None and not step.skip_obbt:
            result = obbt.obbt(step.arg, step.lb, step.ub)
            if result is not None:
                step.update_bounds(*result)

        if hasattr(step, "sync_output_bounds"):
            step.sync_output_bounds()
        # registered as "param" for MIP/SDP compatibility; Simulator registers
        # step output as an iterate with zero init so subsequent steps can reference it.
        # For steps whose output is a composite view (e.g. partial-projection Relu),
        # output_leaves names only the fresh leaves that actually need solver-side
        # variables; non-fresh rows reuse arg leaves and are already registered.
        for leaf in step.output_leaves:
            self.register_var(leaf, var_type="param")

        self.canonicalizer.canonicalize(step)

        # Add this primitive's binary-free relaxation to the OBBT LP for use by
        # subsequent steps. Affine equalities are added too (skip_obbt only
        # gates input tightening, not the relaxation). If a step type has no
        # OBBT relaxation registered, degrade gracefully: the OBBT LP simply
        # omits this step's constraints (looser, still-valid bounds) rather
        # than crashing problem construction.
        if obbt is not None:
            try:
                obbt.canonicalize(step)
            except NotImplementedError:
                log.warning(
                    "OBBT: no relaxation for %s; tightening may be weaker",
                    type(step).__name__,
                )

        return step.output

    @staticmethod
    def _to_satlin_bound(value, ndim: int):
        """
        Broadcast a scalar ``l``/``u`` to an array of length ``ndim``;
        pass arrays and ``LinearExpression`` parameters through.

        The MIP canonicalizer iterates over ``step.l`` and ``step.u`` per
        coordinate (``mip.py:603``), so scalars must be expanded before
        construction.
        """
        if hasattr(value, "lb"):  # LinearExpression — pass through
            return value
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(ndim, float(arr))
        return arr

    def prox_elasticnet(
        self, prox: ElasticNet, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower an elastic-net prox to a ``SoftThreshold`` step plus a
        scalar shrinkage on the output.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma (\alpha\|\cdot\|_1 + \beta\|\cdot\|_2^2)}(v)
            \;=\; \frac{S_{\alpha\gamma}(v)}{1 + 2\beta\gamma}.
        $$

        Backend-independent: registered on the base class so all step
        decomposers inherit it. The ``SoftThreshold`` is recursively
        decomposed per-backend (e.g. SDP turns it into two ReLUs via
        ``_DECOMPOSERS``); the $1/(1 + 2\beta\gamma)$ factor applies
        analytically to the resulting composite ``LinearExpression``.
        """
        soft = self.process_step(SoftThreshold(arg=arg, lambd=prox.alpha * gamma))
        return soft / (1.0 + 2.0 * prox.beta * gamma)

    def prox_hingeloss(
        self, prox: HingeLoss, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the standard hinge prox $g(x) = \sum_i \max(0, 1 - x_i)$
        to a single ``SaturatedLinear`` clip on the margin residual
        $1 - \text{arg}$ to $[0, \gamma]$, added back to ``arg``:

        $$
            \operatorname{prox}_{\gamma g}(v) = v + \mathrm{satlin}(1 - v;\, 0,\, \gamma).
        $$

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        residual = -arg + 1.0
        l = self._to_satlin_bound(0.0, arg.ndim)
        u = self._to_satlin_bound(gamma, arg.ndim)
        clip = self.process_step(SaturatedLinear(arg=residual, l=l, u=u))
        return arg + clip

    def prox_huberloss(
        self, prox: HuberLoss, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower an element-wise Huber prox to a single ``SoftThreshold``
        step plus arithmetic on the output.

        Uses the closed form

        $$
            \operatorname{prox}_{\gamma h}(v)
                \;=\; \frac{v + \gamma\, S_{\delta(1+\gamma)}(v)}{1+\gamma},
        $$

        where the bracketed soft threshold $S_\theta(w)
        = \operatorname{ReLU}(w-\theta) - \operatorname{ReLU}(-w-\theta)$
        encodes the two-ReLU tail behavior. This is one native primitive
        on MIP and Simulator; on SDP it decomposes recursively into two
        ReLUs via ``_DECOMPOSERS``.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        theta = prox.delta * (1.0 + gamma)
        soft = self.process_step(SoftThreshold(arg=arg, lambd=theta))
        return (arg + gamma * soft) / (1.0 + gamma)

    def prox_indaffine(
        self, prox: IndAffine, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the affine-set indicator prox $I_{\{Ax = b\}}$ to a
        two-iterate KKT system on a primal $u$ and an equality
        multiplier $\nu$:

        $$
            \begin{bmatrix} I & A^\top \\ A & 0 \end{bmatrix}
            \begin{bmatrix} u \\ \nu \end{bmatrix}
            =
            \begin{bmatrix} \text{arg} \\ b \end{bmatrix}.
        $$

        Independent of $\gamma$ (indicator prox). This is the equality-only case of
        the generic QP (``P = I``, ``q = -arg``, no inequalities), so it routes through
        :meth:`_qp_kkt`, which posts ``x == arg - A^T nu`` and ``A x == b`` (no
        complementarity — there are no inequality constraints). The tight initial
        bounds on ``u`` and ``ν`` are still derived here via interval propagation
        through a throwaway ``LinearExpression`` that represents $M_{\text{block}}^{-1}$
        applied to the stacked right-hand side (built via :func:`solve_linexpr`) and
        passed to ``_qp_kkt`` as ``primal_bounds`` / ``nu_bounds``; OBBT tightens them
        further. ``b`` may be numeric data or a ``LinearExpression`` parameter (lifted
        uniformly via :meth:`_lift_to_le`).

        Backend-independent: registered on the base class. Simulator
        defers (no per-step activation callback for an implicitly
        defined iterate); see the Simulator override.
        """
        A = prox.A
        m, n = prox.m, prox.n
        b_le = self._lift_to_le(prox.b, m)

        # Block KKT matrix (constant; just numpy) for deriving joint (u, nu) bounds.
        M_block = np.block(
            [
                [np.eye(n), A.T],
                [A, np.zeros((m, m))],
            ]
        )
        # Throwaway LE representing M_block^{-1} @ [arg; b]; never registered.
        combined_lb, combined_ub = solve_linexpr(M_block, stack([arg, b_le])).bound_propagation()

        return self._qp_kkt(
            lin_term=-arg,
            P=None,
            A_eq=A,
            b_eq=b_le,
            primal_bounds=(combined_lb[:n], combined_ub[:n]),
            nu_bounds=(combined_lb[n:], combined_ub[n:]),
        )

    def _lift_to_le(self, value, dim: int) -> LinearExpression:
        """
        Lift a scalar/array to a constant ``LinearExpression``; pass LE params through.

        Parameters
        ----------
        value : float, np.ndarray, or LinearExpression
            The right-hand side value to lift.
        dim : int
            Target dimension for scalar/array broadcasting.
        """
        if hasattr(value, "decomposition_dict"):
            return value
        return LinearExpression(
            decomposition_dict={},
            const=np.broadcast_to(np.asarray(value, dtype=float), (dim,)).copy(),
        )

    def _qp_kkt(
        self,
        *,
        lin_term: LinearExpression,
        P=None,
        is_lp: bool = False,
        A_eq=None,
        b_eq=None,
        C_ineq=None,
        d_ineq=None,
        lam_ub=None,
        nu_bounds=None,
        primal_bounds=None,
    ) -> LinearExpression:
        r"""
        Shared KKT lowering for the convex QP
        :math:`\min_x \tfrac12 x^\top P x + q^\top x` s.t. :math:`A_{eq} x = b_{eq}`,
        :math:`C_{ineq} x \le d_{ineq}`.

        Posts the KKT system

        .. math::
            P x + q + A_{eq}^\top \nu + C_{ineq}^\top \lambda = 0,\quad
            A_{eq} x = b_{eq},\quad C_{ineq} x \le d_{ineq},\quad
            0 \le \lambda \perp (d_{ineq} - C_{ineq} x) \ge 0,

        with :math:`\nu` free and :math:`\lambda \ge 0`. The linear rows precede the
        (non-convex) complementarity so OBBT tightens ``x``/``lambda`` first. ``P``
        may be ``None``/identity (stationarity becomes ``x = -(q + ...)``), a general
        PSD matrix (``P @ x == -(q + ...)``), or all-zero for an LP
        (``is_lp=True``: stationarity is the dual-only equality and ``x`` is pinned by
        feasibility + complementarity).

        Parameters
        ----------
        lin_term : LinearExpression
            The linear cost ``q`` (dim ``n``); for a projection pass ``-arg``.
        P : np.ndarray or scipy.sparse, optional
            The PSD quadratic term. ``None`` or identity uses the ``x = -(...)`` form.
        is_lp : bool, default False
            If ``True``, ``P = 0``: stationarity does not define ``x``.
        A_eq, b_eq : optional
            Equality matrix ``(p, n)`` and right-hand side (dim ``p``).
        C_ineq, d_ineq : optional
            Inequality matrix ``(m, n)`` and right-hand side (dim ``m``).
        lam_ub : np.ndarray, optional
            Finite upper bound(s) on the inequality dual ``lambda``, shape ``(m,)``.
        nu_bounds : tuple[np.ndarray, np.ndarray], optional
            ``(lb, ub)`` on the free equality multiplier ``nu``; ``None`` -> +/- inf.
        primal_bounds : tuple[np.ndarray, np.ndarray], optional
            ``(lb, ub)`` on ``x``; required for singular ``P``/LP when OBBT cannot bound it.

        Returns
        -------
        LinearExpression
            The primal solution ``x`` of dimension ``n``.
        """
        n = lin_term.ndim
        stat = lin_term  # accumulates q + C^T lam + A^T nu
        lam = None
        if C_ineq is not None:
            m = C_ineq.shape[0]
            lam = self.add_iterate(m, lb=np.zeros(m), ub=lam_ub)
            stat = stat + C_ineq.T @ lam
        if A_eq is not None:
            p = A_eq.shape[0]
            if nu_bounds is None:
                nu_lb, nu_ub = np.full(p, -np.inf), np.full(p, np.inf)
            else:
                nu_lb, nu_ub = nu_bounds
            nu = self.add_iterate(p, lb=nu_lb, ub=nu_ub)
            stat = stat + A_eq.T @ nu

        identity = _is_identity_matrix(P)

        # Primal bounds for the fresh iterate ``x``.
        if primal_bounds is not None:
            x_lb, x_ub = primal_bounds
        elif is_lp:
            # P = 0: stationarity does not pin x; rely on primal_bounds/OBBT.
            x_lb, x_ub = np.full(n, -np.inf), np.full(n, np.inf)
        elif identity:
            x_lb, x_ub = _finite_bounds(-stat)  # x == -stat
        else:
            try:
                x_lb, x_ub = _finite_bounds(solve_linexpr(P, -stat))  # P x == -stat (PD)
            except Exception as exc:
                raise ValueError(
                    "Could not derive primal bounds for a non-identity P (it may be "
                    "singular); pass primal_bounds=(lb, ub) to QuadraticProgram."
                ) from exc
        x = self.add_iterate(n, lb=x_lb, ub=x_ub)

        # Stationarity row.
        if is_lp:
            self.add_constraint(stat == 0)  # dual-only equality (no x)
        elif identity:
            self.add_constraint(x == -stat)
        else:
            self.add_constraint(P @ x == -stat)  # P @ x: __rmatmul__, sparse-ok

        # Primal feasibility (linear, OBBT-visible) before the complementarity.
        if A_eq is not None:
            self.add_constraint(A_eq @ x == b_eq)
        if C_ineq is not None:
            self.add_constraint(C_ineq @ x <= d_ineq)
            self.add_constraint(complementarity(lam, d_ineq - C_ineq @ x))
        return x

    def _prox_polyhedral_kkt(
        self, arg, A, b_le, lam_ub, C=None, d_le=None, nu_bounds=None
    ) -> LinearExpression:
        r"""
        Projection onto ``{Au <= b}`` and an optional ``{Cu = d}``, as the QP
        ``min (1/2)||u - arg||^2`` over that polyhedron. A thin wrapper over
        :meth:`_qp_kkt` with ``P = I`` and ``q = -arg``.

        Note the naming inversion preserved for the existing callers: here ``A`` is the
        **inequality** matrix and ``C`` the **equality** matrix (the opposite of the
        general-QP ``A_eq``/``C_ineq`` convention used by :meth:`_qp_kkt`).

        Parameters
        ----------
        arg : LinearExpression
            The point being projected.
        A : np.ndarray
            Inequality matrix, shape ``(m, n)``.
        b_le : LinearExpression
            Inequality right-hand side, dim ``m``.
        lam_ub : np.ndarray
            Finite upper bound(s) on the inequality dual ``lambda``, shape ``(m,)``.
        C : np.ndarray, optional
            Equality matrix, shape ``(p, n)``. ``None`` for inequality-only.
        d_le : LinearExpression, optional
            Equality right-hand side, dim ``p``.
        nu_bounds : tuple[np.ndarray, np.ndarray], optional
            ``(lb, ub)`` for the free equality multiplier ``nu``; required when ``C`` is given.

        Returns
        -------
        LinearExpression
            The primal projection ``u`` of dimension ``n``.
        """
        return self._qp_kkt(
            lin_term=-arg,
            P=None,
            C_ineq=A,
            d_ineq=b_le,
            lam_ub=lam_ub,
            A_eq=C,
            b_eq=d_le,
            nu_bounds=nu_bounds,
        )

    def decompose_quadraticprogram(self, step) -> LinearExpression:
        r"""
        Lower a :class:`QuadraticProgram` / :class:`LinearProgram` step to its KKT
        system via :meth:`_qp_kkt`.

        ``q`` defaults to ``-step.arg`` (projection convenience) when ``None``;
        ``b`` and ``d`` are lifted to ``LinearExpression`` via :meth:`_lift_to_le`
        (numeric or parameter, uniformly). Raises ``ValueError`` eagerly if the QP has
        inequality constraints but no finite ``dual_ub`` (the inequality dual is not
        auto-boundable). Registered in the MIP and SDP ``_DECOMPOSERS`` under both
        ``"quadraticprogram"`` and ``"linearprogram"``.
        """
        q = step.q if step.q is not None else -step.arg
        q_le = self._lift_to_le(q, step.n)

        A_eq, b_eq = step.A, None
        if A_eq is not None:
            b_eq = self._lift_to_le(step.b, A_eq.shape[0])

        C_ineq, d_ineq, lam_ub = step.C, None, None
        if C_ineq is not None:
            if step.dual_ub is None:
                raise ValueError(
                    "QuadraticProgram/LinearProgram with inequality constraints requires a "
                    "finite `dual_ub` on the inequality multipliers (the dual is not "
                    "auto-boundable). Construct it with dual_ub=<finite bound>."
                )
            d_ineq = self._lift_to_le(step.d, C_ineq.shape[0])
            lam_ub = np.broadcast_to(
                np.asarray(step.dual_ub, dtype=float), (C_ineq.shape[0],)
            ).copy()

        return self._qp_kkt(
            lin_term=q_le,
            P=step.P,
            is_lp=getattr(step, "is_lp", False),
            A_eq=A_eq,
            b_eq=b_eq,
            C_ineq=C_ineq,
            d_ineq=d_ineq,
            lam_ub=lam_ub,
            nu_bounds=step.nu_bounds,
            primal_bounds=step.primal_bounds,
        )

    def prox_indpolyhedral(
        self, prox: IndPolyhedral, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the polyhedral-set indicator prox :math:`\operatorname{proj}_{\{Au \le b\}}`
        via :meth:`_prox_polyhedral_kkt` (inequality-only). The complementarity big-M needs
        a finite bound on ``lambda``; the user must supply it via ``prox.dual_ub`` (the
        LP-relaxed dual of a general polytope is unbounded, so it cannot be auto-derived).
        Raises ``ValueError`` eagerly if ``dual_ub`` is ``None``.

        Parameters
        ----------
        prox : IndPolyhedral
            The polyhedral indicator carrying ``A``, ``b`` and a required ``dual_ub``.
        arg : LinearExpression
            The point being projected.
        gamma : float, default 1.0
            Step size (irrelevant for an indicator).

        Returns
        -------
        LinearExpression
            The primal projection ``u``.
        """
        A, m = prox.A, prox.m
        if prox.dual_ub is None:
            raise ValueError(
                "IndPolyhedral requires a finite `dual_ub`: the dual of a general polyhedron "
                "is not auto-boundable (its LP relaxation is unbounded). Construct it as "
                "IndPolyhedral(A, b, dual_ub=<finite bound on the optimal multipliers>)."
            )
        b_le = self._lift_to_le(prox.b, m)
        lam_ub = np.broadcast_to(np.asarray(prox.dual_ub, dtype=float), (m,)).copy()
        return self._prox_polyhedral_kkt(arg, A, b_le, lam_ub)

    def prox_indsimplex(
        self, prox: IndSimplex, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the simplex indicator prox :math:`\operatorname{proj}_{\{u \ge 0,\ 1^\top u = a\}}`
        as the mixed polyhedron ``{Au <= b, Cu = d}`` with ``A = -I``, ``b = 0``, ``C = 1^T``,
        ``d = a``, via :meth:`_prox_polyhedral_kkt`.

        The simplex is bounded, so the duals are bounded and derived analytically from the
        input box (no user ``dual_ub``): with ``u_i = (x_i - nu)_+`` and ``lambda_i =
        (nu - x_i)_+``,

        - ``lambda_i <= max_i x_i - min_i x_i`` over the box, and
        - ``nu`` (the relu threshold) ``in [min(arg_lb) - a_max, max(arg_ub)]``.

        Parameters
        ----------
        prox : IndSimplex
            The simplex indicator carrying the sum ``a`` (scalar or dim-1 LinearExpression).
        arg : LinearExpression
            The point being projected.
        gamma : float, default 1.0
            Step size (irrelevant for an indicator).

        Returns
        -------
        LinearExpression
            The primal projection ``u`` of dimension ``n``.
        """
        n = arg.ndim
        a_le = self._lift_to_le(prox.a, 1)
        a_max = float(a_le.bound_propagation()[1][0])

        arg_lb, arg_ub = arg.bound_propagation()
        lo, hi = float(np.min(arg_lb)), float(np.max(arg_ub))
        # Per-component: lambda_i = (nu - x_i)_+ <= hi - arg_lb[i] (nu <= max_j x_j <= hi),
        # tighter than the global hi - lo for heterogeneous boxes.
        lam_ub = hi - arg_lb
        nu_bounds = (np.array([lo - a_max]), np.array([hi]))  # nu = tau in [min x - a, max x]

        A = -np.eye(n)
        b_le = self._lift_to_le(0.0, n)  # u >= 0  (-u <= 0)
        C = np.ones((1, n))
        return self._prox_polyhedral_kkt(arg, A, b_le, lam_ub, C=C, d_le=a_le, nu_bounds=nu_bounds)

    def prox_indballl1(
        self, prox: IndBallL1, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the :math:`\ell_1`-ball indicator prox onto the lifted KKT system.

        Projection onto :math:`\{u : \|u\|_1 \le r\}` is the QP
        :math:`\min_{u,t} \tfrac12\|u-\text{arg}\|^2` s.t. :math:`-t \le u \le t`,
        :math:`\mathbf{1}^\top t \le r`. With multipliers :math:`\alpha` (``u <= t``),
        :math:`\beta` (``-u <= t``), :math:`\mu` (``1^T t <= r``):

        .. math::
            u = \text{arg} - \alpha + \beta,\quad \alpha + \beta = \mu\mathbf{1},\\
            0 \le \alpha \perp (t - u) \ge 0,\ \
            0 \le \beta \perp (t + u) \ge 0,\ \
            0 \le \mu \perp (r - \mathbf{1}^\top t) \ge 0.

        The projection is routed through the generic QP core :meth:`_qp_kkt` on the
        lifted variable ``x = [u; t]`` (dim ``2n``) with ``P = diag(I_n, 0_n)``,
        ``q = [-arg; 0]`` and the stacked inequalities

        .. math::
            C_{ineq} = \begin{bmatrix} I & -I \\ -I & -I \\ 0 & \mathbf{1}^\top \end{bmatrix},
            \quad d_{ineq} = [0; 0; r].

        The single stacked dual ``lambda = [alpha; beta; mu]`` and the one complementarity
        ``lambda ⊥ (d_ineq - C_ineq x)`` reproduce the three bespoke complementarities
        exactly; the generic stationarity ``P x == -(q + C_ineq^T lambda)`` encodes both the
        ``u``- and ``t``-stationarity (the all-zero ``t``-rows of ``P`` give
        ``alpha + beta = mu 1``). The dual upper bound is auto-derived from the input box
        (``<= max_i|x_i|``), independent of ``r``; ``prox.dual_ub`` overrides it. ``P`` is
        singular, so explicit ``primal_bounds`` are supplied (``u in arg +/- dub``,
        ``t in [0, r]``). The returned projection is the slice ``x[:n]``.

        Parameters
        ----------
        prox : IndBallL1
            Carries the radius ``r`` (scalar or dim-1 LinearExpression) and ``dual_ub``.
        arg : LinearExpression
            The point being projected.
        gamma : float, default 1.0
            Step size (irrelevant for an indicator).

        Returns
        -------
        LinearExpression
            The primal projection ``u`` of dimension ``n`` (the slice ``x[:n]``).
        """
        n = arg.ndim

        r_le = self._lift_to_le(prox.r, 1)
        r_ub = float(r_le.bound_propagation()[1][0])  # max radius -> static upper bound on t

        # Auto dual bound from the input box (independent of r); override via prox.dual_ub.
        arg_lb, arg_ub = arg.bound_propagation()
        M = float(max(np.max(np.abs(arg_lb)), np.max(np.abs(arg_ub))))
        dub = M if prox.dual_ub is None else float(prox.dual_ub)

        # Lifted x = [u; t]: P = diag(I_n, 0_n), q = [-arg; 0].
        zeros_n = self._lift_to_le(0.0, n)
        eye = np.eye(n)
        C_ineq = np.block(
            [
                [eye, -eye],  # u - t <= 0
                [-eye, -eye],  # -u - t <= 0
                [np.zeros((1, n)), np.ones((1, n))],  # 1^T t <= r
            ]
        )
        x = self._qp_kkt(
            lin_term=stack([-arg, zeros_n]),
            P=spa.block_diag([spa.eye(n), spa.csr_matrix((n, n))], format="csr"),
            C_ineq=C_ineq,
            d_ineq=stack([zeros_n, zeros_n, r_le]),
            lam_ub=np.full(2 * n + 1, dub),
            primal_bounds=(
                np.concatenate([arg_lb - dub, np.zeros(n)]),
                np.concatenate([arg_ub + dub, np.full(n, r_ub)]),
            ),
        )
        return x[:n]

    def prox_indballlinf(
        self, prox: IndBallLinf, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower an $\ell_\infty$-ball indicator prox via a single
        ``SaturatedLinear`` step that clips to $[-r, r]$.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{\|\cdot\|_\infty \leq r}}(\text{arg})
        = \operatorname{clip}(\text{arg}, -r, r)$, independent of $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        l = np.full(arg.ndim, -prox.r)
        u = np.full(arg.ndim, prox.r)
        return self.process_step(SaturatedLinear(arg=arg, l=l, u=u))

    def prox_indbox(
        self, prox: IndBox, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a box-indicator prox via a single ``SaturatedLinear`` step
        that clips to $[l, u]$ element-wise.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{[l, u]}}(\text{arg})
        = \min(\max(\text{arg}, l), u)$, independent of $\gamma$. Scalar
        bounds are broadcast to ``arg.ndim``; arrays and
        ``LinearExpression`` parameters are forwarded through.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        l = self._to_satlin_bound(prox.l, arg.ndim)
        u = self._to_satlin_bound(prox.u, arg.ndim)
        return self.process_step(SaturatedLinear(arg=arg, l=l, u=u))

    def prox_indhalfspace(
        self, prox: IndHalfSpace, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a half-space indicator prox to a single scalar ``Relu``
        lifted back to the input dimension:

        $$
            \operatorname{prox}_{\gamma I_{\{a^\top x \le b\}}}(v)
            \;=\; v - \mathrm{ReLU}\!\left(\frac{a^\top v - b}{\|a\|^2}\right)\, a.
        $$

        The ``Relu`` step operates on the scalar signed-distance
        ``LinearExpression`` ``(a @ arg - b) / ||a||^2``; the lift back
        to vector dimension uses ``a.reshape(-1, 1) @ relu_excess`` since
        ``LinearExpression.__mul__`` only accepts scalars. Independent of
        $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        # LinearExpression.__rmatmul__ requires a 2-D matrix on the left, so
        # reshape a to (1, n) for the inner product and (n, 1) for the lift.
        excess = (prox.a.reshape(1, -1) @ arg - prox.b) / prox._norm_a_sq
        relu_excess = self.process_step(Relu(arg=excess))
        return arg - prox.a.reshape(-1, 1) @ relu_excess

    def prox_indnonnegative(
        self, prox: IndNonNegative, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a non-negative-orthant indicator prox into a single ``Relu`` step.

        Uses the identity $\operatorname{prox}_{\gamma I_{\geq 0}}(\text{arg})
        = \max(\text{arg}, 0)$, which is precisely ``Relu(arg)``. The
        result is independent of $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return self.process_step(Relu(arg=arg))

    def prox_indnonpositive(
        self, prox: IndNonPositive, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a non-positive-orthant indicator prox via a single ``Relu`` step.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{\leq 0}}(\text{arg}) =
        \min(\text{arg}, 0) = -\max(-\text{arg}, 0) = -\mathrm{Relu}(-\text{arg})$.
        The result is independent of $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return -self.process_step(Relu(arg=-arg))

    def prox_indpoint(
        self, prox: IndPoint, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a singleton-set indicator prox via a degenerate
        ``SaturatedLinear`` step with $l = u = p$.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{\{p\}}}(\text{arg}) = p$,
        independent of $\gamma$. The lowering encodes this as
        ``SaturatedLinear(arg, l=p, u=p)`` whose output bounds collapse
        to $[p, p]$. Scalar ``p`` is broadcast to ``arg.ndim``.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        p = self._to_satlin_bound(prox.p, arg.ndim)
        return self.process_step(SaturatedLinear(arg=arg, l=p, u=p))

    def prox_labeledhingeloss(
        self, prox: LabeledHingeLoss, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the labeled hinge prox
        $g_y(x) = \sum_i \max(0, 1 - y_i x_i)$ using

        $$
            \operatorname{prox}_{\gamma g_y}(v)
            = v + y \cdot \mathrm{satlin}(1 - y v;\, 0,\, \gamma).
        $$

        Element-wise scaling by $y$ is encoded as
        ``scipy.sparse.diags(y) @ arg`` — ``LinearExpression.__mul__``
        only accepts scalars; the sparse-diagonal matmul keeps the
        resulting LE's coefficients compact (no dense $n \times n$
        matrix is materialized).

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        Y = spa.diags(prox.y)
        residual = -(Y @ arg) + 1.0
        l = self._to_satlin_bound(0.0, arg.ndim)
        u = self._to_satlin_bound(gamma, arg.ndim)
        clip = self.process_step(SaturatedLinear(arg=residual, l=l, u=u))
        return arg + Y @ clip

    def prox_linear(
        self, prox: Linear, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a linear cost prox $f(x) = c^\top x$ to ``arg - gamma * c``.

        Pure ``LinearExpression`` arithmetic — no ``Step`` and no fresh
        leaf variables are introduced. ``prox.c`` may be a scalar, an
        ``np.ndarray``, or a ``LinearExpression`` parameter; arithmetic
        dispatch handles each transparently.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return arg - gamma * prox.c

    def prox_norml1(
        self, prox: NormL1, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a weighted $\ell_1$-norm prox into a single ``SoftThreshold`` step.

        Uses the identity
        $\operatorname{prox}_{\gamma \lambda \|\cdot\|_1}(\text{arg})
        = S_{\gamma \lambda}(\text{arg})$, where $S$ is element-wise soft
        thresholding.

        Backend-independent: registered on the base class so all step
        decomposers inherit it. Per-backend handling of the resulting
        ``SoftThreshold`` happens automatically — e.g. SDP further
        decomposes it into two ReLUs via ``_DECOMPOSERS``.
        """
        step = SoftThreshold(arg=arg, lambd=prox.lambd * gamma)
        return self.process_step(step)

    def prox_norminf(
        self, prox: NormInf, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the :math:`\ell_\infty`-norm prox via Moreau decomposition:

        $$
            \operatorname{prox}_{\gamma\lambda\|\cdot\|_\infty}(v)
            = v - \operatorname{proj}_{\{\|\cdot\|_1 \le \gamma\lambda\}}(v).
        $$

        Reuses :meth:`prox_indballl1` for the ℓ1-ball projection (radius ``γλ``), so the
        KKT + complementarity encoding, the auto-derived dual bound, and both the MIP and
        SDP backends are inherited. ``gamma * prox.lambd`` is a scalar when ``λ`` is a
        scalar and a dim-1 ``LinearExpression`` when ``λ`` is a parameter; ``IndBallL1``
        accepts both.

        Backend-independent: registered on the base class so all step decomposers inherit
        it (the Simulator overrides it to defer, since it calls ``prox_indballl1`` — an
        implicit-iterate prox — directly).
        """
        radius = gamma * prox.lambd
        proj = self.prox_indballl1(IndBallL1(radius, dual_ub=prox.dual_ub), arg)
        return arg - proj

    def prox_totalvariation1d(
        self, prox: TotalVariation1D, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the 1-D total-variation prox (fused lasso) via its KKT system.

        For :math:`\min_x \tfrac12\|x-\text{arg}\|^2 + \tau\|Dx\|_1` (``τ = γλ``, ``D`` the
        first-difference operator), with equality multiplier ``u`` and the split
        ``Dx = z⁺ - z⁻`` (``z⁺,z⁻ ≥ 0``):

        .. math::
            x = \text{arg} - D^\top u,\quad Dx = z^+ - z^-,\quad
            0 \le (\tau - u) \perp z^+ \ge 0,\quad 0 \le (\tau + u) \perp z^- \ge 0.

        The complementarities' nonnegativity sides encode the dual box ``-τ ≤ u ≤ τ``, so all
        bounds are derived from the data (no user ``dual_ub``): ``u ∈ [-τ, τ]`` and ``z⁺/z⁻``
        bounded by the range of ``Dx``. ``λ`` may be a scalar or a dim-1 LinearExpression.

        Parameters
        ----------
        prox : TotalVariation1D
            Carries the weight ``lambd`` (scalar or dim-1 LinearExpression).
        arg : LinearExpression
            The input signal being smoothed.
        gamma : float, default 1.0
            Step size.

        Returns
        -------
        LinearExpression
            The proximal point ``x = arg - Dᵀu`` of dimension ``n``.
        """
        n = arg.ndim
        if n < 2:
            warnings.warn(_SHORT_SIGNAL_MSG.format(n=n), UserWarning, stacklevel=2)
            return arg  # TV of a scalar is 0 -> identity
        m = n - 1
        D = diff_matrix(n)

        # τ = γλ as a dim-m LinearExpression (uniform for scalar & parameter λ).
        lambd_le = self._lift_to_le(prox.lambd, 1)
        tau = gamma * (np.ones((m, 1)) @ lambd_le)
        tau_max = float((gamma * lambd_le).bound_propagation()[1][0])

        u = self.add_iterate(m, lb=np.full(m, -tau_max), ub=np.full(m, tau_max))

        x = arg - D.T @ u  # primal (expression, returned)
        z_lb, z_ub = (D @ x).bound_propagation()
        zp = self.add_iterate(m, lb=np.zeros(m), ub=np.maximum(0.0, z_ub))
        zm = self.add_iterate(m, lb=np.zeros(m), ub=np.maximum(0.0, -z_lb))

        self.add_constraint(D @ x == zp - zm)  # z = Dx split
        self.add_constraint(complementarity(tau - u, zp))  # u <= τ and (τ-u) ⊥ z+
        self.add_constraint(complementarity(tau + u, zm))  # u >= -τ and (τ+u) ⊥ z-
        return x

    def prox_quadratic(
        self, prox: Quadratic, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a convex-quadratic prox $f(x) = \tfrac{1}{2}x^\top Q x + q^\top x$
        to a single linear equality on a fresh iterate ``u``:

        $$
            (I + \gamma Q)\, u \;=\; \text{arg} - \gamma q.
        $$

        This is the unconstrained case of the generic QP, so it routes through
        :meth:`_qp_kkt` with ``P = I + gamma*Q`` and ``q_qp = gamma*q - arg`` (the
        prox objective expands to :math:`\tfrac12 x^\top(I+\gamma Q)x + (\gamma q -
        \text{arg})^\top x`). With no equality/inequality blocks, ``_qp_kkt`` creates
        only the primal iterate and posts the single stationarity equality
        ``(I + gamma*Q) x == arg - gamma*q`` — no duals, no complementarity. Bounds
        on ``x`` are derived in ``_qp_kkt`` via ``solve_linexpr(M, arg - gamma*q)``
        (per-leaf ``np.linalg.solve`` — no inverse materialized) and tightened by OBBT
        when enabled. ``q`` may be numeric data or a ``LinearExpression`` parameter.

        Backend-independent: registered on the base class. Simulator
        overrides this with a deferral (no per-step activation callback
        is available for an implicitly-defined iterate).
        """
        n = arg.ndim
        if spa.issparse(prox.Q):
            # Keep (I + gamma*Q) sparse — _qp_kkt's solve_linexpr/P@x are sparse-aware.
            # A dense ``np.eye(n) + gamma*Q`` would silently yield a numpy.matrix.
            M = spa.eye(n, format="csr") + gamma * prox.Q
        else:
            M = np.eye(n) + gamma * prox.Q
        # lin_term = q_qp = gamma*q - arg (negation of the known-good arg - gamma*q,
        # which keeps the arithmetic on LinearExpression operands).
        return self._qp_kkt(lin_term=-(arg - gamma * prox.q), P=M)

    def prox_softnonnegative(
        self, prox: SoftNonNegative, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the one-sided quadratic penalty prox to a scaled LeakyReLU,
        expressed via the standard ``Relu`` step:

        $$
            \operatorname{prox}_{\gamma f}(v)
                = \tilde\alpha(\gamma)\, v
                  + \bigl(1 - \tilde\alpha(\gamma)\bigr)\operatorname{ReLU}(v),
            \qquad
            \tilde\alpha(\gamma) = \frac{\alpha}{\alpha + \gamma(1 - \alpha)}.
        $$

        At $\gamma = 1$ this is the standard LeakyReLU with slope
        $\alpha$; at $\alpha \to 0^+$ it collapses to plain ReLU.

        NOTE: A dedicated ``LeakyReLU`` step (analogous to ``Relu``) would
        let the canonicalizers encode the activation directly without the
        intermediate "scale + ReLU + scale + add" arithmetic. That can be
        a future addition. For now we reuse the existing ``Relu`` step,
        which is exact and works on all three backends.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        alpha = prox.alpha
        alpha_tilde = alpha / (alpha + gamma * (1.0 - alpha))
        relu_out = self.process_step(Relu(arg=arg))
        return alpha_tilde * arg + (1.0 - alpha_tilde) * relu_out

    def prox_sql2norm(
        self, prox: SqL2Norm, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the squared $\ell_2$-norm prox $f(x) = \tfrac{1}{2}\|x\|_2^2$
        to ``arg / (1 + gamma)``.

        Pure scalar shrinkage on the ``LinearExpression`` — no ``Step``
        and no fresh leaf variables are introduced.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return arg / (1.0 + gamma)

    def prox_sumnegative(
        self, prox: SumNegative, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the sum-of-negative-parts prox $f(x) = \sum_i \max(0, -x_i)$
        to a single ``SaturatedLinear`` clip to $[-\gamma, 0]$ followed
        by a subtraction:

        $$
            \operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v;\, -\gamma,\, 0).
        $$

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        l = self._to_satlin_bound(-gamma, arg.ndim)
        u = self._to_satlin_bound(0.0, arg.ndim)
        clip = self.process_step(SaturatedLinear(arg=arg, l=l, u=u))
        return arg - clip

    def prox_sumpositive(
        self, prox: SumPositive, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the sum-of-positive-parts prox $f(x) = \sum_i \max(0, x_i)$
        to a single ``SaturatedLinear`` clip to $[0, \gamma]$ followed
        by a subtraction:

        $$
            \operatorname{prox}_{\gamma f}(v) = v - \mathrm{satlin}(v;\, 0,\, \gamma).
        $$

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        l = self._to_satlin_bound(0.0, arg.ndim)
        u = self._to_satlin_bound(gamma, arg.ndim)
        clip = self.process_step(SaturatedLinear(arg=arg, l=l, u=u))
        return arg - clip

    _PROX_DECOMPOSERS: ClassVar[dict] = {
        "elasticnet": prox_elasticnet,
        "hingeloss": prox_hingeloss,
        "huberloss": prox_huberloss,
        "indaffine": prox_indaffine,
        "indballl1": prox_indballl1,
        "indballlinf": prox_indballlinf,
        "indbox": prox_indbox,
        "indhalfspace": prox_indhalfspace,
        "indnonnegative": prox_indnonnegative,
        "indnonpositive": prox_indnonpositive,
        "indpoint": prox_indpoint,
        "indpolyhedral": prox_indpolyhedral,
        "indsimplex": prox_indsimplex,
        "labeledhingeloss": prox_labeledhingeloss,
        "leastsquares": prox_quadratic,
        "linear": prox_linear,
        "norminf": prox_norminf,
        "norml1": prox_norml1,
        "quadratic": prox_quadratic,
        "softnonnegative": prox_softnonnegative,
        "sql2norm": prox_sql2norm,
        "sumnegative": prox_sumnegative,
        "sumpositive": prox_sumpositive,
        "totalvariation1d": prox_totalvariation1d,
    }
