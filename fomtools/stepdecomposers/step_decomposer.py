import logging
from typing import Callable, ClassVar

import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import (
    ElasticNet,
    HuberLoss,
    IndBallLinf,
    IndBox,
    IndHalfSpace,
    IndNonNegative,
    IndNonPositive,
    IndPoint,
    Linear,
    NormL1,
    SqL2Norm,
    SumNegative,
    SumPositive,
)
from fomtools.steps import Relu, SaturatedLinear, SoftThreshold
from fomtools.steps.step import Step

log = logging.getLogger(__name__)


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
    register_var : Callable[[LinearExpression], None]
        Callback function to register new leaf variables with both the
        canonicalizer and OBBT handler.

    Attributes
    ----------
    canonicalizer : Canonicalizer
        The canonicalizer instance used for constraint encoding.
    register_var : Callable[[LinearExpression], None]
        Callback for registering new leaf variables.
    """

    _DECOMPOSERS: ClassVar[dict] = {}

    def __init__(self, canonicalizer, register_var: Callable[[LinearExpression], None]):
        self.canonicalizer = canonicalizer
        self.register_var = register_var

    def is_primitive(self, step: Step) -> bool:
        """
        Return True if ``step`` has no registered decomposition for this
        decomposer and should be processed directly as a primitive.
        """
        return type(step).__name__.lower() not in self._DECOMPOSERS

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
            The output expression. For primitive steps, this is step.lhs.
            For composite steps, this is a LinearExpression composed from
            the outputs of the decomposed primitive steps.
        """
        decomposer = self._DECOMPOSERS.get(type(step).__name__.lower())
        if decomposer is None:
            return self._process_primitive(step)
        return decomposer(self, step)

    def process_prox(
        self,
        prox_function,
        rhs: LinearExpression,
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
        rhs : LinearExpression
            The input expression to which the prox is applied.
        gamma : float, default 1.0
            The proximal step size.

        Returns
        -------
        LinearExpression
            The output expression ``prox_{gamma * f}(rhs)``.

        Raises
        ------
        NotImplementedError
            If this decomposer's ``_PROX_DECOMPOSERS`` registry has no
            entry for the concrete type of ``prox_function``.
        """
        proxer = self._PROX_DECOMPOSERS.get(type(prox_function).__name__.lower())
        if proxer is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support "
                f"{type(prox_function).__name__}"
            )
        return proxer(self, prox_function, rhs, gamma)

    def _decompose_saturatedlinear_to_relus(
        self, step: SaturatedLinear
    ) -> LinearExpression:
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
        relu_lower = Relu(rhs=step.rhs - step.l)
        relu_upper = Relu(rhs=step.rhs - step.u)

        # Bounds on (rhs - l): [x_lb - l_ub, x_ub - l_lb]
        relu_lower.update_bounds(x_lb - l_ub, x_ub - l_lb)
        # Bounds on (rhs - u): [x_lb - u_ub, x_ub - u_lb]
        relu_upper.update_bounds(x_lb - u_ub, x_ub - u_lb)

        lhs_lower = self.process_step(relu_lower)
        lhs_upper = self.process_step(relu_upper)

        # satlin(x, l, u) = relu(x - l) - relu(x - u) + l
        return lhs_lower - lhs_upper + step.l

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
            result = obbt.obbt(step.rhs, step.lb, step.ub)
            if result is not None:
                step.update_bounds(*result)

        if hasattr(step, "sync_lhs_bounds"):
            step.sync_lhs_bounds()
        # registered as "param" for MIP/SDP compatibility; Simulator registers
        # step lhs as an iterate with zero init so subsequent steps can reference it.
        # For steps whose lhs is a composite view (e.g. partial-projection Relu),
        # lhs_leaves names only the fresh leaves that actually need solver-side
        # variables; non-fresh rows reuse rhs leaves and are already registered.
        for leaf in step.lhs_leaves:
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

        return step.lhs

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
        self, prox: ElasticNet, rhs: LinearExpression, gamma: float = 1.0
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
        soft = self.process_step(
            SoftThreshold(rhs=rhs, lambd=prox.alpha * gamma)
        )
        return soft / (1.0 + 2.0 * prox.beta * gamma)

    def prox_huberloss(
        self, prox: HuberLoss, rhs: LinearExpression, gamma: float = 1.0
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
        soft = self.process_step(SoftThreshold(rhs=rhs, lambd=theta))
        return (rhs + gamma * soft) / (1.0 + gamma)

    def prox_indballlinf(
        self, prox: IndBallLinf, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower an $\ell_\infty$-ball indicator prox via a single
        ``SaturatedLinear`` step that clips to $[-r, r]$.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{\|\cdot\|_\infty \leq r}}(\text{rhs})
        = \operatorname{clip}(\text{rhs}, -r, r)$, independent of $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        l = np.full(rhs.ndim, -prox.r)
        u = np.full(rhs.ndim, prox.r)
        return self.process_step(SaturatedLinear(rhs=rhs, l=l, u=u))

    def prox_indbox(
        self, prox: IndBox, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a box-indicator prox via a single ``SaturatedLinear`` step
        that clips to $[l, u]$ element-wise.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{[l, u]}}(\text{rhs})
        = \min(\max(\text{rhs}, l), u)$, independent of $\gamma$. Scalar
        bounds are broadcast to ``rhs.ndim``; arrays and
        ``LinearExpression`` parameters are forwarded through.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        l = self._to_satlin_bound(prox.l, rhs.ndim)
        u = self._to_satlin_bound(prox.u, rhs.ndim)
        return self.process_step(SaturatedLinear(rhs=rhs, l=l, u=u))

    def prox_indhalfspace(
        self, prox: IndHalfSpace, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a half-space indicator prox to a single scalar ``Relu``
        lifted back to the input dimension:

        $$
            \operatorname{prox}_{\gamma I_{\{a^\top x \le b\}}}(v)
            \;=\; v - \mathrm{ReLU}\!\left(\frac{a^\top v - b}{\|a\|^2}\right)\, a.
        $$

        The ``Relu`` step operates on the scalar signed-distance
        ``LinearExpression`` ``(a @ rhs - b) / ||a||^2``; the lift back
        to vector dimension uses ``a.reshape(-1, 1) @ relu_excess`` since
        ``LinearExpression.__mul__`` only accepts scalars. Independent of
        $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        # LinearExpression.__rmatmul__ requires a 2-D matrix on the left, so
        # reshape a to (1, n) for the inner product and (n, 1) for the lift.
        excess = (prox.a.reshape(1, -1) @ rhs - prox.b) / prox._norm_a_sq
        relu_excess = self.process_step(Relu(rhs=excess))
        return rhs - prox.a.reshape(-1, 1) @ relu_excess

    def prox_indnonnegative(
        self, prox: IndNonNegative, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a non-negative-orthant indicator prox into a single ``Relu`` step.

        Uses the identity $\operatorname{prox}_{\gamma I_{\geq 0}}(\text{rhs})
        = \max(\text{rhs}, 0)$, which is precisely ``Relu(rhs)``. The
        result is independent of $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return self.process_step(Relu(rhs=rhs))

    def prox_indnonpositive(
        self, prox: IndNonPositive, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a non-positive-orthant indicator prox via a single ``Relu`` step.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{\leq 0}}(\text{rhs}) =
        \min(\text{rhs}, 0) = -\max(-\text{rhs}, 0) = -\mathrm{Relu}(-\text{rhs})$.
        The result is independent of $\gamma$.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return -self.process_step(Relu(rhs=-rhs))

    def prox_indpoint(
        self, prox: IndPoint, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a singleton-set indicator prox via a degenerate
        ``SaturatedLinear`` step with $l = u = p$.

        Uses the identity
        $\operatorname{prox}_{\gamma I_{\{p\}}}(\text{rhs}) = p$,
        independent of $\gamma$. The lowering encodes this as
        ``SaturatedLinear(rhs, l=p, u=p)`` whose output bounds collapse
        to $[p, p]$. Scalar ``p`` is broadcast to ``rhs.ndim``.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        p = self._to_satlin_bound(prox.p, rhs.ndim)
        return self.process_step(SaturatedLinear(rhs=rhs, l=p, u=p))

    def prox_linear(
        self, prox: Linear, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a linear cost prox $f(x) = c^\top x$ to ``rhs - gamma * c``.

        Pure ``LinearExpression`` arithmetic — no ``Step`` and no fresh
        leaf variables are introduced. ``prox.c`` may be a scalar, an
        ``np.ndarray``, or a ``LinearExpression`` parameter; arithmetic
        dispatch handles each transparently.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return rhs - gamma * prox.c

    def prox_norml1(
        self, prox: NormL1, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower a weighted $\ell_1$-norm prox into a single ``SoftThreshold`` step.

        Uses the identity
        $\operatorname{prox}_{\gamma \lambda \|\cdot\|_1}(\text{rhs})
        = S_{\gamma \lambda}(\text{rhs})$, where $S$ is element-wise soft
        thresholding.

        Backend-independent: registered on the base class so all step
        decomposers inherit it. Per-backend handling of the resulting
        ``SoftThreshold`` happens automatically — e.g. SDP further
        decomposes it into two ReLUs via ``_DECOMPOSERS``.
        """
        step = SoftThreshold(rhs=rhs, lambd=prox.lambd * gamma)
        return self.process_step(step)

    def prox_sql2norm(
        self, prox: SqL2Norm, rhs: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        r"""
        Lower the squared $\ell_2$-norm prox $f(x) = \tfrac{1}{2}\|x\|_2^2$
        to ``rhs / (1 + gamma)``.

        Pure scalar shrinkage on the ``LinearExpression`` — no ``Step``
        and no fresh leaf variables are introduced.

        Backend-independent: registered on the base class so all step
        decomposers inherit it.
        """
        return rhs / (1.0 + gamma)

    def prox_sumnegative(
        self, prox: SumNegative, rhs: LinearExpression, gamma: float = 1.0
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
        l = self._to_satlin_bound(-gamma, rhs.ndim)
        u = self._to_satlin_bound(0.0, rhs.ndim)
        clip = self.process_step(SaturatedLinear(rhs=rhs, l=l, u=u))
        return rhs - clip

    def prox_sumpositive(
        self, prox: SumPositive, rhs: LinearExpression, gamma: float = 1.0
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
        l = self._to_satlin_bound(0.0, rhs.ndim)
        u = self._to_satlin_bound(gamma, rhs.ndim)
        clip = self.process_step(SaturatedLinear(rhs=rhs, l=l, u=u))
        return rhs - clip

    _PROX_DECOMPOSERS: ClassVar[dict] = {
        "elasticnet": prox_elasticnet,
        "huberloss": prox_huberloss,
        "indballlinf": prox_indballlinf,
        "indbox": prox_indbox,
        "indhalfspace": prox_indhalfspace,
        "indnonnegative": prox_indnonnegative,
        "indnonpositive": prox_indnonpositive,
        "indpoint": prox_indpoint,
        "linear": prox_linear,
        "norml1": prox_norml1,
        "sql2norm": prox_sql2norm,
        "sumnegative": prox_sumnegative,
        "sumpositive": prox_sumpositive,
    }
