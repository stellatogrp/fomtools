from typing import Callable, ClassVar

import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.steps import Relu, SaturatedLinear
from fomtools.steps.step import Step


class StepDecomposer:
    """
    Base class for step decomposition.

    Decomposes composite steps into primitive steps that the canonicalizer
    can handle directly. Returns LinearExpression (leaf or composite).

    Subclasses declare a class-level ``_DECOMPOSERS`` dict mapping the
    lowercase step class name (e.g. ``"softthreshold"``) to the function
    that decomposes it. Steps whose name is absent from the registry are
    treated as primitives and processed directly via the canonicalizer.

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

        For primitive steps: registers lhs, canonicalizes, returns step.lhs
        For composite steps: decomposes into primitives, returns composite expr

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

        Registers the step's output variable with the canonicalizer and
        OBBT handler, encodes the step constraints, and returns the
        output variable.

        Parameters
        ----------
        step : Step
            A primitive step to process directly.

        Returns
        -------
        LinearExpression
            The step's left-hand side (output) variable.
        """
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
        return step.lhs
