from typing import Callable

from fomtools.linearexpression import LinearExpression
from fomtools.steps.step import Step


class StepDecomposer:
    """
    Base class for step decomposition.

    Decomposes composite steps into primitive steps that the canonicalizer
    can handle directly. Returns LinearExpression (leaf or composite).

    Subclasses can implement ``decompose_{stepname}`` methods to handle
    specific step types. Steps without a corresponding decompose method
    are treated as primitives and processed directly via the canonicalizer.

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

    def __init__(self, canonicalizer, register_var: Callable[[LinearExpression], None]):
        self.canonicalizer = canonicalizer
        self.register_var = register_var

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
        # Convert class name to method name: SoftThreshold -> decompose_softthreshold
        step_name = type(step).__name__.lower()
        method_name = f"decompose_{step_name}"

        try:
            decomposer = getattr(self, method_name)
        except AttributeError:
            # No decomposition needed - treat as primitive
            return self._process_primitive(step)

        return decomposer(step)

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
        # registered as "param" for MIP/SDP compatibility; Simulator silently drops
        # this since lhs has no samples and is not referenced in step.rhs
        self.register_var(step.lhs, var_type="param")

        self.canonicalizer.canonicalize(step)
        return step.lhs
