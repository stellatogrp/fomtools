from fomtools.linearexpression import LinearExpression
from fomtools.stepdecomposers.step_decomposer import StepDecomposer
from fomtools.steps import SaturatedLinear


class MIPStepDecomposer(StepDecomposer):
    """
    MIP step decomposer.

    Passes most steps through as primitives since the MIP canonicalizer
    handles them directly via convex hull formulations.

    The MIP canonicalizer natively supports:
    - ReLU via big-M constraints
    - SoftThreshold via convex hull formulation
    - SaturatedLinear via convex hull formulation (constant bounds only)

    SaturatedLinear steps with LinearExpression (parameter) bounds are
    decomposed into ReLU primitives using the identity:
        satlin(x, l, u) = relu(x - l) - relu(x - u) + l
    """

    def decompose_saturatedlinear(self, step: SaturatedLinear) -> LinearExpression:
        """
        Decompose saturated linear, or fall through to the MIP canonicalizer.

        If both l and u are constant, delegates to the MIP canonicalizer's
        native convex hull formulation. If either is a LinearExpression
        (parameter), decomposes into ReLU primitives.

        Parameters
        ----------
        step : SaturatedLinear
            The saturated linear step to decompose.

        Returns
        -------
        LinearExpression
            The output expression.
        """
        has_param = hasattr(step.l, "lb") or hasattr(step.u, "lb")

        if not has_param:
            return self._process_primitive(step)

        return self._decompose_saturatedlinear_to_relus(step)

    _DECOMPOSERS = {
        "saturatedlinear": decompose_saturatedlinear,
        "quadraticprogram": StepDecomposer.decompose_quadraticprogram,
        "linearprogram": StepDecomposer.decompose_quadraticprogram,
    }
