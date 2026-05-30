import numpy as np

from fomtools.linearexpression import LinearExpression
from fomtools.stepdecomposers.step_decomposer import StepDecomposer
from fomtools.steps import Relu, SaturatedLinear, SoftThreshold


class SDPStepDecomposer(StepDecomposer):
    """
    SDP step decomposer.

    Decomposes composite steps into ReLU primitives that the SDP
    canonicalizer can encode directly. The SDP canonicalizer only
    natively supports ReLU; other steps must be decomposed.

    Decompositions:
    - SoftThreshold: softthresh_lambda(x) = relu(x - lambda) - relu(-x - lambda)
    - SaturatedLinear: satlin(x, l, u) = relu(x - l) - relu(x - u) + l
    """

    def decompose_softthreshold(self, step: SoftThreshold) -> LinearExpression:
        """
        Decompose soft-thresholding into ReLU operations.

        Uses the identity:
            softthresh_lambda(x) = relu(x - lambda) - relu(-x - lambda)

        Parameters
        ----------
        step : SoftThreshold
            The soft-thresholding step to decompose.

        Returns
        -------
        LinearExpression
            A composite expression representing the soft-thresholding
            output as the difference of two ReLU outputs.
        """
        x_lb, x_ub = step.lb, step.ub
        lambd = step.lambd
        lambd_vec = np.full(step.rhs.ndim, lambd)

        # Create ReLU steps for decomposition:
        # relu_pos computes relu(x - lambda)
        # relu_neg computes relu(-x - lambda)
        relu_pos = Relu(rhs=step.rhs - lambd_vec)
        relu_neg = Relu(rhs=-step.rhs - lambd_vec)

        # Update bounds on ReLU inputs, then propagate to outputs
        relu_pos.update_bounds(x_lb - lambd, x_ub - lambd)
        relu_neg.update_bounds(-x_ub - lambd, -x_lb - lambd)

        # Recursively process (handles nested decompositions)
        lhs_pos = self.process_step(relu_pos)
        lhs_neg = self.process_step(relu_neg)

        return lhs_pos - lhs_neg

    def decompose_saturatedlinear(self, step: SaturatedLinear) -> LinearExpression:
        """
        Decompose saturated linear into ReLU operations.

        Delegates to the shared helper in the base class, which uses the identity:
            satlin(x, l, u) = relu(x - l) - relu(x - u) + l

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
        return self._decompose_saturatedlinear_to_relus(step)

    _DECOMPOSERS = {
        "saturatedlinear": decompose_saturatedlinear,
        "softthreshold": decompose_softthreshold,
    }
