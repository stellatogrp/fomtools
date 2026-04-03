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

        The saturated linear function clip(x, l, u) can be expressed as:
            satlin(x, l, u) = relu(x - l) - relu(x - u) + l

        This follows from:
            max(x, l) = relu(x - l) + l
            satlin(x, l, u) = min(max(x, l), u)

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

        # Get saturation bounds (may be arrays or LinearExpressions)
        # For LinearExpression bounds, use their lb/ub attributes
        try:
            l_vec = step.l.lb
            u_vec = step.u.ub
        except AttributeError:
            l_vec = np.atleast_1d(step.l)
            u_vec = np.atleast_1d(step.u)

        # Broadcast to match input dimension if needed
        if np.isscalar(l_vec) or len(l_vec) == 1:
            l_vec = np.full(step.rhs.ndim, l_vec.item() if hasattr(l_vec, "item") else l_vec)
        if np.isscalar(u_vec) or len(u_vec) == 1:
            u_vec = np.full(step.rhs.ndim, u_vec.item() if hasattr(u_vec, "item") else u_vec)

        # Create ReLU steps for decomposition:
        # relu_lower computes relu(x - l)
        # relu_upper computes relu(x - u)
        relu_lower = Relu(rhs=step.rhs - l_vec)
        relu_upper = Relu(rhs=step.rhs - u_vec)

        # Update bounds on ReLU inputs, then propagate to outputs
        relu_lower.update_bounds(x_lb - l_vec, x_ub - l_vec)
        relu_upper.update_bounds(x_lb - u_vec, x_ub - u_vec)

        # Recursively process
        lhs_lower = self.process_step(relu_lower)
        lhs_upper = self.process_step(relu_upper)

        # satlin(x) = relu(x - l) - relu(x - u) + l
        return lhs_lower - lhs_upper + l_vec
