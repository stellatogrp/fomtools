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
        lambd_vec = np.full(step.arg.ndim, lambd)

        # Create ReLU steps for decomposition:
        # relu_pos computes relu(x - lambda)
        # relu_neg computes relu(-x - lambda)
        relu_pos = Relu(arg=step.arg - lambd_vec)
        relu_neg = Relu(arg=-step.arg - lambd_vec)

        # These ReLUs are internal pieces of the soft-threshold prox, not proxes
        # in their own right. Tag them so the SDP canonicalizer does not register
        # them for the subgradient-monotonicity cut; the composite soft-threshold
        # prox is registered below instead. (They still participate in the cross
        # cut, which operates at the ReLU-node level.)
        relu_pos._skip_prox_registration = True
        relu_neg._skip_prox_registration = True

        # Update bounds on ReLU inputs, then propagate to outputs
        relu_pos.update_bounds(x_lb - lambd, x_ub - lambd)
        relu_neg.update_bounds(-x_ub - lambd, -x_lb - lambd)

        # Recursively process (handles nested decompositions)
        output_pos = self.process_step(relu_pos)
        output_neg = self.process_step(relu_neg)

        z = output_pos - output_neg

        # Register the composite soft-threshold prox (ell-1, tau=lambda) for the
        # subgradient-monotonicity cut. Both sub-ReLUs are now canonicalized and
        # their leaves are in var_idx_map, so z is well-formed. Guard on the
        # method's presence to keep non-SDP decomposers (if ever paired) safe.
        register = getattr(self.canonicalizer, "register_prox_step", None)
        if register is not None:
            register(z=z, v=step.arg, tau=lambd)

        return z

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
        "quadraticprogram": StepDecomposer.decompose_quadraticprogram,
        "linearprogram": StepDecomposer.decompose_quadraticprogram,
    }
