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

    Every cut acts on one ReLU at a time, so the difference of the two pieces
    loses the convex hull of the composite map ``v -> z`` over the input bounds
    ``[v_l, v_u]``. Each decomposition therefore also registers that hull as
    linear inequalities (see `_add_hull_rows`); the canonicalizer's
    inequality cuts then multiply them with the bounds of every variable they
    involve (the parameters and earlier iterates included):

    - SoftThreshold: ``-lambda <= v - z <= lambda`` and the chords through
      ``(-lambda, 0)``, ``(v_u, soft(v_u))`` and ``(v_l, soft(v_l))``, ``(lambda, 0)``.
    - SaturatedLinear: the chords through ``(v_l, clip(v_l))``, ``(u, u)`` and
      ``(l, l)``, ``(v_u, clip(v_u))``. The box lines ``l <= z <= u`` are left out:
      the box normal-cone cuts already enforce them.
    """

    def _add_hull_rows(
        self, A_z: np.ndarray, z: LinearExpression, A_v: np.ndarray, v: LinearExpression, ub
    ) -> None:
        """
        Register the rows ``A_z z + A_v v <= ub`` as one inequality constraint.

        Rows with a non-finite coefficient or bound (an undefined chord) are
        dropped. The constraint goes to the canonicalizer's inequality impl, so it
        gets the ``ineq_*_cuts`` strengthenings. The product cuts between its rows
        (which pair coordinates) give most of the tightening, so they stay on.
        Adding the rows to the OBBT LP was measured to change nothing, so they are
        not. Guarded so non-SDP canonicalizers (if ever paired) are left alone.
        """
        impl = getattr(self.canonicalizer, "_add_inequality_constraint_impl", None)
        if impl is None:
            return
        ub = np.asarray(ub, dtype=float)
        keep = np.isfinite(ub) & np.isfinite(A_z).all(axis=1) & np.isfinite(A_v).all(axis=1)
        if not keep.any():
            return
        impl(A_z[keep] @ z + A_v[keep] @ v <= ub[keep])

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
            register(z=z, v=step.arg, tau=lambd, func="l1")

        # Hull of v -> soft(v) on [v_l, v_u], read from the OBBT-tightened ReLU inputs:
        # relu_pos has input v - lambda, relu_neg has input -v - lambda.
        n = step.arg.ndim
        v_l = np.maximum(relu_pos.lb + lambd, -relu_neg.ub - lambd)
        v_u = np.minimum(relu_pos.ub + lambd, -relu_neg.lb - lambd)
        eye, lam = np.eye(n), np.full(n, lambd)
        # Chords: z <= s_u (v + lambda) and z >= s_l (v - lambda), with slopes in [0, 1].
        with np.errstate(divide="ignore", invalid="ignore"):
            s_u = np.where(v_u > -lambd, np.maximum(v_u - lambd, 0) / (v_u + lambd), np.nan)
            s_l = np.where(v_l < lambd, np.maximum(-v_l - lambd, 0) / (lambd - v_l), np.nan)
        # Lines (the residual v - z = clip(v, -lambda, lambda)) and chords, as one
        # constraint so the product cuts also pair rows across the two families.
        self._add_hull_rows(
            np.vstack([-eye, eye, eye, -eye]),
            z,
            np.vstack([eye, -eye, -np.diag(s_u), np.diag(s_l)]),
            step.arg,
            np.concatenate([lam, lam, s_u * lambd, s_l * lambd]),
        )

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
        z, relu_lower, relu_upper = self._decompose_saturatedlinear_to_relus(
            step, return_relus=True
        )

        # Register the composite box-projection prox for the PEP box cuts (normal
        # cone under cross_cut, monotonicity under subgrad_mono_cut).
        #
        # Unlike soft-threshold, the internal ReLUs are NOT tagged
        # ``_skip_prox_registration``: they stay registered as tau=1 orthant proxes.
        # That is correct here because satlin = relu(v - l) - relu(v - u) + l, and
        # relu(v - l), relu(v - u) genuinely ARE orthant projections (tau = 1) of
        # their shifted inputs — so the subgradient cut is valid for them as-is. By
        # contrast soft-threshold's internal relu(v - lambda)/relu(-v - lambda) are
        # NOT the ell-1 prox (whose subgradient needs the tau = lambda
        # normalization), so those are skipped and only the composite is registered.
        # Here the box-level cuts are simply added on top of the valid orthant ones.
        # Guarded for non-SDP decomposers.
        register = getattr(self.canonicalizer, "register_box_prox_step", None)
        if register is not None:
            register(z=z, v=step.arg, l=step.l, u=step.u)

        # Hull chords of v -> clip(v, l, u) on [v_l, v_u]; they need numeric l and u.
        if not isinstance(step.l, LinearExpression) and not isinstance(step.u, LinearExpression):
            n = step.arg.ndim
            lo = np.broadcast_to(np.asarray(step.l, dtype=float), n)
            hi = np.broadcast_to(np.asarray(step.u, dtype=float), n)
            # relu_lower has input v - l, relu_upper has input v - u.
            v_l = np.maximum(relu_lower.lb + lo, relu_upper.lb + hi)
            v_u = np.minimum(relu_lower.ub + lo, relu_upper.ub + hi)
            c_l, c_u = np.clip(v_l, lo, hi), np.clip(v_u, lo, hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                # z <= c_l + s_hi (v - v_l), through (v_l, clip(v_l)) and (u, u).
                s_hi = np.where(v_l < hi, (hi - c_l) / (hi - v_l), np.nan)
                # z >= l + s_lo (v - l), through (l, l) and (v_u, clip(v_u)).
                s_lo = np.where(v_u > lo, (c_u - lo) / (v_u - lo), np.nan)
            eye = np.eye(n)
            self._add_hull_rows(
                np.vstack([eye, -eye]),
                z,
                np.vstack([-np.diag(s_hi), np.diag(s_lo)]),
                step.arg,
                np.concatenate([c_l - s_hi * v_l, s_lo * lo - lo]),
            )

        return z

    _DECOMPOSERS = {
        "saturatedlinear": decompose_saturatedlinear,
        "softthreshold": decompose_softthreshold,
        "quadraticprogram": StepDecomposer.decompose_quadraticprogram,
        "linearprogram": StepDecomposer.decompose_quadraticprogram,
    }
