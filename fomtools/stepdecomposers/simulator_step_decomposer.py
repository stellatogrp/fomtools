from typing import ClassVar

import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.simulator_subproblems import (
    l1_ball_activation,
    norm_inf_activation,
    polyhedral_activation,
    qp_activation,
    simplex_activation,
    tv_activation,
)
from fomtools.linearexpression import LinearExpression, stack
from fomtools.stepdecomposers.step_decomposer import StepDecomposer
from fomtools.steps import Affine
from fomtools.steps.step import Step


def _assert_nonsingular(M: np.ndarray, context: str) -> None:
    """
    Raise a clear error if the dense matrix ``M`` is (numerically) singular.

    The Simulator solves the implicit-iterate linear system once per constant
    matrix and reuses the factorization across all samples; a singular matrix
    would otherwise surface as an opaque ``LinAlgError`` deep inside
    ``solve_model``. Only dense matrices are checked here — sparse systems are
    left to ``splu``, which raises on a singular factorization.
    """
    sign, _ = np.linalg.slogdet(np.asarray(M, dtype=float))
    if sign == 0:
        raise ValueError(
            f"{context}: the linear system matrix is singular and cannot be solved "
            "on the Simulator backend. Ensure the problem is well posed (e.g. the "
            "equality matrix has full row rank)."
        )


class SimulatorStepDecomposer(StepDecomposer):
    """
    Simulator Step Decomposer.

    Lowers each prox/step to a numeric evaluator recorded with the Simulator
    canonicalizer. Three families:

    - **Primitive** activations (``Relu``/``SaturatedLinear``/``SoftThreshold``):
      forwarded directly via :meth:`_process_primitive`.
    - **Pure linear solve** (``Quadratic``/``LeastSquares``, ``IndAffine``, and the
      equality-only / unconstrained ``QuadraticProgram``): lowered to an
      :class:`Affine` step (one batched multi-RHS solve, no Python loop).
    - **Subproblem** activations for the complementarity cases. The simplex,
      ℓ1-ball and ℓ∞-norm proxes use batched solver-free projections; the 1-D
      total-variation prox uses Condat's O(n) algorithm; polyhedral projection and
      inequality QP/LP loop a parametrized cvxpy solve per sample. All are wired
      through :meth:`Simulator.register_subproblem_step` (see
      :mod:`fomtools.canonicalizers.simulator_subproblems`).
    """

    def _process_primitive(self, step: Step) -> LinearExpression:
        # Skip sync_output_bounds — the simulator evaluates numerically and
        # handles affine solves itself via canonicalize_affine.
        for leaf in step.output_leaves:
            self.register_var(leaf, var_type="param")
        self.canonicalizer.canonicalize(step)
        return step.output

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _const_scalar(value, name: str) -> float:
        """
        Resolve a scalar parameter (``a``/``r``/``lambd``) to a Python float.

        Constants and constant-only ``LinearExpression``s are accepted; a genuinely
        per-sample ``LinearExpression`` parameter (one that depends on leaves) is
        rejected with a clear message — the batched projections bake the scalar into
        the activation closure, so it must not vary across samples.
        """
        if hasattr(value, "decomposition_dict"):
            if value.decomposition_dict:
                raise NotImplementedError(
                    f"{name} as a per-sample LinearExpression parameter is not "
                    "supported on the Simulator backend; pass a constant."
                )
            return float(np.asarray(value.const).reshape(-1)[0])
        return float(value)

    def _register_subproblem(
        self, arg_expr: LinearExpression, n: int, activation
    ) -> LinearExpression:
        """Create a fresh output leaf, register it, and record the subproblem step."""
        out = LinearExpression.from_int(n)
        self.register_var(out, var_type="param")
        self.canonicalizer.register_subproblem_step(arg_expr, out, activation)
        return out

    # ------------------------------------------------------------------
    # Pure linear solve — lower to an Affine step
    # ------------------------------------------------------------------

    def prox_quadratic(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        r"""
        Lower a (convex) quadratic / least-squares prox to an :class:`Affine` step.

        The prox solves :math:`(I + \gamma Q)\,x = \text{arg} - \gamma q`, a single
        linear system whose matrix ``I + γQ`` is positive definite (``Q ⪰ 0``,
        ``γ > 0``) and therefore always nonsingular. Both ``Quadratic`` and
        ``LeastSquares`` expose ``Q`` and ``q`` (the latter as ``Q = AᵀA``,
        ``q = -Aᵀb``), so this one override covers both registry entries.
        """
        n = arg.ndim
        M = np.eye(n) + gamma * prox.Q
        return self.process_step(Affine(arg=arg - gamma * prox.q, output_mat=M))

    def prox_indaffine(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        r"""
        Lower the affine-set indicator prox (projection onto :math:`\{Ax = b\}`)
        to an :class:`Affine` step on the block KKT system

        .. math::
            \begin{bmatrix} I & A^\top \\ A & 0 \end{bmatrix}
            \begin{bmatrix} u \\ \nu \end{bmatrix}
            = \begin{bmatrix} \text{arg} \\ b \end{bmatrix},

        returning the primal slice ``u = x[:n]``. The block matrix is invertible
        when ``A`` has full row rank (assumed by ``IndAffine``). ``b`` may be
        numeric data or a ``LinearExpression`` parameter (lifted via
        :meth:`_lift_to_le`).
        """
        A, m, n = prox.A, prox.m, prox.n
        b_le = self._lift_to_le(prox.b, m)
        M_block = np.block([[np.eye(n), A.T], [A, np.zeros((m, m))]])
        _assert_nonsingular(M_block, "IndAffine prox")
        out = self.process_step(Affine(arg=stack([arg, b_le]), output_mat=M_block))
        return out[:n]

    # ------------------------------------------------------------------
    # Subproblem proxes — batched / Condat / cvxpy activations
    # ------------------------------------------------------------------

    def prox_indsimplex(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        """Project each sample onto the scaled simplex ``{u >= 0, 1^T u = a}`` (batched)."""
        a = self._const_scalar(prox.a, "IndSimplex sum a")
        return self._register_subproblem(arg, arg.ndim, simplex_activation(a))

    def prox_indballl1(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        """Project each sample onto the ℓ1-ball ``{||u||_1 <= r}`` (batched)."""
        r = self._const_scalar(prox.r, "IndBallL1 radius r")
        return self._register_subproblem(arg, arg.ndim, l1_ball_activation(r))

    def prox_norminf(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        """ℓ∞-norm prox via Moreau: ``v - proj_{||.||_1 <= gamma*lambd}(v)`` (batched)."""
        lambd = self._const_scalar(prox.lambd, "NormInf weight lambd")
        return self._register_subproblem(arg, arg.ndim, norm_inf_activation(gamma * lambd))

    def prox_totalvariation1d(
        self, prox, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        """1-D total-variation prox via Condat's direct algorithm (per sample)."""
        lambd = self._const_scalar(prox.lambd, "TotalVariation1D weight lambd")
        return self._register_subproblem(arg, arg.ndim, tv_activation(gamma * lambd))

    def prox_indpolyhedral(
        self, prox, arg: LinearExpression, gamma: float = 1.0
    ) -> LinearExpression:
        """Project each sample onto ``{u : A u <= b}`` via a per-sample cvxpy solve."""
        A, m, n = prox.A, prox.m, prox.n
        if hasattr(prox.b, "decomposition_dict") and prox.b.decomposition_dict:
            # Per-sample b: stack it after the point so the activation can read it.
            arg_expr = stack([arg, prox.b])
            activation = polyhedral_activation(A, None, n, m, b_is_param=True)
            return self._register_subproblem(arg_expr, n, activation)

        if hasattr(prox.b, "decomposition_dict"):  # constant-only LinearExpression
            b_val = np.asarray(prox.b.const, dtype=float)
        else:
            b_val = np.broadcast_to(np.asarray(prox.b, dtype=float), (m,)).copy()
        activation = polyhedral_activation(A, b_val, n, m, b_is_param=False)
        return self._register_subproblem(arg, n, activation)

    # ------------------------------------------------------------------
    # QuadraticProgram / LinearProgram step
    # ------------------------------------------------------------------

    def decompose_quadraticprogram(self, step: Step) -> LinearExpression:
        r"""
        Lower a :class:`QuadraticProgram` / :class:`LinearProgram` step.

        Equality-only / unconstrained QPs with a nonsingular ``P`` are a pure linear
        system and lower to an :class:`Affine` block-KKT solve (batched). QPs with
        inequality constraints, and all LPs (``P = 0``, whose vertex solution is set
        by the active inequalities), are solved per sample with a parametrized cvxpy
        problem (:func:`qp_activation`).
        """
        has_ineq = step.C is not None
        is_lp = getattr(step, "is_lp", False)
        p_is_zero = (spa.issparse(step.P) and step.P.nnz == 0) or (
            not spa.issparse(step.P) and not np.any(step.P)
        )
        if has_ineq or is_lp or p_is_zero:
            return self._qp_via_cvxpy(step)

        # Equality-only / unconstrained linear-solve path.
        n = step.n
        q = step.q if step.q is not None else -step.arg
        q_le = self._lift_to_le(q, n)

        if step.A is None:
            P = step.P
            if not spa.issparse(P):
                _assert_nonsingular(P, "QuadraticProgram (unconstrained)")
            return self.process_step(Affine(arg=-q_le, output_mat=P))

        A = step.A
        p = A.shape[0]
        b_le = self._lift_to_le(step.b, p)
        P_dense = step.P.toarray() if spa.issparse(step.P) else np.asarray(step.P, dtype=float)
        A_dense = A.toarray() if spa.issparse(A) else np.asarray(A, dtype=float)
        M_block = np.block([[P_dense, A_dense.T], [A_dense, np.zeros((p, p))]])
        _assert_nonsingular(M_block, "QuadraticProgram (equality-constrained)")
        out = self.process_step(Affine(arg=stack([-q_le, b_le]), output_mat=M_block))
        return out[:n]

    def _qp_via_cvxpy(self, step: Step) -> LinearExpression:
        """Record a per-sample cvxpy QP/LP solve, stacking ``[q | b | d]`` as input."""
        n = step.n
        q = step.q if step.q is not None else -step.arg
        blocks = [self._lift_to_le(q, n)]

        p = m = 0
        if step.A is not None:
            p = step.A.shape[0]
            blocks.append(self._lift_to_le(step.b, p))
        if step.C is not None:
            m = step.C.shape[0]
            blocks.append(self._lift_to_le(step.d, m))

        arg_expr = stack(blocks) if len(blocks) > 1 else blocks[0]
        activation = qp_activation(step.P, step.A, step.C, n, p, m, getattr(step, "is_lp", False))

        out = step.output  # the QP step's own fresh dim-n leaf
        self.register_var(out, var_type="param")
        self.canonicalizer.register_subproblem_step(arg_expr, out, activation)
        return out

    _DECOMPOSERS: ClassVar[dict] = {
        "quadraticprogram": decompose_quadraticprogram,
        "linearprogram": decompose_quadraticprogram,
    }

    _PROX_DECOMPOSERS: ClassVar[dict] = {
        **StepDecomposer._PROX_DECOMPOSERS,
        "quadratic": prox_quadratic,
        "leastsquares": prox_quadratic,
        "indaffine": prox_indaffine,
        "indpolyhedral": prox_indpolyhedral,
        "indballl1": prox_indballl1,
        "norminf": prox_norminf,
        "indsimplex": prox_indsimplex,
        "totalvariation1d": prox_totalvariation1d,
    }
