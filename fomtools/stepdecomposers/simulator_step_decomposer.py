from typing import ClassVar

import numpy as np
import scipy.sparse as spa

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

    Most steps are forwarded as primitive operations handled directly by the
    Simulator canonicalizer. Implicit-iterate proxes/steps whose output is
    defined by a *pure linear system* with a constant matrix and a per-sample
    right-hand side (``Quadratic``/``LeastSquares``, ``IndAffine``, and the
    equality-only ``QuadraticProgram``) are lowered to an :class:`Affine` step,
    which the simulator evaluates with a single batched multi-RHS solve.

    Implicit-iterate proxes/steps carrying complementarity constraints
    (``IndPolyhedral``, ``IndBallL1``, ``NormInf``, ``IndSimplex``,
    ``TotalVariation1D``, and any QP/LP with inequalities) are *not* a single
    linear system — the active set varies per sample — so they remain deferred.
    """

    def _process_primitive(self, step: Step) -> LinearExpression:
        # Skip sync_output_bounds — the simulator evaluates numerically and
        # handles affine solves itself via canonicalize_affine.
        for leaf in step.output_leaves:
            self.register_var(leaf, var_type="param")
        self.canonicalizer.canonicalize(step)
        return step.output

    # ------------------------------------------------------------------
    # Type-1 (pure linear solve) prox overrides — lower to an Affine step
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
    # QuadraticProgram step override — equality-only / unconstrained QP
    # ------------------------------------------------------------------

    def decompose_quadraticprogram(self, step: Step) -> LinearExpression:
        r"""
        Lower an equality-only / unconstrained :class:`QuadraticProgram` to an
        :class:`Affine` step.

        With no inequality constraints, the QP optimum is the unique solution of
        the KKT linear system

        .. math::
            \begin{bmatrix} P & A^\top \\ A & 0 \end{bmatrix}
            \begin{bmatrix} x \\ \nu \end{bmatrix}
            = \begin{bmatrix} -q \\ b \end{bmatrix}
            \qquad (\text{or } P x = -q \text{ when } A \text{ is absent}),

        which the simulator solves batched. QPs with inequality constraints
        (``step.C is not None``) and linear programs (``P = 0``, whose vertex
        solution is set by active inequalities/bounds, not a linear system) are
        not a single linear solve and are deferred.
        """
        if step.C is not None:
            self._qp_step_not_supported(step)
        if (
            getattr(step, "is_lp", False)
            or (spa.issparse(step.P) and step.P.nnz == 0)
            or (not spa.issparse(step.P) and not np.any(step.P))
        ):
            self._qp_step_not_supported(step)

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

    def _qp_step_not_supported(self, step: Step) -> LinearExpression:
        # QuadraticProgram with inequalities / LinearProgram lower to a KKT system
        # with complementarity (a per-sample active-set / QP solve), which the
        # Simulator's single stacked linear solve cannot produce. Defer.
        raise NotImplementedError(
            f"{type(step).__name__} with inequality constraints is not supported on the "
            "Simulator backend (it requires a per-sample active-set solve, not a single "
            "linear system)."
        )

    _DECOMPOSERS: ClassVar[dict] = {
        "quadraticprogram": decompose_quadraticprogram,
        "linearprogram": _qp_step_not_supported,
    }

    def _implicit_iterate_not_supported(self, prox, arg, gamma=1.0):
        # Shared deferral for complementarity-based implicit-iterate proxes
        # (IndPolyhedral, NormInf, IndSimplex, IndBallL1, TotalVariation1D).
        # These are not a single linear system: the active set varies per
        # sample, so a stacked linear solve cannot evaluate them. Supporting
        # them would require a per-sample projection / QP solver.
        raise NotImplementedError(
            f"{type(prox).__name__} prox is not yet supported on the Simulator backend"
        )

    _PROX_DECOMPOSERS: ClassVar[dict] = {
        **StepDecomposer._PROX_DECOMPOSERS,
        "quadratic": prox_quadratic,
        "leastsquares": prox_quadratic,
        "indaffine": prox_indaffine,
        "indpolyhedral": _implicit_iterate_not_supported,
        "indballl1": _implicit_iterate_not_supported,
        "norminf": _implicit_iterate_not_supported,
        "indsimplex": _implicit_iterate_not_supported,
        "totalvariation1d": _implicit_iterate_not_supported,
    }
