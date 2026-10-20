from typing import ClassVar

import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.simulator import (
    l1_ball_activation,
    norm_inf_activation,
    polyhedral_activation,
    qp_activation,
    simplex_activation,
    tv_activation,
)
from fomtools.linearexpression import LinearExpression, stack
from fomtools.stepdecomposers.step_decomposer import StepDecomposer
from fomtools.steps import Affine, SaturatedLinear
from fomtools.steps.step import Step

# A linear system this ill-conditioned yields solutions with no significant digits
# (the relative error ~ cond * eps exceeds ~1e-4 in double precision), so reject it
# as a developer-facing guard rather than silently returning garbage.
_COND_THRESHOLD = 1e12


def _assert_nonsingular(M: np.ndarray, context: str, check_conditioning: bool = True) -> None:
    """
    Raise a clear error if the dense matrix ``M`` is singular or near-singular.

    The Simulator solves the implicit-iterate linear system once per constant
    matrix and reuses the factorization across all samples; a singular matrix
    would otherwise surface as an opaque ``LinAlgError`` deep inside
    ``solve_model``, and a merely *near*-singular one would return inaccurate
    solutions with no warning. We flag both: exact singularity via ``slogdet`` and
    near-singularity via the 2-norm condition number (> ``_COND_THRESHOLD``). Only
    dense matrices are checked here — sparse systems are left to ``splu``, which
    raises on a singular factorization.

    ``check_conditioning`` gates *only* the condition-number test. Saddle-point KKT
    blocks ``[[P, Aᵀ], [A, 0]]`` have a condition number that scales as
    ``O(cond(P)·cond(A)²)``, so a legitimately well-posed equality-constrained QP
    (e.g. ``cond(A) ~ 1e4``) would trip ``_COND_THRESHOLD`` even though the solve is
    accurate. Callers pass ``check_conditioning=False`` for such block systems and
    rely on the exact-singularity check plus ``splu``/``solve`` raising on a genuinely
    singular factorization.
    """
    M = np.asarray(M, dtype=float)
    sign, _ = np.linalg.slogdet(M)
    if sign == 0:
        raise ValueError(
            f"{context}: the linear system matrix is singular and cannot be solved "
            "on the Simulator backend. Ensure the problem is well posed (e.g. the "
            "equality matrix has full row rank)."
        )
    if check_conditioning:
        cond = np.linalg.cond(M)
        if cond > _COND_THRESHOLD:
            raise ValueError(
                f"{context}: the linear system matrix is near-singular (condition number "
                f"{cond:.3e} > {_COND_THRESHOLD:g}); the Simulator solution would be "
                "numerically unreliable. Ensure the problem is well conditioned."
            )


class SimulatorStepDecomposer(StepDecomposer):
    """
    Simulator Step Decomposer.

    Lowers each prox/step to a numeric evaluator recorded with the Simulator
    canonicalizer. Three families:

    - **Primitive** activations (``Relu``/``SaturatedLinear``/``SoftThreshold``):
      forwarded directly via `_process_primitive`.
    - **Pure linear solve** (``Quadratic``/``LeastSquares``, ``IndAffine``, and the
      equality-only / unconstrained ``QuadraticProgram``): lowered to an
      `Affine` step (one batched multi-RHS solve, no Python loop).
    - **Subproblem** activations for the complementarity cases. The simplex,
      ℓ1-ball and ℓ∞-norm proxes use batched solver-free projections; the 1-D
      total-variation prox uses Condat's O(n) algorithm; polyhedral projection and
      inequality QP/LP loop a parametrized cvxpy solve per sample. All are wired
      through `Simulator.register_subproblem_step` (the activation factories
      live alongside it in `fomtools.canonicalizers.simulator`).
    """

    def _process_primitive(self, step: Step) -> LinearExpression:
        # Skip sync_output_bounds — the simulator evaluates numerically and
        # handles affine solves itself via canonicalize_affine.
        for leaf in step.output_leaves:
            self.register_var(leaf, var_type="param")
        self.canonicalizer.canonicalize(step)
        return step.output

    def decompose_saturatedlinear(self, step: SaturatedLinear) -> LinearExpression:
        """Evaluate symbolic saturation bounds through per-sample expressions.

        Constant bounds keep the simulator's native ``clip`` activation. When
        either bound is a ``LinearExpression``, reuse the shared identity

            satlin(x, l, u) = relu(x - l) - relu(x - u) + l,

        so sampled bounds flow through the simulator as ordinary expression
        inputs and are evaluated independently for every sample.
        """
        has_symbolic_bound = hasattr(step.l, "decomposition_dict") or hasattr(
            step.u, "decomposition_dict"
        )
        if not has_symbolic_bound:
            return self._process_primitive(step)
        return self._decompose_saturatedlinear_to_relus(step)

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
        """Create a fresh output leaf, register it, and record the subproblem step.

        Used by the **prox** lowerings (simplex, ℓ1-ball, ℓ∞-norm, TV, polyhedral),
        which have no Step object and therefore no pre-allocated output leaf — so we
        mint one here via ``LinearExpression.from_int(n)``. This differs from the
        **step** path (`_qp_via_cvxpy`), where the ``QuadraticProgram`` step has
        already allocated ``step.output`` and we reuse it rather than create another.
        """
        out = LinearExpression.from_int(n)
        self.register_var(out, var_type="param")
        self.canonicalizer.register_subproblem_step(arg_expr, out, activation)
        return out

    # ------------------------------------------------------------------
    # Pure linear solve — lower to an Affine step
    # ------------------------------------------------------------------

    def prox_quadratic(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        r"""
        Lower a (convex) quadratic / least-squares prox to an `Affine` step.

        The prox solves $(I + \gamma Q)\,x = \text{arg} - \gamma q$, a single
        linear system whose matrix ``I + γQ`` is positive definite (``Q ⪰ 0``,
        ``γ > 0``) and therefore always nonsingular. Both ``Quadratic`` and
        ``LeastSquares`` expose ``Q`` and ``q`` (the latter as ``Q = AᵀA``,
        ``q = -Aᵀb``), so this one override covers both registry entries.
        """
        n = arg.ndim
        if spa.issparse(prox.Q):
            # Keep the system matrix sparse — Affine / canonicalize_affine factorize
            # it with splu. Densifying here would defeat a sparse LeastSquares AᵀA.
            M = spa.eye(n, format="csr") + gamma * prox.Q
        else:
            M = np.eye(n) + gamma * prox.Q
        return self.process_step(Affine(arg=arg - gamma * prox.q, output_mat=M))

    def prox_indaffine(self, prox, arg: LinearExpression, gamma: float = 1.0) -> LinearExpression:
        r"""
        Lower the affine-set indicator prox (projection onto $\{Ax = b\}$)
        to an `Affine` step on the block KKT system

        $$
        \begin{bmatrix} I & A^\top \\ A & 0 \end{bmatrix}
        \begin{bmatrix} u \\ \nu \end{bmatrix}
        = \begin{bmatrix} \text{arg} \\ b \end{bmatrix},
        $$

        returning the primal slice ``u = x[:n]``. The block matrix is invertible
        when ``A`` has full row rank (assumed by ``IndAffine``). ``b`` may be
        numeric data or a ``LinearExpression`` parameter (lifted via
        `_lift_to_le`).
        """
        A, m, n = prox.A, prox.m, prox.n
        b_le = self._lift_to_le(prox.b, m)
        M_block = np.block([[np.eye(n), A.T], [A, np.zeros((m, m))]])
        # Saddle-point block: skip the condition-number test (it scales as cond(A)^2
        # and false-positives on well-posed constraints); exact-singularity + solve suffice.
        _assert_nonsingular(M_block, "IndAffine prox", check_conditioning=False)
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
        Lower a `QuadraticProgram` / `LinearProgram` step.

        Equality-only / unconstrained QPs with a nonsingular ``P`` are a pure linear
        system and lower to an `Affine` block-KKT solve (batched). QPs with
        inequality constraints, and all LPs (``P = 0``, whose vertex solution is set
        by the active inequalities), are solved per sample with a parametrized cvxpy
        problem (`qp_activation`).
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
        # Saddle-point block: skip the condition-number test (see prox_indaffine).
        _assert_nonsingular(
            M_block, "QuadraticProgram (equality-constrained)", check_conditioning=False
        )
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

        # Reuse the step's own pre-allocated output leaf (unlike the prox path in
        # `_register_subproblem`, which must mint a fresh one — a prox has no Step).
        out = step.output  # the QP step's own fresh dim-n leaf
        self.register_var(out, var_type="param")
        self.canonicalizer.register_subproblem_step(arg_expr, out, activation)
        return out

    _DECOMPOSERS: ClassVar[dict] = {
        "saturatedlinear": decompose_saturatedlinear,
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
