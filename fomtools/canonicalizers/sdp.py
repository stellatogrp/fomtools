import warnings
from itertools import combinations

import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import Constraint
from fomtools.linearexpression import LinearExpression
from fomtools.steps import Affine, Relu
from fomtools.utils.math_utils import repair_crossed_bounds
from fomtools.utils.range_utils import process_1d_ranges, process_2d_ranges
from fomtools.utils.sparse_utils import efficient_csc_pad


class SDP(Canonicalizer):
    """
    Semidefinite programming canonicalizer for verification problems.

    This class converts verification problems involving ReLU and other piecewise
    linear operations into semidefinite programming (SDP) formulations. It builds
    the SDP problem incrementally by adding variables and constraints, then solves
    using Clarabel or CVXPY.

    The SDP uses a lifted matrix variable X where X[0,0] = 1 (homogenization).
    Variables are embedded in the diagonal blocks, and constraints are encoded
    as linear matrix inequalities.

    Attributes
    ----------
    solver_name : str
        Name of the SDP solver backend ('clarabel', 'cvxpy', 'cuclarabel').
    range_marker : int
        Current dimension of the SDP matrix variable.
    var_idx_map : dict
        Mapping from LinearExpression to (start_idx, end_idx) in SDP matrix.
    Aeq_matrices : list[scipy.sparse.csc_matrix]
        Equality constraint matrices.
    beq_consts : list[float]
        Equality constraint right-hand sides.
    Aineq_matrices : list[scipy.sparse.csc_matrix]
        Inequality constraint matrices.
    bineq_upperbounds : list[float]
        Inequality constraint upper bounds.
    psd_cone_indices : list[set]
        Indices defining PSD cone constraints.
    bound_constraint_indices : dict
        Mapping from variable to its bound constraint indices.
    rlt_cuts : bool
        If True, add RLT (Reformulation-Linearization Technique) cuts
        between off-diagonal pairs of SDP indices that co-exist in some
        ``psd_cone_indices`` entry. Each pair yields 4 McCormick
        inequalities. Cuts are lazily rebuilt when bounds, cones, or
        variables change. See paper §4.5 eq 18.
    triangle_cuts : bool
        If True, add lifted triangle cuts for every ReLU step's
        mixed-bound output rows. Each cut multiplies the per-output
        triangle inequality by a nonneg bound multiplier, yielding a
        quadratic-in-X inequality. See paper §4.5 eq 25/26.
    C_obj : scipy.sparse.csc_matrix
        Objective matrix for maximization.
    """

    def __init__(
        self,
        solver_params=None,
        solver="clarabel",
        rlt_cuts=False,
        triangle_cuts=False,
        ineq_product_cuts=True,
        ineq_box_cuts=True,
        ineq_square_cuts=True,
        relu_envelope_cuts=False,
        rlt_all_pairs=False,
        full_psd_cone=False,
        cross_cut=False,
        subgrad_mono_cut=False,
    ):
        """
        Initialize the SDP canonicalizer.

        Parameters
        ----------
        solver_params : dict, optional
            Dictionary of solver-specific parameters to pass to the backend.
        solver : str, default='clarabel'
            Name of the SDP solver to use. Options: 'clarabel', 'cvxpy',
            'cuclarabel'.
        rlt_cuts : bool, default=False
            If True, add RLT (Reformulation-Linearization Technique) cuts
            between cone-mate SDP indices to tighten the relaxation.
        triangle_cuts : bool, default=False
            If True, add lifted triangle cuts for ReLU steps with mixed-bound
            output dimensions to tighten the relaxation.
        ineq_product_cuts : bool, default=True
            For each inequality constraint ``Ax + b <= 0``, add the off-diagonal
            products ``g_i g_j >= 0`` (valid since each ``g_i <= 0``).
        ineq_box_cuts : bool, default=True
            For each inequality constraint, add the RLT products of the constraint
            with the variable box bounds (``g_i (x_p - l_p) <= 0`` and
            ``g_i (u_p - x_p) <= 0``).
        ineq_square_cuts : bool, default=True
            For each inequality constraint, add the self-square upper bound
            ``g_i^2 <= g_i_lower * g_i`` using a propagated lower bound on ``g_i``.
        relu_envelope_cuts : bool, default=False
            If True, add the linear ReLU convex-hull inequalities on the
            first-moment (vector) variables: the lower envelope ``y_i >= a_i^T x
            + b_i`` for every projected row and, on mixed-bound rows
            (``l_i < 0 < u_i``), the upper chord ``y_i <= E_ii (a_i^T x + b_i -
            l_i)`` with ``E_ii = u_i / (u_i - l_i)``. The ``y_i >= 0`` line is
            omitted because the y-leaf box bound plus the PSD constraint already
            imply it.
        rlt_all_pairs : bool, default=False
            If True, RLT *every* pair of variables (not just cone-mates) by
            registering a cheap ``{0, p, q}`` 3x3 PSD minor for each pair so the
            all-pairs McCormick cuts actually bind, then firing ``_build_rlt_cuts``
            on the resulting (now all-pairs) cone structure. This is the middle
            tier between cone-mate-only ``rlt_cuts`` and the dense
            ``full_psd_cone``: ``baseline >= rlt_cuts >= rlt_all_pairs >=
            full_psd_cone + rlt_cuts``. Pairs already inside a larger cone add no
            new minor (``add_psd_cone_indices`` drops subsumed cones), so it only
            couples genuinely-uncoupled pairs. Implies RLT cut emission on its own
            (no need to also set ``rlt_cuts``).
        full_psd_cone : bool, default=False
            If True, union all PSD cones into a single full moment-matrix block
            at solve time. Combined with ``rlt_cuts=True`` this yields the full
            Shor + RLT relaxation (RLT then fires on every variable pair, not
            just cone-mates). Tightest but most expensive configuration.
        cross_cut : bool, default=False
            If True, add the PEP "cross" interpolation cuts ``z_a r_b <= 0`` over
            ordered pairs of ReLU output components, where ``z_a >= 0`` is a ReLU
            output and ``r_b = v_b - z_b <= 0`` is another ReLU's residual (valid
            for every orthant projection). Couples *different* steps, which the
            per-step encoding does not. Each emitted cut self-registers the minimal
            PSD cone over the indices it touches so it binds without
            ``full_psd_cone`` (Clarabel handles many small cones efficiently; the
            ``add_psd_cone_indices`` antichain subsumes a cone already covered by a
            larger one). For saturated-linear (box projection) steps this flag also
            emits the box normal-cone cuts (per coordinate ``j``: ``r_j(z_j-l_j)>=0``,
            ``r_j(z_j-u_j)>=0`` within a step, and ``r_j^b(z_j^a-z_j^b)<=0`` across
            same-box steps), via :meth:`_build_box_normalcone_cuts`.
        subgrad_mono_cut : bool, default=False
            If True, add the PEP subgradient-monotonicity cuts
            ``<s^i/tau_i - s^j/tau_j, z^i> >= 0`` over ordered pairs of registered
            prox steps, with ``s = v - z`` and ``tau`` the prox threshold
            (``tau = 1`` for ReLU/orthant, ``tau = lambda`` for soft-threshold).
            Normalizing by ``tau`` places the subgradient in ``d||.||_1`` so steps
            of the same (homogeneous) function are comparable. On the orthant this
            reduces to the cross cut; on the soft-threshold (ell-1 prox) it is a
            genuinely distinct coupling. Self-registers the PSD cone over the leaves
            each cut couples (without needing ``full_psd_cone``); note that for a
            composite prox (e.g. soft-threshold ``z = relu_pos - relu_neg``) this
            cone spans several index ranges, so it is not as tight as the cross
            cut's truly-minimal ``{0, ya, yb} U x_b``. For saturated-linear (box
            projection) steps this flag also
            emits the box firm-nonexpansiveness/monotonicity cuts (per coordinate:
            ``(z_j^a-z_j^b)(r_j^a-r_j^b)>=0`` across same-box steps), via
            :meth:`_build_box_mono_cuts`.
        """
        super(SDP, self).__init__(solver_params, solver)
        self.solver_name = solver.lower()  # Store for dispatch

        # cuClarabel cannot yet apply custom solver options (its Julia settings
        # are hardcoded). Reject construction-time params eagerly here rather
        # than surprising the user with a failure only at solve time.
        if self.solver_name == "cuclarabel" and self.solver_params:
            raise ValueError(
                "cuClarabel does not support custom solver options yet; "
                f"remove params={self.solver_params} from the Verifier."
            )
        self.range_marker = 1
        self.var_idx_map = {}

        self.Aeq_matrices = []
        self.beq_consts = []
        self.Aineq_matrices = []
        self.bineq_upperbounds = []
        self.psd_cone_indices = []
        self.bound_constraint_indices = {}  # var -> (start_idx, end_idx)

        self.rlt_cuts = rlt_cuts
        self.triangle_cuts = triangle_cuts
        self.ineq_product_cuts = ineq_product_cuts
        self.ineq_box_cuts = ineq_box_cuts
        self.ineq_square_cuts = ineq_square_cuts
        self.relu_envelope_cuts = relu_envelope_cuts
        self.rlt_all_pairs = rlt_all_pairs
        self.full_psd_cone = full_psd_cone
        self.cross_cut = cross_cut
        self.subgrad_mono_cut = subgrad_mono_cut
        self._cuts_dirty = True
        self._cut_ineq_matrices = []
        self._cut_bineq_upperbounds = []
        self._relu_steps = []  # registry for triangle / envelope / cross cuts
        self._prox_steps = []  # registry for subgradient-monotonicity cuts
        self._box_prox_steps = []  # registry for satlin (box projection) PEP cuts
        self._ineq_constraints = []  # (constraint, add_product_cuts) for inequality cuts

        self.C_obj = None
        self._objective_spec = None
        self._solved_copy = None
        self._initialize_corner_constraint()

    def _initialize_corner_constraint(self) -> None:
        """
        Add the homogenization constraint X[0,0] = 1.

        This constraint ensures the lifted SDP formulation is properly
        normalized. It is added automatically during initialization.
        """
        curr_prob_dim = self.get_sdp_var_dim()
        new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
        new_A[0, 0] = 1
        self.Aeq_matrices.append(symmetrize(new_A.tocsc()))
        self.beq_consts.append(1)

    def _pad_existing_A_matrices(self) -> None:
        """
        Expand existing constraint matrices to accommodate new variables.

        When new variables are added, the SDP matrix dimension increases.
        This method pads all existing equality and inequality constraint
        matrices with zeros to match the new dimension.
        """
        new_dim = self.get_sdp_var_dim()
        new_Aeq_matrices = [efficient_csc_pad(A, (new_dim, new_dim)) for A in self.Aeq_matrices]
        new_Aineq_matrices = [efficient_csc_pad(A, (new_dim, new_dim)) for A in self.Aineq_matrices]
        self.Aeq_matrices = new_Aeq_matrices
        self.Aineq_matrices = new_Aineq_matrices

    def add_var(self, var: LinearExpression, **kwargs) -> None:
        """
        Add a variable to the SDP with bound constraints.

        Embeds the variable in the SDP matrix diagonal and adds quadratic
        bound constraints of the form (x - l)(x - u) <= 0 for each
        component, which encode the box constraints l <= x <= u.

        Parameters
        ----------
        var : LinearExpression
            The variable to add. Must have finite bounds in ``var.lb``
            and ``var.ub``.

        Raises
        ------
        NotImplementedError
            If any bound is infinite.
        """
        lb, ub = var.lb, var.ub
        dim = var.ndim
        bounds = (self.range_marker, self.range_marker + dim)
        self.range_marker += dim
        self.var_idx_map[var] = bounds

        # whenever a new var is added, need to pad existing A matrices
        self._pad_existing_A_matrices()

        curr_prob_dim = self.get_sdp_var_dim()
        constraint_start = len(self.Aineq_matrices)

        for i, l, u in zip(range(bounds[0], bounds[1]), lb, ub):
            if np.isinf(l) or np.isinf(u):
                raise NotImplementedError("Infinite bounds not yet supported in SDP canonicalizer")
            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            new_A[i, i] = 1
            new_A[i, 0] = -(l + u)

            new_A = new_A.tocsc()
            new_A = symmetrize(new_A)
            self.Aineq_matrices.append(new_A)
            self.bineq_upperbounds.append(-u * l)

        constraint_end = len(self.Aineq_matrices)
        self.bound_constraint_indices[var] = (constraint_start, constraint_end)
        self._cuts_dirty = True

    def get_sdp_var_dim(self) -> int:
        """
        Get the current dimension of the SDP matrix variable.

        Returns
        -------
        int
            The current size of the symmetric matrix variable X.
        """
        return self.range_marker

    def update_bounds(
        self, expression: LinearExpression, lb: float | np.ndarray, ub: float | np.ndarray
    ) -> None:
        """
        Update bounds for a variable by regenerating its constraint matrices.

        Modifies the existing inequality constraint matrices for the given
        variable to reflect new lower and upper bounds. This is used during
        bound tightening procedures.

        Parameters
        ----------
        expression : LinearExpression
            The variable whose bounds are being updated. Must already be
            registered via ``add_var``.
        lb : np.ndarray
            New lower bounds. Must have length equal to ``expression.ndim``.
        ub : np.ndarray
            New upper bounds. Must have length equal to ``expression.ndim``.

        Raises
        ------
        ValueError
            If the expression is not found in the canonicalizer, or if
            the bounds do not have the expected length.
        NotImplementedError
            If any new bound is infinite.
        """
        if expression not in self.bound_constraint_indices:
            raise ValueError("Expression not found in SDP canonicalizer")

        constraint_start, constraint_end = self.bound_constraint_indices[expression]
        sdp_start, sdp_end = self.var_idx_map[expression]
        curr_prob_dim = self.get_sdp_var_dim()

        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)

        expected_len = sdp_end - sdp_start
        if len(lb) != expected_len:
            raise ValueError(f"Expected lb of length {expected_len}, got {len(lb)}")
        if len(ub) != expected_len:
            raise ValueError(f"Expected ub of length {expected_len}, got {len(ub)}")

        for j, (i, l, u) in enumerate(zip(range(sdp_start, sdp_end), lb, ub)):
            if np.isinf(l) or np.isinf(u):
                raise NotImplementedError("Infinite bounds not yet supported in SDP canonicalizer")

            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            new_A[i, i] = 1
            new_A[i, 0] = -(l + u)

            constraint_idx = constraint_start + j
            self.Aineq_matrices[constraint_idx] = symmetrize(new_A.tocsc())
            self.bineq_upperbounds[constraint_idx] = -u * l

        self._cuts_dirty = True

    def add_param_stack_var(self, param_stack_var, param_list):
        """
        Stack parameter variables (not implemented for SDP).

        Raises
        ------
        NotImplementedError
            This method is not supported in the SDP canonicalizer.
        """
        raise NotImplementedError

    def copy(self) -> "SDP":
        """
        Create an independent copy of this SDP canonicalizer.

        Returns a new SDP canonicalizer with deep copies of all mutable
        state (constraint matrices, cone indices, etc.). The copy can be
        modified without affecting the original.

        Returns
        -------
        SDP
            A new canonicalizer instance with copied state.
        """
        new_canon = SDP.__new__(SDP)
        new_canon.solver_params = dict(self.solver_params) if self.solver_params else None
        new_canon.solver_name = self.solver_name
        new_canon.solver = None  # SDP solver is instantiated at solve time
        new_canon.range_marker = self.range_marker
        new_canon.var_idx_map = self.var_idx_map.copy()  # shallow copy ok (immutable values)

        # Deep copy mutable arrays/lists
        new_canon.Aeq_matrices = [m.copy() for m in self.Aeq_matrices]
        new_canon.beq_consts = self.beq_consts.copy()
        new_canon.Aineq_matrices = [m.copy() for m in self.Aineq_matrices]
        new_canon.bineq_upperbounds = self.bineq_upperbounds.copy()
        new_canon.psd_cone_indices = [s.copy() for s in self.psd_cone_indices]
        new_canon.bound_constraint_indices = self.bound_constraint_indices.copy()
        new_canon.rlt_cuts = self.rlt_cuts
        new_canon.triangle_cuts = self.triangle_cuts
        new_canon.ineq_product_cuts = self.ineq_product_cuts
        new_canon.ineq_box_cuts = self.ineq_box_cuts
        new_canon.ineq_square_cuts = self.ineq_square_cuts
        new_canon.relu_envelope_cuts = self.relu_envelope_cuts
        new_canon.rlt_all_pairs = self.rlt_all_pairs
        new_canon.full_psd_cone = self.full_psd_cone
        new_canon.cross_cut = self.cross_cut
        new_canon.subgrad_mono_cut = self.subgrad_mono_cut
        # Cuts get rebuilt from current state on the copy's first solve.
        new_canon._cuts_dirty = True
        new_canon._cut_ineq_matrices = []
        new_canon._cut_bineq_upperbounds = []
        new_canon._relu_steps = list(self._relu_steps)  # shallow: Relu objs are shared
        new_canon._prox_steps = list(self._prox_steps)  # shallow: exprs/scalars immutable
        new_canon._box_prox_steps = list(self._box_prox_steps)  # shallow: exprs/arrays immutable
        new_canon._ineq_constraints = list(self._ineq_constraints)  # shallow: (constr, flag) tuples
        new_canon.C_obj = self.C_obj.copy() if self.C_obj is not None else None
        new_canon.obbt_handler = self.obbt_handler
        new_canon._objective_spec = self._objective_spec
        new_canon._solved_copy = None

        return new_canon

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint to the SDP instance.

        Dispatches to the appropriate implementation based on the constraint
        operator type (equality or inequality).

        Parameters
        ----------
        constraint : Constraint
            Constraint object to add to the SDP instance.
        """
        match constraint.operator:
            case "=":
                self._add_equality_constraint_impl(constraint)
            case "<=":
                self._add_inequality_constraint_impl(constraint)
            case "complementarity":
                self._add_complementarity_constraint_impl(constraint)
            case "sum_squares":
                self._add_sum_squares_constraint_impl(constraint)
            case _:
                raise ValueError(f"Unknown constraint operator: {constraint.operator}")

    def _add_equality_constraint_impl(self, constraint: Constraint) -> None:
        """
        Add an equality constraint and its quadratic strengthening.

        For Ax + b = 0, adds:
        1. Linear constraint in lifted form
        2. Quadratic constraint: diag([b|A] X [b|A]^T) = 0

        With the PSD constraint, the diagonal being zero implies the full
        quadratic form is zero. Note this is already the full equality-RLT
        closure within the cone: ``c_i^T X c_i = 0`` with ``X ⪰ 0`` forces the
        whole column ``X c_i = 0``, i.e. every product ``h_i(x) x_p = 0`` for
        cone-mate ``p`` — so no separate equality-RLT cuts are needed.

        Parameters
        ----------
        constraint : Constraint
            An equality constraint (operator "=").
        """
        A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(constraint.lhs)
        b_vec = constraint.lhs.const

        m, n = A_mat.shape
        curr_prob_dim = self.get_sdp_var_dim()

        # Linear: X[0, x_indices] (row=True, then symmetrize handles X[x, 0])
        x_idx_ranges_1d = process_1d_ranges(x_idx_bounds, row=True)

        # Quadratic: X[[0, x], [0, x]] for the outer product
        all_idx_bounds = [(0, 1)] + x_idx_bounds
        all_idx_ranges_2d = process_2d_ranges(all_idx_bounds, all_idx_bounds)

        for i in range(m):
            # Linear: A_mat[i, :] @ X[x, 0] = -b[i]
            M_lin = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            M_lin[x_idx_ranges_1d] = A_mat[i, :]
            self.Aeq_matrices.append(symmetrize(M_lin.tocsc()))
            self.beq_consts.append(-b_vec[i])

            # Quadratic: c_i^T X c_i = 0 where c_i = [b_i, A_i]
            c_i = np.concatenate([[b_vec[i]], np.asarray(A_mat[i, :]).flatten()])
            M_quad = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            M_quad[all_idx_ranges_2d] = np.outer(c_i, c_i)
            self.Aeq_matrices.append(symmetrize(M_quad.tocsc()))
            self.beq_consts.append(0)

        # PSD cone for {0} ∪ x_indices (also marks _cuts_dirty via add_psd_cone_indices)
        self.add_psd_cone_indices(x_idx_bounds)

    def _add_inequality_constraint_impl(
        self, constraint: Constraint, add_product_cuts: bool = True
    ) -> None:
        """
        Add an inequality constraint ``Ax + b <= 0`` and register it for RLT cuts.

        Adds the base linear constraint ``a_i^T x <= -b_i`` per row (lifted into
        ``Aineq_matrices`` / ``bineq_upperbounds``) and a PSD cone over ``{0} ∪ x``
        so the quadratic block is constrained (matching the equality impl). The
        constraint is recorded in ``self._ineq_constraints`` so the lazy cut builder
        can emit the optional RLT strengthenings gated by the ``ineq_*_cuts`` flags;
        those are (re)built in ``_rebuild_cuts_if_dirty``.

        Parameters
        ----------
        constraint : Constraint
            An inequality constraint (operator "<=").
        add_product_cuts : bool, default True
            Whether the **product cuts** (``g_i g_j >= 0`` for ``i < j``, the
            ``ineq_product_cuts`` family) may be emitted for this constraint. The box
            and square cut families are always eligible (subject to their global flags);
            only the cross-product family is gated here.

            Set ``False`` for the **nonnegativity sides of a complementarity constraint**
            (``e1 >= 0`` / ``e2 >= 0`` emitted by
            :meth:`_add_complementarity_constraint_impl`). Rationale:

            - The product family is ``O(m^2)`` per constraint — the dominant cut cost —
              and each complementarity adds two such sides, so leaving it on multiplies
              the cut count for multi-complementarity proxes (``IndBallL1``, ``TV1D``).
            - Empirically these product cuts add **no** tightening on those sides: the box
              and square cuts already drive the relaxation to the same value (e.g.
              ``IndBallL1`` stays exact at a fixed point with product cuts off but box/
              square on). Keeping box/square is what preserves tightness.
            - For ``IndPolyhedral`` the ``e2 = b - A u >= 0`` side also duplicates the
              explicitly-posted ``A u <= b``; dropping its product cuts removes that
              duplication too.

            The genuine problem inequalities (e.g. the explicit ``A u <= b``) are added
            with the default ``True`` and keep generating product cuts — those *do*
            tighten (the ``IndPolyhedral`` 4.0 -> 0.25 test relies on them), so the family
            is gated per-constraint rather than globally.
        """
        A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(constraint.lhs)
        b_vec = constraint.lhs.const

        m, _n = A_mat.shape
        curr_prob_dim = self.get_sdp_var_dim()

        # Linear: X[0, x_indices] (row=True, then symmetrize handles X[x, 0]).
        x_idx_ranges_1d = process_1d_ranges(x_idx_bounds, row=True)

        for i in range(m):
            # a_i^T x <= -b_i
            M_lin = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            M_lin[x_idx_ranges_1d] = A_mat[i, :]
            self.Aineq_matrices.append(symmetrize(M_lin.tocsc()))
            self.bineq_upperbounds.append(-b_vec[i])

        # PSD cone for {0} ∪ x_indices (also marks _cuts_dirty via add_psd_cone_indices).
        self.add_psd_cone_indices(x_idx_bounds)

        # Register for lazy cut generation. The per-constraint ``add_product_cuts`` flag is
        # stored alongside so the builder can skip only the (expensive) product family for
        # complementarity nonnegativity sides while still emitting box/square cuts.
        self._ineq_constraints.append((constraint, add_product_cuts))
        self._cuts_dirty = True

    def _add_sum_squares_constraint_impl(self, constraint) -> None:
        r"""
        Add a sum-of-squares constraint ``sum_i (a_i^T x + b_i)^2  op  r^2``.

        With ``C = [b; A]^T [b; A]`` the squared two-norm is linear in the lifted matrix:
        ``sum_i (a_i^T x + b_i)^2 = tr(C X)`` (since ``X[0, 0] = 1``). Each sense becomes a
        single linear constraint on ``X``:

        - ``"<="`` : ``tr(C X) <= r^2``      -> ``Aineq``  with ub ``r^2``
        - ``">="`` : ``tr(C X) >= r^2``      -> ``Aineq``  on ``-C`` with ub ``-r^2``
        - ``"=="`` : ``tr(C X)  = r^2``      -> ``Aeq``    with rhs ``r^2``

        All senses are valid relaxations: the rank-1 lift ``X = [1; x][1; x]^T`` attains
        ``tr(C X) = ||A x + b||^2`` exactly, so no feasible point is cut (for ``>=`` / ``==``
        the PSD relaxation is merely looser, never invalid). A PSD cone over ``{0} U x`` is
        registered so the quadratic block is coupled, matching the (in)equality impls.

        Parameters
        ----------
        constraint : QuadraticConstraint
            A sum-of-squares constraint (operator ``"sum_squares"``).
        """
        A_mat, idx_bounds = self.linexpr_to_Amat_and_bounds(constraint.expr)
        b_vec = constraint.expr.const.reshape((-1, 1))
        curr_prob_dim = self.get_sdp_var_dim()

        # Build C = [b; A]^T [b; A] in lifted coordinates, mirroring _apply_two_norm_objective.
        C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
        C_mat[process_2d_ranges(idx_bounds, idx_bounds)] = A_mat.T @ A_mat
        C_mat[process_1d_ranges(idx_bounds, row=True)] = b_vec.T @ A_mat
        C_mat[process_1d_ranges(idx_bounds, row=False)] = A_mat.T @ b_vec
        C_mat[0, 0] = float(b_vec.flatten() @ b_vec.flatten())  # b^T b (scalar)
        C = symmetrize(C_mat.tocsc())

        r_sq = constraint.radius_sq
        match constraint.sense:
            case "<=":
                self.Aineq_matrices.append(C)
                self.bineq_upperbounds.append(r_sq)
            case ">=":
                self.Aineq_matrices.append(-C)
                self.bineq_upperbounds.append(-r_sq)
            case "==":
                self.Aeq_matrices.append(C)
                self.beq_consts.append(r_sq)

        # PSD cone for {0} U x_indices (also marks _cuts_dirty via add_psd_cone_indices).
        self.add_psd_cone_indices(idx_bounds)

    def _add_complementarity_constraint_impl(self, constraint) -> None:
        r"""
        Encode elementwise complementarity ``e1 ⊥ e2`` (both ``>= 0``) in the lifted SDP.

        With homogenized coefficient vectors ``c1_i`` / ``c2_i`` (so ``e1_i = c1_i^T w``,
        ``e2_i = c2_i^T w``), the product ``e1_i e2_i = c1_i^T X c2_i`` is linear in ``X``.
        For each row ``i`` this adds the equality ``<sym(c1_i c2_i^T), X> = 0`` (the
        cross-product generalization of the equality strengthening in
        :meth:`_add_equality_constraint_impl`), the nonnegativities ``e1 >= 0`` and
        ``e2 >= 0`` (via :meth:`_add_inequality_constraint_impl`), and a PSD cone over
        ``{0} ∪ vars(e1) ∪ vars(e2)`` so the bilinear entries are constrained.

        Parameters
        ----------
        constraint : ComplementarityConstraint
            Carries ``expr1`` / ``expr2`` (each a length-``ndim`` LinearExpression).
        """
        e1, e2 = constraint.expr1, constraint.expr2
        A1, idx1 = self.linexpr_to_Amat_and_bounds(e1)
        A2, idx2 = self.linexpr_to_Amat_and_bounds(e2)
        b1 = np.asarray(e1.const).flatten()
        b2 = np.asarray(e2.const).flatten()

        # The compact outer-product scatter assumes the two expressions reference
        # disjoint variables (true for λ ⊥ (b−Au): λ and u are distinct leaves).
        # Overlapping leaves would double-index the block; guard rather than mis-encode.
        # Compare the underlying leaf variables directly (decomposition_dict keys), not the
        # (start, end) index ranges.
        if set(e1.decomposition_dict) & set(e2.decomposition_dict):
            raise NotImplementedError(
                "SDP complementarity between expressions sharing variables is not supported."
            )

        n1 = sum(e - s for (s, e) in idx1)
        n2 = sum(e - s for (s, e) in idx2)
        curr = self.get_sdp_var_dim()
        # Compact layout over [const, e1 vars, e2 vars].
        all_idx_bounds = [(0, 1)] + idx1 + idx2
        idx_2d = process_2d_ranges(all_idx_bounds, all_idx_bounds)

        for i in range(constraint.ndim):
            c1 = np.concatenate([[b1[i]], np.asarray(A1[i, :]).flatten(), np.zeros(n2)])
            c2 = np.concatenate([[b2[i]], np.zeros(n1), np.asarray(A2[i, :]).flatten()])
            M = spa.lil_matrix((curr, curr))
            M[idx_2d] = np.outer(c1, c2)
            self.Aeq_matrices.append(symmetrize(M.tocsc()))
            self.beq_consts.append(0)

        # Nonnegativity of both sides (``e1 >= 0`` -> ``-e1 <= 0``). Skip the (O(m^2),
        # tightening-neutral) product-cut family on these sides; box/square cuts still
        # apply and are what preserve tightness (see ``add_product_cuts``).
        self._add_inequality_constraint_impl(e1 >= 0, add_product_cuts=False)
        self._add_inequality_constraint_impl(e2 >= 0, add_product_cuts=False)

        # Constrain the bilinear entries.
        self.add_psd_cone_indices(idx1 + idx2)

    def register_prox_step(
        self, z: LinearExpression, v: LinearExpression, tau: float, *, func: str
    ) -> None:
        """
        Register a prox step for the subgradient-monotonicity cut.

        Records ``(z, v, tau, func)`` where ``z`` is the prox output expression,
        ``v`` is the prox input expression (so the subgradient is ``s = v - z``),
        ``tau`` is the prox threshold (``tau = 1`` for orthant projection / ReLU,
        ``tau = lambda`` for soft-thresholding), and ``func`` names the sublinear
        function the step is the prox of. The subgradient cut builder
        (:meth:`_build_subgrad_mono_cuts`) consumes this registry and pairs only
        steps with the same ``func`` (the cut is invalid across functions).

        Parameters
        ----------
        z : LinearExpression
            Prox output. May be composite (e.g. ``relu_pos - relu_neg`` for
            soft-thresholding); leaves must already be registered in
            ``var_idx_map`` at call time.
        v : LinearExpression
            Prox input (``step.arg``).
        tau : float
            Prox threshold used to normalize the subgradient.
        func : str
            Identifier of the sublinear function (``"nonneg_orthant"`` for ReLU,
            ``"l1"`` for soft-thresholding).
        """
        self._prox_steps.append({"z": z, "v": v, "tau": float(tau), "func": func})
        self._cuts_dirty = True

    def register_box_prox_step(self, z: LinearExpression, v: LinearExpression, l, u) -> None:
        """
        Register a saturated-linear (box projection) step for the PEP box cuts.

        Records ``(z, v, l, u)`` where ``z = clip(v, l, u)`` is the composite
        projection output (``relu(v - l) - relu(v - u) + l`` in the SDP lift), ``v``
        is the input expression (``step.arg``; residual ``r = v - z``), and ``l`` /
        ``u`` are the box bounds (constant arrays or ``LinearExpression`` if
        parametric). Consumed by :meth:`_build_box_normalcone_cuts` (under
        ``cross_cut``) and :meth:`_build_box_mono_cuts` (under ``subgrad_mono_cut``).

        Parameters
        ----------
        z : LinearExpression
            Box-projection output (composite); leaves already in ``var_idx_map``.
        v : LinearExpression
            Box-projection input (``step.arg``).
        l, u : np.ndarray | LinearExpression
            Lower/upper box bounds (per the operator's ``l`` / ``u``).
        """
        self._box_prox_steps.append({"z": z, "v": v, "l": l, "u": u})
        self._cuts_dirty = True

    def add_psd_cone_indices(self, new_indices):
        """
        Register a new PSD cone constraint on a submatrix.

        Adds the constraint that the submatrix of X indexed by ``new_indices``
        (including the homogenization index 0) is positive semidefinite.

        This method maintains an antichain: no stored set is a subset of another.
        If a matrix X is PSD, any principal submatrix is also PSD, so subset
        constraints are redundant.

        Parameters
        ----------
        new_indices : list[tuple[int, int]]
            List of (start, end) index ranges defining the submatrix.
            Index 0 is automatically included for homogenization.
        """
        new_idx_set = range_to_indices(new_indices)
        new_idx_set.add(0)

        # Check if new set is contained in any existing set (redundant)
        for existing_set in self.psd_cone_indices:
            if new_idx_set <= existing_set:
                return

        # Filter out existing sets that are strict subsets of the new set
        self.psd_cone_indices = [s for s in self.psd_cone_indices if not s < new_idx_set]

        self.psd_cone_indices.append(new_idx_set)
        self._cuts_dirty = True

    def canonicalize_relu(self, step: Relu) -> None:
        """
        Encode a ReLU step as SDP constraints.

        Converts the relationship y = ReLU(Ax + b) into equality constraints
        on the SDP matrix variable and registers the appropriate PSD cone
        constraint for the lifted formulation.

        Operates uniformly over ``step.proj_indices`` and binds equalities to
        ``step.output_leaves[0]``. For full ReLU, ``proj_indices`` covers every
        arg row and ``output_leaves[0]`` is the full output leaf — equivalent to a
        standard full-output encoding. For partial-projection ReLU, only the
        projected rows get lifted equalities and ``output_leaves[0]`` is the
        smaller fresh proj-leaf; the non-projected rows are structurally
        fused into shared arg leaves at composition time, so no equalities
        and no SDP rows are emitted for them.

        Parameters
        ----------
        step : Relu
            The ReLU step containing input expression (arg) and output
            variable (output).
        """
        arg = step.arg
        A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(arg)
        b_vec = arg.const

        y_leaf = step.output_leaves[0]
        proj_list = sorted(step.proj_indices)

        y_idx_bounds = [self.var_idx_map[y_leaf]]
        relu_psd_cone_indices = y_idx_bounds + x_idx_bounds

        y_idx_ranges_2d = process_2d_ranges(y_idx_bounds, y_idx_bounds)
        y_idx_ranges_1d = process_1d_ranges(y_idx_bounds)
        xy_idx_ranges = process_2d_ranges(x_idx_bounds, y_idx_bounds)
        I = np.eye(y_leaf.ndim)

        curr_prob_dim = self.get_sdp_var_dim()
        for j, i in enumerate(proj_list):
            # j indexes y_leaf (size = len(proj_list));
            # i indexes arg row in A_mat / b_vec (size = arg.ndim).
            Ij = I[j]
            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            new_A[y_idx_ranges_2d] = np.outer(Ij, Ij)
            new_A[xy_idx_ranges] = -np.outer(A_mat[i], Ij)
            new_A[y_idx_ranges_1d] = -b_vec[i] * Ij

            self.Aeq_matrices.append(symmetrize(new_A.tocsc()))
            self.beq_consts.append(0)

        self.add_psd_cone_indices(relu_psd_cone_indices)
        self._relu_steps.append(step)

        # Register the ReLU as a prox step (orthant projection, tau=1) for the
        # subgradient-monotonicity cut — unless this ReLU is an internal piece of
        # a soft-threshold decomposition, in which case the composite prox is
        # registered by the decomposer instead (see SDPStepDecomposer). The ReLU
        # still joins ``_relu_steps`` above so the cross cut sees every node.
        if not getattr(step, "_skip_prox_registration", False):
            self.register_prox_step(z=step.output, v=step.arg, tau=1.0, func="nonneg_orthant")

        self._cuts_dirty = True

    def canonicalize_affine(self, step: Affine) -> None:
        """
        Encode an Affine step as a linear equality constraint.

        The Affine step asserts ``step.output_mat @ step.output == step.arg``;
        this is the same equality MIP encodes (``mip.py:404``). On SDP it
        flows through ``add_constraint`` → ``_add_equality_constraint_impl``,
        which emits the linear matrix, the quadratic strengthening, and
        the PSD cone membership for the involved leaves.

        Parameters
        ----------
        step : Affine
            The affine step to canonicalize.
        """
        self.add_constraint(step.output_mat @ step.output == step.arg)

    _CANONICALIZERS = {
        "affine": canonicalize_affine,
        "relu": canonicalize_relu,
    }

    def linexpr_to_Amat_and_bounds(self, expr: LinearExpression):
        """
        Extract the coefficient matrix and SDP indices from a LinearExpression.

        Converts a LinearExpression of the form Ax + b into the coefficient
        matrix A and the index ranges where the corresponding variables are
        located in the SDP matrix.

        Parameters
        ----------
        expr : LinearExpression
            The expression to decompose.

        Returns
        -------
        tuple[np.ndarray, list[tuple[int, int]]]
            A tuple containing:
            - The horizontally stacked coefficient matrix.
            - List of (start, end) index bounds for each variable block.
        """
        all_A = []
        idx_bounds = []
        for leaf_var, mat in expr.decomposition_dict.items():
            all_A.append(mat.todense() if spa.issparse(mat) else mat)
            idx_bounds.append(self.var_idx_map[leaf_var])

        return np.hstack(all_A), idx_bounds

    def set_two_norm_objective(self, expr_list, lb_list=None, ub_list=None) -> None:
        """
        Store two-norm objective specification for deferred application.

        The objective will be applied to a copy of the canonicalizer at
        solve time.

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose squared norms are summed in the
            objective.
        lb_list : list[np.ndarray], optional
            Lower bounds for each expression (unused in SDP, for interface consistency).
        ub_list : list[np.ndarray], optional
            Upper bounds for each expression (unused in SDP, for interface consistency).
        """
        self._objective_spec = ("two_norm", expr_list)

    def _apply_two_norm_objective(self, canon_copy, expr_list) -> None:
        """
        Apply two-norm objective to a canonicalizer copy.

        Configures the SDP objective matrix C to compute
        sum_i ||A_i x + b_i||_2^2 = sum_i tr(Q_i @ X) for appropriate
        quadratic forms Q_i derived from each expression.

        Also adds a PSD cone constraint covering all variables involved
        in the objective to ensure cross-terms are properly constrained.

        Parameters
        ----------
        canon_copy : SDP
            The canonicalizer copy to apply the objective to.
        expr_list : list[LinearExpression]
            List of expressions whose squared norms are summed in the
            objective.
        """
        curr_prob_dim = canon_copy.get_sdp_var_dim()
        overall_C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
        all_idx_bounds = []

        for expr in expr_list:
            C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            A_mat, idx_bounds = canon_copy.linexpr_to_Amat_and_bounds(expr)
            b_vec = expr.const.reshape((-1, 1))

            # Collect all index bounds for the combined PSD cone
            all_idx_bounds.extend(idx_bounds)

            ATA = A_mat.T @ A_mat
            bTA = b_vec.T @ A_mat
            ATb = A_mat.T @ b_vec
            bbT = b_vec.T @ b_vec

            idx_ranges_2d = process_2d_ranges(idx_bounds, idx_bounds)
            idx_ranges_1d_row = process_1d_ranges(idx_bounds, row=True)
            idx_ranges_1d_col = process_1d_ranges(idx_bounds, row=False)

            C_mat[idx_ranges_2d] = ATA
            C_mat[idx_ranges_1d_row] = bTA
            C_mat[idx_ranges_1d_col] = ATb
            C_mat[0, 0] = np.trace(bbT)

            overall_C_mat += symmetrize(C_mat)

        # Add PSD cone covering all variables in the objective
        # This ensures cross-terms between different variables are constrained
        canon_copy.add_psd_cone_indices(all_idx_bounds)

        canon_copy.C_obj = overall_C_mat

    def set_infinity_norm_objective(self, expr_list, lb=None, ub=None) -> None:
        """
        Store infinity-norm objective specification for deferred application.

        Uses the identities:
        - abs(x) = relu(x) + relu(-x)
        - max(a, b) = a + relu(b - a)

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose infinity-norm maximum is the objective.
        lb, ub : optional
            Unused (kept for interface compatibility with MIP).
        """
        self._objective_spec = ("infinity_norm", expr_list)

    def set_one_norm_objective(self, expr_list) -> None:
        """
        Store one-norm objective specification for deferred application.

        The objective will be applied to a copy of the canonicalizer at
        solve time using the identity: ||x||_1 = 1^T @ (relu(x) + relu(-x)).

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose one-norms are summed in the objective.
        """
        self._objective_spec = ("one_norm", expr_list)

    def _apply_one_norm_objective(self, canon_copy, expr_list) -> None:
        """
        Apply one-norm objective to a canonicalizer copy.

        Uses the identity: ||x||_1 = 1^T @ (relu(x) + relu(-x))

        Creates auxiliary ReLU variables, adds their constraints, and sets
        up a linear objective to sum the ReLU outputs.

        Parameters
        ----------
        canon_copy : SDP
            The canonicalizer copy to apply the objective to.
        expr_list : list[LinearExpression]
            List of expressions whose one-norms are summed in the objective.
        """
        relu_output_vars = []

        for expr in expr_list:
            relu_output_vars.append(self._create_relu_step(canon_copy, expr))
            relu_output_vars.append(self._create_relu_step(canon_copy, -expr))

        # Build linear objective: maximize sum of all relu outputs
        # For lifted SDP with X = [1, x]^T @ [1, x], linear term c^T x
        # is encoded as C[0, idx] = c/2, C[idx, 0] = c/2
        curr_prob_dim = canon_copy.get_sdp_var_dim()
        C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))

        all_idx_bounds = []
        for output_var in relu_output_vars:
            idx_bounds = canon_copy.var_idx_map[output_var]
            all_idx_bounds.append(idx_bounds)
            start, end = idx_bounds
            # Coefficient 1 for each component, divided by 2 for symmetry
            C_mat[0, start:end] = 0.5
            C_mat[start:end, 0] = 0.5

        # Add PSD cone covering all relu output variables
        canon_copy.add_psd_cone_indices(all_idx_bounds)

        canon_copy.C_obj = C_mat.tocsc()

    def _apply_infinity_norm_objective(self, canon_copy, expr_list) -> None:
        """
        Apply infinity-norm objective using ReLU decomposition.

        Computes ||x||_∞ = max_i |x_i| via:
        - abs(x_i) = relu(x_i) + relu(-x_i)
        - max(M, abs_i) = M + relu(abs_i - M)

        Total ReLUs: 3n - 1 (2n for absolute values, n-1 for max operations)

        Parameters
        ----------
        canon_copy : SDP
            The canonicalizer copy to apply the objective to.
        expr_list : list[LinearExpression]
            List of expressions whose infinity-norm maximum is the objective.
        """
        all_idx_bounds = []
        M_current = None  # Running maximum (scalar LinearExpression)

        for expr in expr_list:
            for i in range(expr.ndim):
                x_i = expr[i]  # Scalar slice

                # abs(x_i) = relu(x_i) + relu(-x_i)
                relu_pos = self._create_relu_step(canon_copy, x_i)
                relu_neg = self._create_relu_step(canon_copy, -x_i)
                abs_i = relu_pos + relu_neg

                all_idx_bounds.append(canon_copy.var_idx_map[relu_pos])
                all_idx_bounds.append(canon_copy.var_idx_map[relu_neg])

                if M_current is None:
                    M_current = abs_i
                else:
                    # max(M, abs_i) = M + relu(abs_i - M)
                    diff = abs_i - M_current
                    relu_max = self._create_relu_step(canon_copy, diff)
                    all_idx_bounds.append(canon_copy.var_idx_map[relu_max])
                    M_current = M_current + relu_max

        # Build objective matrix for M_current (scalar linear expression)
        curr_prob_dim = canon_copy.get_sdp_var_dim()
        C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))

        A_mat, idx_bounds = canon_copy.linexpr_to_Amat_and_bounds(M_current)
        b_vec = M_current.const

        # Linear c^T x encoded as C[0, idx] = c/2, C[idx, 0] = c/2
        idx_ranges_1d_row = process_1d_ranges(idx_bounds, row=True)
        idx_ranges_1d_col = process_1d_ranges(idx_bounds, row=False)

        C_mat[idx_ranges_1d_row] = 0.5 * A_mat
        C_mat[idx_ranges_1d_col] = 0.5 * A_mat.T
        C_mat[0, 0] = float(b_vec.sum())

        canon_copy.add_psd_cone_indices(all_idx_bounds)
        canon_copy.C_obj = C_mat.tocsc()

    def _create_relu_step(self, canon_copy: "SDP", expr: LinearExpression) -> LinearExpression:
        """
        Create a ReLU step and register it with the canonicalizer.

        Parameters
        ----------
        canon_copy : SDP
            The canonicalizer copy to register the ReLU with.
        expr : LinearExpression
            The input expression for the ReLU.

        Returns
        -------
        LinearExpression
            The output (output) of the ReLU step.
        """
        step = Relu(arg=expr)
        step.update_bounds(*canon_copy.get_expression_bounds(expr))
        for leaf in step.output_leaves:
            canon_copy.add_var(leaf)
        canon_copy.canonicalize_relu(step)
        return step.output

    def set_zero_objective(self):
        """
        Set objective to zero (not yet implemented for SDP).

        Raises
        ------
        NotImplementedError
            This method is not yet implemented for the SDP canonicalizer.
        """
        raise NotImplementedError(
            "set_zero_objective is not yet implemented for the SDP canonicalizer."
        )

    def get_solution_value(self, expression: LinearExpression) -> np.ndarray:
        """
        Get the solution value for an expression (not yet implemented for SDP).

        Parameters
        ----------
        expression : LinearExpression
            The expression to evaluate.

        Returns
        -------
        np.ndarray
            The solution values.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented for the SDP canonicalizer.
        """
        raise NotImplementedError(
            "get_solution_value is not yet implemented for the SDP canonicalizer. "
            "SDP solves a relaxation and does not provide primal variable values."
        )

    def add_theory_cut(
        self,
        margin: float,
        target_expr: LinearExpression,
        lb: float | np.ndarray,
        ub: float | np.ndarray,
    ):
        """
        Add theory cut constraints (not yet implemented for SDP).

        Parameters
        ----------
        margin : float
            Relaxation margin for the bounds.
        target_expr : LinearExpression
            The expression whose bounds are to be relaxed.
        lb : float | np.ndarray
            Lower bound(s) for target_expr.
        ub : float | np.ndarray
            Upper bound(s) for target_expr.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented for the SDP canonicalizer.
        """
        raise NotImplementedError(
            "add_theory_cut is not yet implemented for the SDP canonicalizer."
        )

    def _build_sdp_idx_bounds(self) -> dict:
        """
        Build a reverse map ``sdp_idx -> (lb_scalar, ub_scalar)`` from the
        current state of ``var_idx_map``. Used by RLT cut construction to
        look up the per-index bounds without scanning leaves repeatedly.

        Returns
        -------
        dict[int, tuple[float, float]]
            Mapping from each SDP matrix index (excluding 0) to its current
            (lower, upper) bound pair.
        """
        sdp_idx_bounds = {}
        for leaf, (start, end) in self.var_idx_map.items():
            for offset, sdp_idx in enumerate(range(start, end)):
                sdp_idx_bounds[sdp_idx] = (float(leaf.lb[offset]), float(leaf.ub[offset]))
        return sdp_idx_bounds

    def _rebuild_cuts_if_dirty(self) -> None:
        """
        Rebuild the cut inequality block if any cut-affecting state has
        changed since the last rebuild.

        Triggered by ``add_var``, ``canonicalize_relu``, ``add_psd_cone_indices``,
        ``_add_equality_constraint_impl``, and ``update_bounds``. Cuts live in
        ``_cut_ineq_matrices`` / ``_cut_bineq_upperbounds``, which are
        concatenated with the regular ineq block at solve time.

        When all cut flags are False (or there are no registered inequality
        constraints) this is effectively a no-op (cut lists stay empty).
        """
        if not self._cuts_dirty:
            return

        # Option A: collapse the antichain of cones into one full moment-matrix
        # block so that ``_build_rlt_cuts`` (which keys off cone membership)
        # fires on every variable pair. Solve always runs on ``self.copy()``
        # (see ``solve_model``), so mutating the cone here is local to the copy.
        if self.full_psd_cone:
            full_cone = set(range(self.get_sdp_var_dim()))
            # Guard on the exact target, not the cone count: a single cone that is
            # not the full moment matrix (e.g. iterates merged but some auxiliary
            # variables still uncoupled) must still be unioned.
            if self.psd_cone_indices != [full_cone]:
                self.psd_cone_indices = [full_cone]
        elif self.rlt_all_pairs:
            # Middle tier: couple every pair via a cheap {0, p, q} 3x3 PSD minor
            # so the all-pairs McCormick cuts bind. add_psd_cone_indices keeps the
            # antichain, so a pair already inside a larger cone adds no new minor.
            # (Each call sets _cuts_dirty = True; that churn is harmless — it is
            # overridden by _cuts_dirty = False at the end of this method.)
            for p, q in combinations(range(1, self.get_sdp_var_dim()), 2):
                self.add_psd_cone_indices([(p, p + 1), (q, q + 1)])

        self._cut_ineq_matrices = []
        self._cut_bineq_upperbounds = []

        if self.rlt_cuts or self.rlt_all_pairs or self.triangle_cuts:
            sdp_idx_bounds = self._build_sdp_idx_bounds()

            if self.rlt_cuts or self.rlt_all_pairs:
                self._build_rlt_cuts(sdp_idx_bounds)

            if self.triangle_cuts:
                self._build_triangle_cuts()

        if self.relu_envelope_cuts:
            self._build_relu_envelope_cuts()

        # PEP cross-step interpolation cuts. Each builder self-registers the
        # minimal PSD cone for the indices it touches so the cross-step products
        # bind without ``full_psd_cone`` (the antichain subsumes a cone already
        # covered by a larger one). The cone additions set ``_cuts_dirty = True``;
        # that churn is harmless — it is overridden by ``_cuts_dirty = False`` at
        # the end of this method (same pattern as the ``rlt_all_pairs`` branch).
        if self.cross_cut:
            self._build_cross_cuts()
            self._build_box_normalcone_cuts()

        if self.subgrad_mono_cut:
            self._build_subgrad_mono_cuts()
            self._build_box_mono_cuts()

        if self._ineq_constraints and (
            self.ineq_product_cuts or self.ineq_box_cuts or self.ineq_square_cuts
        ):
            self._build_ineq_cuts()

        self._cuts_dirty = False

    def _build_ineq_cuts(self) -> None:
        r"""
        Append RLT-style strengthening cuts for every registered inequality
        constraint ``Ax + b <= 0``.

        For each row ``g_i = a_i^T x + b_i <= 0`` with homogenized coefficient
        vector ``c_i = [b_i; a_i]`` (so ``g_i = c_i^T w`` and
        ``g_i g_j = c_i^T X c_j`` is linear in the lifted ``X``), emit the cuts
        enabled by the ``ineq_*_cuts`` flags:

        - ``ineq_product_cuts``: ``g_i g_j >= 0`` for ``i < j`` (the diagonal
          ``g_i^2 >= 0`` is implied by ``X ⪰ 0`` and is skipped).
        - ``ineq_box_cuts``: ``g_i (x_p - l_p) <= 0`` and ``g_i (u_p - x_p) <= 0``
          for each variable index ``p`` appearing in the constraint.
        - ``ineq_square_cuts``: ``g_i^2 <= g_i_lower * g_i`` using a propagated
          lower bound ``g_i_lower`` on ``g_i`` (analog of the per-variable box
          product in :meth:`add_var`, generalized from ``e_i`` to ``c_i``).

        Bounds are read fresh from ``_build_sdp_idx_bounds`` / ``bound_propagation``
        each rebuild so the cuts track leaf tightening (cf. ``_build_triangle_cuts``).
        Cuts are appended to ``_cut_ineq_matrices`` / ``_cut_bineq_upperbounds`` in
        the ``<A, X> <= b`` convention; a ``>= 0`` cut is pushed as ``-sym(M) <= 0``.
        """
        sdp_idx_bounds = self._build_sdp_idx_bounds()
        n_dim = self.get_sdp_var_dim()

        for constraint, product_cuts_ok in self._ineq_constraints:
            A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(constraint.lhs)
            b_vec = np.asarray(constraint.lhs.const).flatten()
            m = A_mat.shape[0]

            # Compact index layout over {0} ∪ x for outer-product scatter.
            all_idx_bounds = [(0, 1)] + x_idx_bounds
            idx_2d = process_2d_ranges(all_idx_bounds, all_idx_bounds)
            # SDP indices of the x-block in compact order (compact pos k -> k+1).
            x_sdp_indices = [idx for (s, e) in x_idx_bounds for idx in range(s, e)]

            # c_i = [b_i, a_i] in compact ordering matching all_idx_bounds.
            c_list = [
                np.concatenate([[b_vec[i]], np.asarray(A_mat[i, :]).flatten()]) for i in range(m)
            ]

            if self.ineq_square_cuts:
                # g_i_lower <= g_i <= 0; use the propagated lower bound on g.
                g_lb, _g_ub = constraint.lhs.bound_propagation()

            for i in range(m):
                c_i = c_list[i]

                # Cut (3): g_i^2 <= g_i_lower * g_i  ->  c_i c_i^T - g_lb*(c_i e0^T) <= 0.
                if self.ineq_square_cuts:
                    Mc = np.outer(c_i, c_i)
                    Mc[:, 0] -= float(g_lb[i]) * c_i
                    M = spa.lil_matrix((n_dim, n_dim))
                    M[idx_2d] = Mc
                    self._cut_ineq_matrices.append(symmetrize(M.tocsc()))
                    self._cut_bineq_upperbounds.append(0.0)

                # Cut (1): g_i g_j >= 0 for i < j  ->  -sym(c_i c_j^T) <= 0.
                # Per-constraint ``product_cuts_ok`` skips this O(m^2) family for
                # complementarity nonnegativity sides (see ``add_product_cuts``).
                if self.ineq_product_cuts and product_cuts_ok:
                    for j in range(i + 1, m):
                        Mc = np.outer(c_i, c_list[j])
                        M = spa.lil_matrix((n_dim, n_dim))
                        M[idx_2d] = Mc
                        self._cut_ineq_matrices.append(-symmetrize(M.tocsc()))
                        self._cut_bineq_upperbounds.append(0.0)

                # Cut (2): g_i (x_p - l_p) <= 0 and g_i (u_p - x_p) <= 0.
                if self.ineq_box_cuts:
                    for k, p in enumerate(x_sdp_indices):
                        l_p, u_p = sdp_idx_bounds[p]
                        # compact position of x index p is k + 1
                        d_lower = np.zeros_like(c_i)
                        d_lower[k + 1] = 1.0
                        d_lower[0] = -l_p
                        d_upper = np.zeros_like(c_i)
                        d_upper[k + 1] = -1.0
                        d_upper[0] = u_p
                        for d in (d_lower, d_upper):
                            M = spa.lil_matrix((n_dim, n_dim))
                            M[idx_2d] = np.outer(c_i, d)
                            self._cut_ineq_matrices.append(symmetrize(M.tocsc()))
                            self._cut_bineq_upperbounds.append(0.0)

    def _build_rlt_cuts(self, sdp_idx_bounds: dict) -> None:
        """
        Append RLT cuts for off-diagonal cone-mate pairs.

        For every pair ``(p, q)`` of distinct, nonzero SDP indices that
        co-exist in some entry of ``psd_cone_indices``, emit the 4
        McCormick inequalities (paper §4.5 eq 18). Each pair is processed
        at most once even if it appears in multiple cones.

        Parameters
        ----------
        sdp_idx_bounds : dict[int, tuple[float, float]]
            Mapping from SDP index to its current ``(lb, ub)`` pair.
        """
        added_pairs = set()
        for cone_set in self.psd_cone_indices:
            sorted_indices = sorted(i for i in cone_set if i != 0)
            for p, q in combinations(sorted_indices, 2):
                if (p, q) in added_pairs:
                    continue
                added_pairs.add((p, q))
                l_p, u_p = sdp_idx_bounds[p]
                l_q, u_q = sdp_idx_bounds[q]
                self._build_rlt_cuts_for_pair(p, q, l_p, u_p, l_q, u_q)

    def _build_rlt_cuts_for_pair(
        self,
        p: int,
        q: int,
        l_p: float,
        u_p: float,
        l_q: float,
        u_q: float,
    ) -> None:
        """
        Append the 4 McCormick RLT inequalities for an off-diagonal pair.

        Each unsymmetrized matrix ``M`` has at most 3 nonzero entries:
        the off-diagonal ``[p, q]`` slot and the two linear slots
        ``[0, p]`` and ``[0, q]``. After ``symmetrize`` and a trace inner
        product against the symmetric lifted variable ``X``, the
        coefficient on ``X[p, q]`` is whatever was written at ``[p, q]``
        and the coefficients on ``X[0, p]`` / ``X[0, q]`` are whatever
        was written at those slots.

        Parameters
        ----------
        p, q : int
            Distinct SDP indices, both ``> 0``.
        l_p, u_p, l_q, u_q : float
            Lower/upper bounds at indices ``p`` and ``q``.
        """
        n = self.get_sdp_var_dim()

        # Cut 1: (X_p - l_p)(X_q - l_q) >= 0
        #   -> -X[p,q] + l_q X[0,p] + l_p X[0,q] <= l_p l_q
        M1 = spa.lil_matrix((n, n))
        M1[p, q] = -1.0
        M1[0, p] = l_q
        M1[0, q] = l_p
        self._cut_ineq_matrices.append(symmetrize(M1.tocsc()))
        self._cut_bineq_upperbounds.append(l_p * l_q)

        # Cut 2: (u_p - X_p)(u_q - X_q) >= 0
        #   -> -X[p,q] + u_q X[0,p] + u_p X[0,q] <= u_p u_q
        M2 = spa.lil_matrix((n, n))
        M2[p, q] = -1.0
        M2[0, p] = u_q
        M2[0, q] = u_p
        self._cut_ineq_matrices.append(symmetrize(M2.tocsc()))
        self._cut_bineq_upperbounds.append(u_p * u_q)

        # Cut 3: (X_p - l_p)(u_q - X_q) >= 0
        #   -> X[p,q] - u_q X[0,p] - l_p X[0,q] <= -l_p u_q
        M3 = spa.lil_matrix((n, n))
        M3[p, q] = 1.0
        M3[0, p] = -u_q
        M3[0, q] = -l_p
        self._cut_ineq_matrices.append(symmetrize(M3.tocsc()))
        self._cut_bineq_upperbounds.append(-l_p * u_q)

        # Cut 4: (u_p - X_p)(X_q - l_q) >= 0
        #   -> X[p,q] - l_q X[0,p] - u_p X[0,q] <= -u_p l_q
        M4 = spa.lil_matrix((n, n))
        M4[p, q] = 1.0
        M4[0, p] = -l_q
        M4[0, q] = -u_p
        self._cut_ineq_matrices.append(symmetrize(M4.tocsc()))
        self._cut_bineq_upperbounds.append(-u_p * l_q)

    @staticmethod
    def _relu_arg_bounds(step: Relu) -> tuple[np.ndarray, np.ndarray]:
        """
        Bounds on a ReLU step's input used by the ReLU cut builders.

        Intersects the step's cached input bounds ``step.lb``/``step.ub`` (which
        carry any OBBT tightening applied in ``_process_primitive``) with a fresh
        ``step.arg.bound_propagation()`` (which reflects leaf tightening since the
        step was canonicalized). Both are valid, so the intersection is too; tiny
        solver-noise crossings are repaired.
        """
        prop_lb, prop_ub = step.arg.bound_propagation()
        d = step.arg.ndim
        lb = np.maximum(np.broadcast_to(np.asarray(step.lb, dtype=float), d), prop_lb)
        ub = np.minimum(np.broadcast_to(np.asarray(step.ub, dtype=float), d), prop_ub)
        return repair_crossed_bounds(lb, ub)

    def _build_triangle_cuts(self) -> None:
        """
        Append lifted triangle cuts for every registered ReLU step.

        For each step, iterate over output index pairs ``(i, j)``: emit one
        cut whenever row ``i`` has mixed bounds (``arg_lb[i] < 0 <
        arg_ub[i]``) and column ``j`` has finite upper bound. Bounds come
        from :meth:`_relu_arg_bounds`: the OBBT-tightened ``step.lb``/``step.ub``
        intersected with a fresh ``step.arg.bound_propagation()`` (which
        reflects any leaf tightening since ``canonicalize_relu`` ran). See
        paper §4.5 eq 25/26 for the derivation.
        """
        for step in self._relu_steps:
            arg = step.arg
            arg_lb, arg_ub = self._relu_arg_bounds(step)
            A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(arg)
            b_vec = np.asarray(arg.const)

            # Map each arg row to its position in the fresh y-leaf. For full
            # relu this is the identity dict (proj_indices spans arg.ndim);
            # for partial relu it maps only projected arg rows and skips the
            # rest (handled by the ``if i not in rhs_to_y_pos`` guard below).
            y_leaf = step.output_leaves[0]
            rhs_to_y_pos = {i: j for j, i in enumerate(sorted(step.proj_indices))}
            y_idx_bounds = [self.var_idx_map[y_leaf]]

            d = arg.ndim
            for i in range(d):
                if i not in rhs_to_y_pos:
                    continue
                if arg_lb[i] >= 0 or arg_ub[i] <= 0:
                    continue
                for j in range(d):
                    if not np.isfinite(arg_ub[j]):
                        continue
                    self._build_triangle_cut_for_pair(
                        i,
                        j,
                        A_mat,
                        b_vec,
                        x_idx_bounds,
                        y_idx_bounds,
                        arg_lb,
                        arg_ub,
                        y_pos=rhs_to_y_pos[i],
                    )

    def _build_triangle_cut_for_pair(
        self,
        i: int,
        j: int,
        A_mat: np.ndarray,
        b_vec: np.ndarray,
        x_idx_bounds: list,
        y_idx_bounds: list,
        arg_lb: np.ndarray,
        arg_ub: np.ndarray,
        y_pos: int | None = None,
    ) -> None:
        """
        Append the lifted triangle cut for output pair ``(i, j)`` of a ReLU
        step.

        Derivation:
        ``(E_ii (a_i^T x + b_i) + c_i − y_i)(u_j − a_j^T x − b_j) ≥ 0``
        where ``E_ii = u_i / (u_i − l_i)``, ``c_i = −E_ii · l_i``, and
        ``(l_i, u_i, u_j)`` are taken from ``arg_lb`` / ``arg_ub``, the bounds
        of the ReLU's input expression from :meth:`_relu_arg_bounds` (OBBT
        bounds intersected with fresh propagation) — see ``_build_triangle_cuts``.

        Expanding gives 9 product groups that land in the lifted-matrix
        slots:
        - xx-block: coefficient ``−E_ii · outer(a_i, a_j)``
        - y_i ⊗ x row: coefficient ``+a_j``
        - y_i linear (``[y_i_idx, 0]``): ``(b_j − u_j)``
        - x linear (``[0, x]``): ``E_ii u_j a_i − E_ii b_j a_i − E_ii b_i a_j
          − c_i a_j``
        - constant (``[0, 0]``): ``E_ii b_i u_j − E_ii b_i b_j + c_i u_j
          − c_i b_j``

        The cut is ``tr(symmetrize(N) X) ≥ 0``, stored as
        ``tr(−symmetrize(N) X) ≤ 0``.
        """
        n_dim = self.get_sdp_var_dim()

        lb_i = arg_lb[i]
        ub_i = arg_ub[i]
        ub_j = arg_ub[j]
        E_ii = ub_i / (ub_i - lb_i)
        c_i = -E_ii * lb_i

        a_i = np.asarray(A_mat[i, :]).flatten()
        a_j = np.asarray(A_mat[j, :]).flatten()
        b_i = float(b_vec[i])
        b_j = float(b_vec[j])

        y_start, _y_end = y_idx_bounds[0]
        # ``y_pos`` is the row of the (possibly smaller) fresh leaf that
        # corresponds to arg row ``i``. For full-projection it equals ``i``.
        y_i_idx = y_start + (y_pos if y_pos is not None else i)

        N = spa.lil_matrix((n_dim, n_dim))

        # xx-block: -E_ii * outer(a_i, a_j) at process_2d_ranges(x, x)
        xx_idx = process_2d_ranges(x_idx_bounds, x_idx_bounds)
        N[xx_idx] = -E_ii * np.outer(a_i, a_j)

        # y_i ⊗ x row: +a_j at row y_i_idx, cols x-concat
        yi_x_idx = process_2d_ranges([(y_i_idx, y_i_idx + 1)], x_idx_bounds)
        N[yi_x_idx] = a_j

        # y_i linear: (b_j - ub_j) at [y_i_idx, 0]
        N[y_i_idx, 0] = b_j - ub_j

        # x linear at row 0
        lin = E_ii * ub_j * a_i - E_ii * b_j * a_i - E_ii * b_i * a_j - c_i * a_j
        x_row0_idx = process_1d_ranges(x_idx_bounds, row=True)
        N[x_row0_idx] = lin

        # Constant at [0, 0]
        const = E_ii * b_i * ub_j - E_ii * b_i * b_j + c_i * ub_j - c_i * b_j
        N[0, 0] = const

        # Cut: tr(symmetrize(N) X) >= 0  =>  push -symmetrize(N) with ub = 0
        N_sym = symmetrize(N.tocsc())
        self._cut_ineq_matrices.append(-N_sym)
        self._cut_bineq_upperbounds.append(0.0)

    def _build_relu_envelope_cuts(self) -> None:
        r"""
        Append the linear ReLU convex-hull inequalities on the first-moment
        (vector) variables for every registered ReLU step.

        Unlike the *lifted* triangle cut (a quadratic-in-X product), these are
        linear in the first moments ``X[0, .]`` and so restrict the relaxation
        in a way the lifted equality ``y^2 = (a^T x + b) y`` plus PSD does not.
        For each projected arg row ``i`` with ``hat_x_i = a_i^T x + b_i``:

        - Lower envelope ``y_i >= hat_x_i`` (valid for every row, since
          ``ReLU(t) >= t``).
        - Upper chord ``y_i <= E_ii (hat_x_i - l_i)`` with
          ``E_ii = u_i / (u_i - l_i)``, emitted only on mixed-bound rows
          (``l_i < 0 < u_i``, finite ``u_i``) where the secant of ``max(0, .)``
          through ``(l_i, 0)`` and ``(u_i, u_i)`` dominates the true map. On a
          passthrough row (``l_i >= 0``) this chord would cut the true point, so
          it is skipped (gated exactly like the triangle cut).

        ``y_i >= 0`` is intentionally omitted: the y-leaf box bound (lb >= 0)
        together with the PSD ``2x2`` minor already implies it.

        Bounds come from :meth:`_relu_arg_bounds` (OBBT-tightened ``step.lb``/
        ``step.ub`` intersected with fresh propagation; parity with
        ``_build_triangle_cuts``).
        """
        n_dim = self.get_sdp_var_dim()
        for step in self._relu_steps:
            arg = step.arg
            arg_lb, arg_ub = self._relu_arg_bounds(step)
            A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(arg)
            b_vec = np.asarray(arg.const)

            y_leaf = step.output_leaves[0]
            rhs_to_y_pos = {i: j for j, i in enumerate(sorted(step.proj_indices))}
            y_start, _y_end = self.var_idx_map[y_leaf]
            x_row0_idx = process_1d_ranges(x_idx_bounds, row=True)

            for i in range(arg.ndim):
                if i not in rhs_to_y_pos:
                    continue
                a_i = np.asarray(A_mat[i, :]).flatten()
                b_i = float(b_vec[i])
                y_i_idx = y_start + rhs_to_y_pos[i]

                # Lower envelope: y_i - (a_i^T x + b_i) >= 0.
                N = spa.lil_matrix((n_dim, n_dim))
                N[y_i_idx, 0] = 1.0
                N[x_row0_idx] = -a_i
                N[0, 0] = -b_i
                self._cut_ineq_matrices.append(-symmetrize(N.tocsc()))
                self._cut_bineq_upperbounds.append(0.0)

                # Upper chord (mixed rows only): E_ii (a_i^T x + b_i - l_i) - y_i >= 0.
                lb_i = arg_lb[i]
                ub_i = arg_ub[i]
                if lb_i >= 0 or ub_i <= 0 or not np.isfinite(ub_i):
                    continue
                E_ii = ub_i / (ub_i - lb_i)
                N = spa.lil_matrix((n_dim, n_dim))
                N[x_row0_idx] = E_ii * a_i
                N[0, 0] = E_ii * b_i - E_ii * lb_i
                N[y_i_idx, 0] = -1.0
                self._cut_ineq_matrices.append(-symmetrize(N.tocsc()))
                self._cut_bineq_upperbounds.append(0.0)

    def _relu_output_components(self) -> list[dict]:
        """
        Flatten all registered ReLU steps into per-output-component records.

        Each record describes one projected ReLU output ``z = relu(v)`` at a
        single coordinate:

        - ``y_idx``  : its SDP matrix index (used for the ``z`` factor),
        - ``a``      : coefficient row of the affine input ``v = a^T x + beta``,
        - ``beta``   : constant of ``v`` at this row,
        - ``x_bounds``: the input leaves' SDP index ranges (parallel to ``a``).

        Used by :meth:`_build_cross_cuts`. The arg coefficients are read via
        ``linexpr_to_Amat_and_bounds`` once per step.
        """
        comps = []
        for step in self._relu_steps:
            A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(step.arg)
            b_vec = np.asarray(step.arg.const)
            y_leaf = step.output_leaves[0]
            y_start, _y_end = self.var_idx_map[y_leaf]
            rhs_to_y_pos = {i: j for j, i in enumerate(sorted(step.proj_indices))}
            for rb, y_pos in rhs_to_y_pos.items():
                comps.append(
                    {
                        "y_idx": y_start + y_pos,
                        "a": np.asarray(A_mat[rb, :]).flatten(),
                        "beta": float(b_vec[rb]),
                        "x_bounds": x_idx_bounds,
                    }
                )
        return comps

    def _build_cross_cuts(self) -> None:
        r"""
        Append the PEP "cross" interpolation cuts ``z_a r_b <= 0``.

        For any two ReLU output components ``a`` and ``b`` (across all registered
        ReLU steps), ``z_a >= 0`` and the residual ``r_b = v_b - z_b <= 0``, so
        ``z_a r_b <= 0`` is valid (PDF eq 3; for ReLU = orthant projection it is
        the indicator-function specialization of the subgradient inequality). It
        couples *different* steps, which the per-step encoding does not. With
        ``r_b = v_b - z_b`` and ``v_b = a_b^T x + beta_b`` this is linear in ``X``:

        ``z_a r_b = sum_p a_b[p] X[ya, x_p] + beta_b X[0, ya] - X[ya, yb] <= 0``.

        The diagonal ``a == b`` (``ya == yb``) is the complementarity identity
        ``z_a r_a = 0`` already encoded exactly by :meth:`canonicalize_relu`, so it
        is skipped. Each cut registers the minimal PSD cone ``{0, ya, yb} U x_b``
        so its second-moment entries bind without ``full_psd_cone``.

        TODO(scaling): this emits O(N^2) cuts and cones, where ``N`` is the total
        number of scalar ReLU outputs across the unrolled network, so cost grows
        quadratically with depth. Fine at current scales (a few seconds for the
        tested instances); large networks will want a budget/cap or per-step
        batching of the (a, b) pairs.
        """
        n_dim = self.get_sdp_var_dim()
        comps = self._relu_output_components()

        for comp_a in comps:
            ya_idx = comp_a["y_idx"]
            for comp_b in comps:
                yb_idx = comp_b["y_idx"]
                if ya_idx == yb_idx:
                    # Complementarity diagonal z_a r_a = 0 (already exact); skip.
                    continue

                a_b = comp_b["a"]
                beta_b = comp_b["beta"]
                xb_idx_bounds = comp_b["x_bounds"]

                # Couple {0, ya, yb} U x_b so X[ya, yb] and X[ya, x_p] are
                # constrained by a PSD minor (else the cut is vacuous).
                self.add_psd_cone_indices(
                    [(ya_idx, ya_idx + 1), (yb_idx, yb_idx + 1)] + xb_idx_bounds
                )

                M = spa.lil_matrix((n_dim, n_dim))
                # z_a * (a_b^T x): single y-row times the x-concat columns.
                M[process_2d_ranges([(ya_idx, ya_idx + 1)], xb_idx_bounds)] = a_b
                # beta_b * z_a (linear).
                M[ya_idx, 0] = beta_b
                # - z_a z_b (second moment).
                M[ya_idx, yb_idx] = -1.0

                # Cut is already <= 0: push +symmetrize(M) with ub 0 (no negation).
                self._cut_ineq_matrices.append(symmetrize(M.tocsc()))
                self._cut_bineq_upperbounds.append(0.0)

    def _cross_innerprod_block(self, expr1: LinearExpression, expr2: LinearExpression):
        r"""
        Build the unsymmetrized lifted block for the inner product
        ``<expr1, expr2> = <E w + e, F w + f>``.

        Expanding ``(E w + e)^T (F w + f) = w^T E^T F w + e^T F w + f^T E w +
        e^T f`` and reading each monomial off the lifted ``X`` (``X[p, q]`` the
        second moment, ``X[0, p]`` the first moment, ``X[0, 0] = 1``):

        - ``E^T F``    -> ``M[idxE, idxF]``  (the second-moment block),
        - ``f^T E``    -> ``X[0, idxE]``     (first-moment row),
        - ``e^T F``    -> ``X[0, idxF]``     (first-moment row),
        - ``e^T f``    -> ``M[0, 0]``.

        Generalizes :meth:`_apply_two_norm_objective` (the self case
        ``<Ax+b, Ax+b>``) to two *different* expressions. The two first-moment
        contributions are accumulated **additively** over the union of their SDP
        indices and scattered once, so a leaf shared by ``expr1`` and ``expr2``
        does not get one write clobbered by the other. The caller symmetrizes.

        Returns
        -------
        scipy.sparse.lil_matrix
            The unsymmetrized block ``M`` with ``tr(M X) = <expr1, expr2>``.
        """
        n_dim = self.get_sdp_var_dim()
        E_mat, idxE = self.linexpr_to_Amat_and_bounds(expr1)
        F_mat, idxF = self.linexpr_to_Amat_and_bounds(expr2)
        e = np.asarray(expr1.const).reshape(-1, 1)
        f = np.asarray(expr2.const).reshape(-1, 1)

        M = spa.lil_matrix((n_dim, n_dim))

        # Second-moment block E^T F over (idxE rows, idxF cols).
        M[process_2d_ranges(idxE, idxF)] = E_mat.T @ F_mat

        # First moments: accumulate f^T E (on idxE) and e^T F (on idxF) additively
        # into a single dense row over the union of SDP indices, then scatter once.
        # asarray guards against np.matrix (linexpr_to_Amat_and_bounds may return a
        # matrix), whose .flatten() stays 2-D and breaks integer indexing below.
        fTE = np.asarray(f.T @ E_mat).flatten()  # length = sum of idxE block widths
        eTF = np.asarray(e.T @ F_mat).flatten()  # length = sum of idxF block widths
        row0 = {}
        for bounds, vec in ((idxE, fTE), (idxF, eTF)):
            offset = 0
            for s, end in bounds:
                for col in range(s, end):
                    row0[col] = row0.get(col, 0.0) + vec[offset]
                    offset += 1
        for col, val in row0.items():
            M[0, col] = val

        M[0, 0] = float(np.dot(np.asarray(e).flatten(), np.asarray(f).flatten()))
        return M

    def _build_subgrad_mono_cuts(self) -> None:
        r"""
        Append the PEP subgradient-monotonicity cuts
        ``<s^i/tau_i - s^j/tau_j, z^i> >= 0`` over ordered prox pairs.

        For prox steps ``i, j`` with output ``z``, input ``v``, threshold ``tau``
        and subgradient ``s = v - z`` (PDF eq 4, normalized by ``tau`` so the
        subgradient lives in the subdifferential of the same homogeneous
        function). Expanding,

        ``<s^i/tau_i - s^j/tau_j, z^i>
            = (1/tau_i)[<v^i, z^i> - <z^i, z^i>]
            - (1/tau_j)[<v^j, z^i> - <z^j, z^i>]``,

        each term a lifted inner product built by :meth:`_cross_innerprod_block`.
        On the orthant (``tau = 1``, ``z`` a ReLU output) this reduces to the
        cross cut; on the soft-threshold (``tau = lambda``, ``z`` composite) it is
        a distinct coupling. Each cut registers a PSD cone over the indices of
        ``z^i, v^i, z^j, v^j`` so the cross-step moments bind (this is the smallest
        cone that couples the involved leaves, but for composite proxes it spans
        several index ranges, so it is not as small as the cross cut's cone).

        The cut is valid only when ``partial f_j(0) subseteq partial f_i(0)``, so
        pairs are restricted to proxes of the same function (``func``) and the
        same dimension. Skipping mismatched-dimension orthant pairs loses nothing:
        there the cut is a sum of per-coordinate cross cuts (``cross_cut``).
        """
        prox = self._prox_steps
        for i, pi in enumerate(prox):
            zi, vi, tau_i = pi["z"], pi["v"], pi["tau"]
            if tau_i < 1e-9:
                continue
            for j, pj in enumerate(prox):
                if i == j:
                    continue
                zj, vj, tau_j = pj["z"], pj["v"], pj["tau"]
                if tau_j < 1e-9:
                    continue
                # Same sublinear function and dimension only (else the cut is invalid).
                if pi["func"] != pj["func"] or zi.ndim != zj.ndim:
                    continue

                # Couple all involved leaves so the cross-step moments bind.
                _, idx_zi = self.linexpr_to_Amat_and_bounds(zi)
                _, idx_vi = self.linexpr_to_Amat_and_bounds(vi)
                _, idx_zj = self.linexpr_to_Amat_and_bounds(zj)
                _, idx_vj = self.linexpr_to_Amat_and_bounds(vj)
                self.add_psd_cone_indices(idx_zi + idx_vi + idx_zj + idx_vj)

                # Accumulate the four (unsymmetrized) inner-product blocks.
                M = (
                    (1.0 / tau_i) * self._cross_innerprod_block(vi, zi)
                    - (1.0 / tau_i) * self._cross_innerprod_block(zi, zi)
                    - (1.0 / tau_j) * self._cross_innerprod_block(vj, zi)
                    + (1.0 / tau_j) * self._cross_innerprod_block(zj, zi)
                )

                # Cut is >= 0: push -symmetrize(M) with ub 0.
                self._cut_ineq_matrices.append(-symmetrize(M.tocsc()))
                self._cut_bineq_upperbounds.append(0.0)

    @staticmethod
    def _box_bound_scalar(bound, j):
        """Scalar j-th box bound: a float for constant arrays, or the j-th slice
        (a length-1 LinearExpression) when the bound is parametric."""
        if isinstance(bound, LinearExpression):
            return bound[j]
        return float(np.atleast_1d(bound)[j])

    @staticmethod
    def _box_bound_eq(x, y) -> bool:
        """Whether two box bounds are 'the same' for cross-step pairing: identity
        for parametric (LinearExpression) bounds, value-equality for constants."""
        if isinstance(x, LinearExpression) or isinstance(y, LinearExpression):
            return x is y
        return np.array_equal(np.atleast_1d(x), np.atleast_1d(y))

    def _same_box(self, a: dict, b: dict) -> bool:
        """Two box-prox steps project onto the same box ``[l, u]``."""
        return self._box_bound_eq(a["l"], b["l"]) and self._box_bound_eq(a["u"], b["u"])

    def _emit_scalar_product_cut(self, e1: LinearExpression, e2: LinearExpression, sense: str):
        r"""
        Emit the per-component cut ``e1 . e2  {<=,>=}  0`` where ``e1``, ``e2`` are
        scalar (length-1) affine expressions. ``tr(M X) = e1 . e2`` via
        :meth:`_cross_innerprod_block` (inner product of 1-vectors = the scalar
        product). Registers the minimal PSD cone over the involved leaves so the
        bilinear term binds, then pushes ``+sym`` (for ``<=``) or ``-sym`` (for
        ``>=``) with ub 0 — the standard convention.
        """
        _, idx1 = self.linexpr_to_Amat_and_bounds(e1)
        _, idx2 = self.linexpr_to_Amat_and_bounds(e2)
        self.add_psd_cone_indices(idx1 + idx2)
        M = self._cross_innerprod_block(e1, e2)
        sym = symmetrize(M.tocsc())
        self._cut_ineq_matrices.append(sym if sense == "<=" else -sym)
        self._cut_bineq_upperbounds.append(0.0)

    def _build_box_normalcone_cuts(self) -> None:
        r"""
        Append the PEP box-projection normal-cone cuts for satlin steps (the box
        analog of the ReLU cross cut; emitted under ``cross_cut``).

        For a box projection ``z_j = clip(v_j, l_j, u_j)`` with residual
        ``r_j = v_j - z_j`` (``r_j in N_{[l_j,u_j]}(z_j)``):

        - **C1/C2 (within step):** ``r_j (z_j - l_j) >= 0`` and ``r_j (z_j - u_j) >= 0``
          (the residual vs the two box corners).
        - **C3 (cross step, same coord, same box):** ``r_j^b (z_j^a - z_j^b) <= 0``
          for any other box-projection output ``z^a`` onto the same box (since
          ``z_j^a in [l_j, u_j]`` and ``r_j^b`` is in the normal cone at ``z_j^b``).

        Cuts are **per-coordinate** (the box ``z``/``r`` are not sign-definite, so
        cross-coordinate products are NOT valid — unlike the ReLU cone cross cut).
        """
        boxes = self._box_prox_steps
        for b_idx, b in enumerate(boxes):
            zb, vb = b["z"], b["v"]
            rb = vb - zb
            n = zb.ndim
            for j in range(n):
                rbj = rb[j]
                lj = self._box_bound_scalar(b["l"], j)
                uj = self._box_bound_scalar(b["u"], j)
                # C1: r_j (z_j - l_j) >= 0 ; C2: r_j (z_j - u_j) >= 0
                self._emit_scalar_product_cut(rbj, zb[j] - lj, ">=")
                self._emit_scalar_product_cut(rbj, zb[j] - uj, ">=")

            # C3: r_j^b (z_j^a - z_j^b) <= 0 over ordered same-box pairs.
            for a_idx, a in enumerate(boxes):
                if a_idx == b_idx or not self._same_box(a, b):
                    continue
                za = a["z"]
                for j in range(n):
                    self._emit_scalar_product_cut(rb[j], za[j] - zb[j], "<=")

    def _build_box_mono_cuts(self) -> None:
        r"""
        Append the PEP box-projection firm-nonexpansiveness / monotonicity cuts for
        satlin steps (the box analog of the subgradient-monotonicity cut; emitted
        under ``subgrad_mono_cut``).

        For two box projections onto the **same** box, per coordinate ``j``:
        ``(z_j^a - z_j^b)(r_j^a - r_j^b) >= 0`` with ``r = v - z`` — equivalently
        the firm-nonexpansiveness inequality ``<z^a-z^b, v^a-v^b> >= ||z^a-z^b||^2``
        read per coordinate. Valid because the scalar clip and its residual are both
        monotone nondecreasing. Unordered pairs (the cut is symmetric in ``a, b``).
        """
        boxes = self._box_prox_steps
        for a_idx, b_idx in combinations(range(len(boxes)), 2):
            a, b = boxes[a_idx], boxes[b_idx]
            if not self._same_box(a, b):
                continue
            za, va = a["z"], a["v"]
            zb, vb = b["z"], b["v"]
            ra, rb = va - za, vb - zb
            for j in range(za.ndim):
                self._emit_scalar_product_cut(za[j] - zb[j], ra[j] - rb[j], ">=")

    def solve_model(self, steps=None, convex_hull_cuts=False, **solver_kwargs):
        """Solve the SDP problem using the configured solver.

        Creates a copy of the canonicalizer, applies the objective to the copy,
        solves it, and keeps a reference for future use. This prevents objective
        constraints from accumulating across multiple solve calls.

        Parameters
        ----------
        steps : list, optional
            Unused by SDP, accepted for API compatibility.
        convex_hull_cuts : bool, optional
            Unused by SDP, accepted for API compatibility.
        **solver_kwargs
            Native solver options forwarded verbatim to the backend
            (e.g. ``max_iter=500``, ``tol_gap_abs=1e-9`` for
            Clarabel/cvxpy). They override construction-time ``params``
            for this call. A ``ValueError`` is raised if the backend
            rejects an option.

        Returns
        -------
        float
            The objective value.
        """
        # Create copy to apply objective without modifying original
        self._solved_copy = self.copy()

        # Apply objective to the copy
        if self._objective_spec is not None:
            obj_type = self._objective_spec[0]
            match obj_type:
                case "two_norm":
                    _, expr_list = self._objective_spec
                    self._apply_two_norm_objective(self._solved_copy, expr_list)
                case "one_norm":
                    _, expr_list = self._objective_spec
                    self._apply_one_norm_objective(self._solved_copy, expr_list)
                case "infinity_norm":
                    _, expr_list = self._objective_spec
                    self._apply_infinity_norm_objective(self._solved_copy, expr_list)

        # Solve using the copy's data
        return self._solved_copy._solve_internal(**solver_kwargs)

    def _solve_internal(self, **solver_kwargs):
        """
        Actually solve using this canonicalizer's data.

        This is the internal solve method called on the copy after
        the objective has been applied.

        Parameters
        ----------
        **solver_kwargs
            Native solver options forwarded verbatim to the backend.

        Returns
        -------
        float
            The objective value.
        """
        match self.solver_name:
            case "clarabel":
                return self._solve_via_clarabel_internal(**solver_kwargs)
            case "cvxpy":
                return self._solve_via_cvxpy_internal(**solver_kwargs)
            case "cuclarabel":
                return self._solve_via_cuclarabel_internal(**solver_kwargs)
            case _:
                raise ValueError(f"Unknown SDP solver: {self.solver_name}")

    def _ensure_objective_applied(self):
        """
        Ensure the objective has been applied to this canonicalizer.

        If the objective specification is stored but not yet applied,
        apply it now. This is used for backward compatibility with
        direct solver calls.
        """
        if self._objective_spec is not None and self.C_obj is None:
            obj_type = self._objective_spec[0]
            match obj_type:
                case "two_norm":
                    _, expr_list = self._objective_spec
                    self._apply_two_norm_objective(self, expr_list)
                case "one_norm":
                    _, expr_list = self._objective_spec
                    self._apply_one_norm_objective(self, expr_list)
                case "infinity_norm":
                    _, expr_list = self._objective_spec
                    self._apply_infinity_norm_objective(self, expr_list)

    def solve_via_cvxpy(self, **solver_kwargs):
        """
        Solve the SDP using the CVXPY modeling interface.

        Constructs a CVXPY problem from the stored constraint matrices
        and solves it using the Clarabel backend via CVXPY.

        Parameters
        ----------
        **solver_kwargs
            Native solver options forwarded verbatim to ``Problem.solve``.

        Returns
        -------
        float
            The optimal objective value.
        """
        self._ensure_objective_applied()
        return self._solve_via_cvxpy_internal(**solver_kwargs)

    def _solve_via_cvxpy_internal(self, **solver_kwargs):
        """Internal CVXPY solve without checking objective."""
        self._rebuild_cuts_if_dirty()
        Aineq_all = self.Aineq_matrices + self._cut_ineq_matrices
        bineq_all = self.bineq_upperbounds + self._cut_bineq_upperbounds

        dim = self.get_sdp_var_dim()
        X = cp.Variable((dim, dim), symmetric=True)
        obj = cp.Maximize(cp.trace(self.C_obj @ X))
        constraints = []
        for A, b in zip(self.Aeq_matrices, self.beq_consts):
            constraints += [
                cp.trace(A @ X) == b,
            ]
        for A, b in zip(Aineq_all, bineq_all):
            constraints += [
                cp.trace(A @ X) <= b,
            ]
        # Add PSD constraints for each structured cone (matching Clarabel path)
        for cone_indices in self.psd_cone_indices:
            idx = sorted(cone_indices)
            constraints += [X[np.ix_(idx, idx)] >> 0]
        self.last_problem = prob = cp.Problem(obj, constraints)
        # `solver` is fixed to CLARABEL below (it is a named arg of cvxpy's
        # Problem.solve, so a user `solver=` kwarg would collide there);
        # solve-time options override construction-time params.
        merged = {**(self.solver_params or {}), **solver_kwargs}
        try:
            res = prob.solve(solver=cp.CLARABEL, **merged)
        except TypeError as e:
            raise ValueError(f"Invalid cvxpy/Clarabel solver option in {merged}: {e}") from e
        except cp.error.SolverError as e:
            raise RuntimeError(f"CVXPY/Clarabel failed to solve the SDP: {e}") from e
        if (
            prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            or res is None
            or not np.isfinite(res)
        ):
            raise RuntimeError(f"CVXPY terminated with {prob.status}; no accepted SDP bound.")
        if prob.status == cp.OPTIMAL_INACCURATE:
            warnings.warn(
                "CVXPY terminated optimal_inaccurate; the SDP bound holds only to the "
                "solver's reduced tolerances.",
                stacklevel=2,
            )
        # For this linear maximization SDP, the dual objective is b^T lambda.
        # PSD constraints have zero right-hand sides. Keep the Problem for
        # status, primal, and dual inspection; acceptance is numerical.
        rhs = list(self.beq_consts) + list(bineq_all)
        if any(c.dual_value is None for c in constraints[: len(rhs)]):
            raise RuntimeError("CVXPY returned no SDP dual solution.")
        bound = float(sum(b * float(c.dual_value) for b, c in zip(rhs, constraints)))
        if not np.isfinite(bound):
            raise RuntimeError("CVXPY returned a nonfinite SDP dual objective.")
        return bound

    def solve_via_clarabel(self, **solver_kwargs):
        """
        Solve the SDP using the native Clarabel interface.

        Directly invokes the Clarabel solver without going through CVXPY,
        which can be more efficient for large problems.

        Parameters
        ----------
        **solver_kwargs
            Native Clarabel settings forwarded verbatim to the solver.

        Returns
        -------
        float
            The optimal objective value.
        """
        self._ensure_objective_applied()
        return self._solve_via_clarabel_internal(**solver_kwargs)

    def _solve_via_clarabel_internal(self, **solver_kwargs):
        """Internal Clarabel solve without checking objective."""
        self._rebuild_cuts_if_dirty()
        Aineq_all = self.Aineq_matrices + self._cut_ineq_matrices
        bineq_all = self.bineq_upperbounds + self._cut_bineq_upperbounds

        from fomtools.solvers.clarabel import ClarabelSolver

        solver = ClarabelSolver(solver_params=self.solver_params)
        solver.setup_clarabel_data(
            self.get_sdp_var_dim(),
            self.C_obj,
            self.Aeq_matrices,
            self.beq_consts,
            Aineq_all,
            bineq_all,
            self.psd_cone_indices,
        )
        self.last_solver = solver
        return solver.solve(**solver_kwargs)

    def solve_via_cuclarabel(self, **solver_kwargs):
        """
        Solve the SDP using GPU-accelerated Clarabel.

        Uses the CUDA-enabled Clarabel solver for GPU acceleration.
        Requires CuPy and Julia with CUDA.jl and Clarabel.jl.

        Parameters
        ----------
        **solver_kwargs
            Native Clarabel settings forwarded verbatim to the solver.

        Returns
        -------
        float
            The optimal objective value.

        Raises
        ------
        ValueError
            If CUDA dependencies (cupy) are not available.
        """
        self._ensure_objective_applied()
        return self._solve_via_cuclarabel_internal(**solver_kwargs)

    def _solve_via_cuclarabel_internal(self, **solver_kwargs):
        """Internal CuClarabel solve without checking objective."""
        self._rebuild_cuts_if_dirty()
        Aineq_all = self.Aineq_matrices + self._cut_ineq_matrices
        bineq_all = self.bineq_upperbounds + self._cut_bineq_upperbounds

        from fomtools.solvers.cuclarabel import CuClarabelSolver

        solver = CuClarabelSolver(solver_params=self.solver_params)
        solver.setup_clarabel_data(
            self.get_sdp_var_dim(),
            self.C_obj,
            self.Aeq_matrices,
            self.beq_consts,
            Aineq_all,
            bineq_all,
            self.psd_cone_indices,
        )
        self.last_solver = solver
        return solver.solve(**solver_kwargs)


def symmetrize(mat):
    """
    Symmetrize a sparse matrix.

    Parameters
    ----------
    mat : scipy.sparse matrix
        The matrix to symmetrize.

    Returns
    -------
    scipy.sparse matrix
        The symmetric matrix (mat + mat.T) / 2.
    """
    return (mat + mat.T) / 2


def range_to_indices(idx_ranges):
    """
    Convert a list of (start, end) range tuples to a set of indices.

    Parameters
    ----------
    idx_ranges : list[tuple[int, int]]
        List of (start, end) pairs defining half-open ranges.

    Returns
    -------
    set[int]
        The union of all indices in the specified ranges.
    """
    out_set = set()
    for bounds in idx_ranges:
        out_set.update(range(bounds[0], bounds[1]))
    return out_set
