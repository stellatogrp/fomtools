import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import Constraint
from fomtools.linearexpression import LinearExpression
from fomtools.solvers.clarabel import ClarabelSolver
from fomtools.solvers.cuclarabel import CuClarabelSolver
from fomtools.steps import Relu
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
    C_obj : scipy.sparse.csc_matrix
        Objective matrix for maximization.
    """

    def __init__(
        self,
        solver_params=None,
        solver="clarabel",
        use_rlt_cuts=False,
        use_triangle_cuts=False,
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
        use_rlt_cuts : bool, default=False
            If True, add RLT (Reformulation-Linearization Technique) cuts
            between cone-mate SDP indices to tighten the relaxation.
        use_triangle_cuts : bool, default=False
            If True, add lifted triangle cuts for ReLU steps with mixed-bound
            output dimensions to tighten the relaxation.
        """
        super(SDP, self).__init__(solver_params, solver)
        self.solver_name = solver.lower()  # Store for dispatch
        self.range_marker = 1
        self.var_idx_map = {}

        self.Aeq_matrices = []
        self.beq_consts = []
        self.Aineq_matrices = []
        self.bineq_upperbounds = []
        self.psd_cone_indices = []
        self.bound_constraint_indices = {}  # var -> (start_idx, end_idx)

        self.use_rlt_cuts = use_rlt_cuts
        self.use_triangle_cuts = use_triangle_cuts
        self._cuts_dirty = True
        self._cut_ineq_matrices = []
        self._cut_bineq_upperbounds = []
        self._relu_steps = []  # registry for triangle cuts

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
        new_canon.solver_params = self.solver_params
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
        new_canon.use_rlt_cuts = self.use_rlt_cuts
        new_canon.use_triangle_cuts = self.use_triangle_cuts
        # Cuts get rebuilt from current state on the copy's first solve.
        new_canon._cuts_dirty = True
        new_canon._cut_ineq_matrices = []
        new_canon._cut_bineq_upperbounds = []
        new_canon._relu_steps = list(self._relu_steps)  # shallow: Relu objs are shared
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
            case _:
                raise ValueError(f"Unknown constraint operator: {constraint.operator}")

    def _add_equality_constraint_impl(self, constraint: Constraint) -> None:
        """
        Add an equality constraint and its quadratic strengthening.

        For Ax + b = 0, adds:
        1. Linear constraint in lifted form
        2. Quadratic constraint: diag([b|A] X [b|A]^T) = 0

        With the PSD constraint, the diagonal being zero implies the full
        quadratic form is zero.

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

        # PSD cone for {0} ∪ x_indices
        self.add_psd_cone_indices(x_idx_bounds)
        self._cuts_dirty = True

    def _add_inequality_constraint_impl(self, constraint: Constraint) -> None:
        """
        Add an inequality constraint.

        For Ax + b <= 0, adds:
        1. Linear constraint in lifted form
        2. Optionally, RLT quadratic constraint (weak strengthening)

        Parameters
        ----------
        constraint : Constraint
            An inequality constraint (operator "<=").
        """
        raise NotImplementedError

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

        Parameters
        ----------
        step : Relu
            The ReLU step containing input expression (rhs) and output
            variable (lhs).
        """
        lhs, rhs = step.lhs, step.rhs
        A_mat, x_idx_bounds = self.linexpr_to_Amat_and_bounds(rhs)
        b_vec = rhs.const
        y_idx_bounds = [self.var_idx_map[lhs]]

        relu_psd_cone_indices = y_idx_bounds + x_idx_bounds

        m, n = A_mat.shape
        y_idx_ranges_2d = process_2d_ranges(y_idx_bounds, y_idx_bounds)
        y_idx_ranges_1d = process_1d_ranges(y_idx_bounds)
        xy_idx_ranges = process_2d_ranges(x_idx_bounds, y_idx_bounds)
        I = np.eye(lhs.ndim)

        curr_prob_dim = self.get_sdp_var_dim()
        for i in range(m):
            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            Ii = I[i]
            new_A[y_idx_ranges_2d] = np.outer(Ii, Ii)
            new_A[xy_idx_ranges] = -np.outer(A_mat[i], Ii)
            new_A[y_idx_ranges_1d] = -b_vec[i] * Ii

            self.Aeq_matrices.append(symmetrize(new_A.tocsc()))
            self.beq_consts.append(0)

        self.add_psd_cone_indices(relu_psd_cone_indices)
        self._relu_steps.append(step)
        self._cuts_dirty = True

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
        relu_lhs_vars = []

        for expr in expr_list:
            relu_lhs_vars.append(self._create_relu_step(canon_copy, expr))
            relu_lhs_vars.append(self._create_relu_step(canon_copy, -expr))

        # Build linear objective: maximize sum of all relu outputs
        # For lifted SDP with X = [1, x]^T @ [1, x], linear term c^T x
        # is encoded as C[0, idx] = c/2, C[idx, 0] = c/2
        curr_prob_dim = canon_copy.get_sdp_var_dim()
        C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))

        all_idx_bounds = []
        for lhs_var in relu_lhs_vars:
            idx_bounds = canon_copy.var_idx_map[lhs_var]
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
            The output (lhs) of the ReLU step.
        """
        step = Relu(rhs=expr)
        step.update_bounds(*canon_copy.get_expression_bounds(expr))
        canon_copy.add_var(step.lhs)
        canon_copy.canonicalize_relu(step)
        return step.lhs

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

        When both ``use_rlt_cuts`` and ``use_triangle_cuts`` are False this is
        effectively a no-op (cut lists stay empty).
        """
        if not self._cuts_dirty:
            return

        self._cut_ineq_matrices = []
        self._cut_bineq_upperbounds = []

        if self.use_rlt_cuts or self.use_triangle_cuts:
            sdp_idx_bounds = self._build_sdp_idx_bounds()

            if self.use_rlt_cuts:
                self._build_rlt_cuts(sdp_idx_bounds)

            if self.use_triangle_cuts:
                self._build_triangle_cuts()

        self._cuts_dirty = False

    def _build_rlt_cuts(self, sdp_idx_bounds: dict) -> None:
        """Append RLT cuts for off-diagonal cone-mate pairs (Phase 2)."""
        # Implementation lands in Phase 2.
        return

    def _build_triangle_cuts(self) -> None:
        """Append lifted triangle cuts for ReLU steps (Phase 3)."""
        # Implementation lands in Phase 3.
        return

    def solve_model(self, steps=None, convex_hull_cuts=False):
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
        return self._solved_copy._solve_internal()

    def _solve_internal(self):
        """
        Actually solve using this canonicalizer's data.

        This is the internal solve method called on the copy after
        the objective has been applied.

        Returns
        -------
        float
            The objective value.
        """
        match self.solver_name:
            case "clarabel":
                return self._solve_via_clarabel_internal()
            case "cvxpy":
                return self._solve_via_cvxpy_internal()
            case "cuclarabel":
                return self._solve_via_cuclarabel_internal()
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

    def solve_via_cvxpy(self):
        """
        Solve the SDP using the CVXPY modeling interface.

        Constructs a CVXPY problem from the stored constraint matrices
        and solves it using the Clarabel backend via CVXPY.

        Returns
        -------
        float
            The optimal objective value.
        """
        self._ensure_objective_applied()
        return self._solve_via_cvxpy_internal()

    def _solve_via_cvxpy_internal(self):
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
        prob = cp.Problem(obj, constraints)
        res = prob.solve(solver=cp.CLARABEL)
        return res

    def solve_via_clarabel(self):
        """
        Solve the SDP using the native Clarabel interface.

        Directly invokes the Clarabel solver without going through CVXPY,
        which can be more efficient for large problems.

        Returns
        -------
        float
            The optimal objective value.
        """
        self._ensure_objective_applied()
        return self._solve_via_clarabel_internal()

    def _solve_via_clarabel_internal(self):
        """Internal Clarabel solve without checking objective."""
        self._rebuild_cuts_if_dirty()
        Aineq_all = self.Aineq_matrices + self._cut_ineq_matrices
        bineq_all = self.bineq_upperbounds + self._cut_bineq_upperbounds

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
        return solver.solve()

    def solve_via_cuclarabel(self):
        """
        Solve the SDP using GPU-accelerated Clarabel.

        Uses the CUDA-enabled Clarabel solver for GPU acceleration.
        Requires CuPy and Julia with CUDA.jl and Clarabel.jl.

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
        return self._solve_via_cuclarabel_internal()

    def _solve_via_cuclarabel_internal(self):
        """Internal CuClarabel solve without checking objective."""
        self._rebuild_cuts_if_dirty()
        Aineq_all = self.Aineq_matrices + self._cut_ineq_matrices
        bineq_all = self.bineq_upperbounds + self._cut_bineq_upperbounds

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
        return solver.solve()


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
