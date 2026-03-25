import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.canonicalizer import Canonicalizer
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

    def __init__(self, solver_params=None, solver="clarabel"):
        """
        Initialize the SDP canonicalizer.

        Parameters
        ----------
        solver_params : dict, optional
            Dictionary of solver-specific parameters to pass to the backend.
        solver : str, default='clarabel'
            Name of the SDP solver to use. Options: 'clarabel', 'cvxpy',
            'cvxpy_sdp', 'cuclarabel'.
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

        self.C_obj = None
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
        new_Aeq_matrices = [
            efficient_csc_pad(A, (new_dim, new_dim)) for A in self.Aeq_matrices
        ]
        new_Aineq_matrices = [
            efficient_csc_pad(A, (new_dim, new_dim)) for A in self.Aineq_matrices
        ]
        self.Aeq_matrices = new_Aeq_matrices
        self.Aineq_matrices = new_Aineq_matrices

    def add_var(self, var: LinearExpression) -> None:
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

        for (i, l, u) in zip(range(bounds[0], bounds[1]), lb, ub):
            if np.isinf(l) or np.isinf(u):
                raise NotImplementedError("Infinite bounds not yet supported in SDP canonicalizer")
            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            new_A[i, i] = 1
            new_A[i, 0] = - (l + u)

            new_A = new_A.tocsc()
            new_A = symmetrize(new_A)
            self.Aineq_matrices.append(new_A)
            self.bineq_upperbounds.append(- u * l)

        constraint_end = len(self.Aineq_matrices)
        self.bound_constraint_indices[var] = (constraint_start, constraint_end)

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
        lb : float or np.ndarray
            New lower bound(s).
        ub : float or np.ndarray
            New upper bound(s).

        Raises
        ------
        ValueError
            If the expression is not found in the canonicalizer.
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

        for j, (i, l, u) in enumerate(zip(range(sdp_start, sdp_end), lb, ub)):
            if np.isinf(l) or np.isinf(u):
                raise NotImplementedError("Infinite bounds not yet supported in SDP canonicalizer")

            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            new_A[i, i] = 1
            new_A[i, 0] = -(l + u)

            constraint_idx = constraint_start + j
            self.Aineq_matrices[constraint_idx] = symmetrize(new_A.tocsc())
            self.bineq_upperbounds[constraint_idx] = -u * l

    def add_param_stack_var(self, param_stack_var, param_list):
        """
        Stack parameter variables (not implemented for SDP).

        Raises
        ------
        NotImplementedError
            This method is not supported in the SDP canonicalizer.
        """
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        """
        Add a linear equality constraint (not implemented for SDP).

        Raises
        ------
        NotImplementedError
            This method is not supported in the SDP canonicalizer.
        """
        raise NotImplementedError

    def add_psd_cone_indices(self, new_indices):
        """
        Register a new PSD cone constraint on a submatrix.

        Adds the constraint that the submatrix of X indexed by ``new_indices``
        (including the homogenization index 0) is positive semidefinite.

        Parameters
        ----------
        new_indices : list[tuple[int, int]]
            List of (start, end) index ranges defining the submatrix.
            Index 0 is automatically included for homogenization.
        """
        new_idx_set = range_to_indices(new_indices)
        new_idx_set.add(0)

        # TODO: in principle, there should never be set containment
        # but need to investigate
        self.psd_cone_indices.append(new_idx_set)

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

    def set_two_norm_objective(self, expr_list) -> None:
        """
        Set the objective to maximize the sum of squared two-norms.

        Configures the SDP objective matrix C to compute
        sum_i ||A_i x + b_i||_2^2 = sum_i tr(Q_i @ X) for appropriate
        quadratic forms Q_i derived from each expression.

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose squared norms are summed in the
            objective.
        """
        curr_prob_dim = self.get_sdp_var_dim()
        overall_C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))

        for expr in expr_list:
            C_mat = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            A_mat, idx_bounds = self.linexpr_to_Amat_and_bounds(expr)
            b_vec = expr.const.reshape((-1, 1))

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

        self.C_obj = overall_C_mat

    def set_infinity_norm_objective(self, expr_list, lb, ub):
        """
        Set infinity-norm objective (not supported for SDP).

        Parameters
        ----------
        expr_list : list[LinearExpression]
            Unused.
        lb : float
            Unused.
        ub : float
            Unused.

        Raises
        ------
        NotImplementedError
            Infinity-norm objectives require MIP formulation.
        """
        raise NotImplementedError(
            "Infinity-norm objective is not supported with SDP. Use set_two_norm_objective instead."
        )

    def solve_model(self, steps=None, convex_hull_cuts=False):
        """Solve the SDP problem using the configured solver.

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
        match self.solver_name:
            case "clarabel":
                return self.solve_via_clarabel()
            case "cvxpy" | "cvxpy_sdp":
                return self.solve_via_cvxpy()
            case "cuclarabel":
                return self.solve_via_cuclarabel()
            case _:
                raise ValueError(f"Unknown SDP solver: {self.solver_name}")

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
        dim = self.get_sdp_var_dim()
        X = cp.Variable((dim, dim), symmetric=True)
        obj = cp.Maximize(cp.trace(self.C_obj @ X))
        constraints = []
        for A, b in zip(self.Aeq_matrices, self.beq_consts):
            constraints += [
                cp.trace(A @ X) == b,
            ]
        for A, b in zip(self.Aineq_matrices, self.bineq_upperbounds):
            constraints += [
                cp.trace(A @ X) <= b,
            ]
        constraints += [X >> 0]
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
        solver = ClarabelSolver(solver_params=self.solver_params)
        solver.setup_clarabel_data(
            self.get_sdp_var_dim(),
            self.C_obj,
            self.Aeq_matrices,
            self.beq_consts,
            self.Aineq_matrices,
            self.bineq_upperbounds,
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
        """
        solver = CuClarabelSolver(solver_params=self.solver_params)
        solver.setup_clarabel_data(
            self.get_sdp_var_dim(),
            self.C_obj,
            self.Aeq_matrices,
            self.beq_consts,
            self.Aineq_matrices,
            self.bineq_upperbounds,
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
