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
    def __init__(self, solver_params=None, solver="clarabel"):
        super(SDP, self).__init__(solver_params, solver)
        self.range_marker = 1
        self.var_idx_map = {}

        self.Aeq_matrices = []
        self.beq_consts = []
        self.Aineq_matrices = []
        self.bineq_upperbounds = []
        self.psd_cone_indices = []

        self.C_obj = None
        self._initialize_corner_constraint()

    def _initialize_corner_constraint(self) -> None:
        # adds the constraint that the upper left corner must equal 1
        curr_prob_dim = self.get_sdp_var_dim()
        new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
        new_A[0, 0] = 1
        self.Aeq_matrices.append(symmetrize(new_A.tocsc()))
        self.beq_consts.append(1)

    def _pad_existing_A_matrices(self) -> None:
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
        lb, ub = var.lb, var.ub
        dim = var.ndim
        bounds = (self.range_marker, self.range_marker + dim)
        self.range_marker += dim
        self.var_idx_map[var] = bounds

        # whenever a new var is added, need to pad existing A matrices
        self._pad_existing_A_matrices()

        curr_prob_dim = self.get_sdp_var_dim()
        for (i, l, u) in zip(range(bounds[0], bounds[1]), lb, ub):
            # TODO: handle case where lb or ub is inf
            new_A = spa.lil_matrix((curr_prob_dim, curr_prob_dim))
            new_A[i, i] = 1
            new_A[i, 0] = - (l + u)

            new_A = new_A.tocsc()
            new_A = symmetrize(new_A)
            self.Aineq_matrices.append(new_A)
            self.bineq_upperbounds.append(- u * l)

    def get_sdp_var_dim(self) -> int:
        return self.range_marker

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError

    def add_psd_cone_indices(self, new_indices):
        new_idx_set = range_to_indices(new_indices)
        new_idx_set.add(0)

        # TODO: in principal, there should never be set containment
        # but need to investigate
        self.psd_cone_indices.append(new_idx_set)

    def canonicalize_relu(self, step: Relu) -> None:
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
        # turns linexpr into Ax + b into a full matrix A
        # along with corresponding indices in the SDP matrix var
        all_A = []
        idx_bounds = []
        for leaf_var, mat in expr.decomposition_dict.items():
            all_A.append(mat.todense() if spa.issparse(mat) else mat)
            idx_bounds.append(self.var_idx_map[leaf_var])

        return np.hstack(all_A), idx_bounds

    def set_two_norm_objective(self, expr_list) -> None:
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

    def solve_via_cvxpy(self):
        dim = self.get_sdp_var_dim()
        X = cp.Variable((dim, dim), symmetric=True)
        obj = cp.Maximize(cp.trace(self.C_obj @ X))
        # constraints = [X >> 0]
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
        res = prob.solve()
        print(res)

    def solve_via_clarabel(self):
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
        solver.solve()
    
    def solve_via_cuclarabel(self):
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
        solver.solve()


def symmetrize(mat):
    return (mat + mat.T) / 2


def range_to_indices(idx_ranges):
    out_set = set([])
    for bounds in idx_ranges:
        out_set.update(range(bounds[0], bounds[1]))
    return out_set
