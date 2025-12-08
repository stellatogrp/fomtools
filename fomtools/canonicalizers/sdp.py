import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.linearexpression import LinearExpression
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

        self.C_obj = None

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

    def get_sdp_var_dim(self):
        return self.range_marker

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError

    def canonicalize_relu(self, step: Relu) -> None:
        # raise NotImplementedError
        pass

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

            overall_C_mat += C_mat

        self.C_obj = overall_C_mat


def symmetrize(mat):
    return (mat + mat.T) / 2
