import scipy.sparse as spa

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.linearexpression import LinearExpression
from fomtools.steps import Relu
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

def symmetrize(mat):
    return (mat + mat.T) / 2
