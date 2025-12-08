import numpy as np

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.linearexpression import LinearExpression
from fomtools.steps import Relu


class SDP(Canonicalizer):
    def __init__(self, solver_params=None, solver="clarabel"):
        super(SDP, self).__init__(solver_params, solver)
        self.range_marker = 1
        self.var_idx_map = {}

        self.Aeq_matrices = []
        self.beq_consts = []
        self.Aineq_matrices = []
        self.bineq_lowerbounds = []
        self.bineq_upperbounds = []

    def add_var(self, var: LinearExpression, lb=-np.inf, ub=np.inf) -> None:
        dim = var.get_output_dim()
        bounds = (self.range_marker, self.range_marker + dim)
        self.range_marker += dim
        self.var_idx_map[var] = bounds

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError

    def add_relu_constraints(self, step: Relu) -> None:
        pass
