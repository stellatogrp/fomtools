import numpy as np

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.linearexpression import LinearExpression


class SDP(Canonicalizer):
    def __init__(self, solver_params=None, solver="gurobi"):
        super(SDP, self).__init__(solver_params, solver)
        self.range_marker = 1
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.var_bound_map = {}

    def add_var(self, var: LinearExpression, lb=-np.inf, ub=np.inf) -> None:
        dim = var.get_output_dim()
        bounds = (self.range_marker, self.range_marker + dim)
        self.range_marker += dim
        self.var_bound_map[var] = bounds
        self.lower_bounds[var] = lb
        self.upper_bounds[var] = ub

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
