import numpy as np

from algoverify.methods.method import Method


class SDP(Method):
    def __init__(self, solver_params=None, solver="gurobi"):
        super(SDP, self).__init__(solver_params, solver)

    def add_var(self, param, lb=-np.inf, ub=np.inf):
        raise NotImplementedError

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
