import numpy as np

from algoverify.canonicalizers.solver.solver import Solver


class SDPCanonicalizer(Solver):

    def __init__(self, solver_params={}):
        super().__init__(solver_params=solver_params)

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        raise NotImplementedError

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        raise NotImplementedError

    def add_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
