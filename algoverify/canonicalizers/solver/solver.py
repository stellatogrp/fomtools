import numpy as np


class Solver(object):

    def __init__(self, solver_params={}):
        self.solver_params = solver_params

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
