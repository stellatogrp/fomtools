import numpy as np


class Solver(object):
    def __init__(self, solver_params=None):
        self._relaxed_model = None
        self.vector_var_map = {}
        if solver_params is not None:
            self.set_solver_params(solver_params)

    def set_solver_params(self, solver_params):
        for k, v in solver_params.items():
            self.set_param(k, v)

    def set_param(self, param, value):
        raise NotImplementedError

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_constr(self, constr):
        raise NotImplementedError

    def add_var(self, iterate, lb=-np.inf, ub=np.inf):
        raise NotImplementedError

    def constraints_to_solver(self, constraints):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
