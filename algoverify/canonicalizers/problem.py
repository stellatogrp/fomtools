import numpy as np
from collections import namedtuple

from algoverify.solver import gurobi


class Problem(object):
    def __init__(self, solver_params=None, solver=gurobi.GurobiSolver):
        self.params = []
        self.param = namedtuple("Param", ["term", "lb", "ub"])

        self.solver_params = solver_params
        self.solver = solver

    def _add_term(self, term, lb=-np.inf, ub=np.inf):
        self.params.append(self.param(term=term, lb=lb, ub=ub))

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        self._add_term(param, lb, ub)

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_iterate(self, iterate, lb=-np.inf, ub=np.inf):
        self._add_term(iterate, lb, ub)

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
