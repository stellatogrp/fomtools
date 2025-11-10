from typing import List

import highspy as hp
from highspy import ObjSense, HighsModelStatus
import numpy as np

from algoverify import LinearExpression
from algoverify.constraint import Constraint
from algoverify.solvers.solver import Solver
import scipy.sparse as spa

class HiGHSSolver(Solver):
    def __init__(self, solver_params={}):
        super().__init__(solver_params)
        self.model = hp.Highs()

    def __getitem__(self, item):
        val = {
            'objVal': self.model.getInfo().objective_function_value
        }
        return val[item]

    def add_var(self, var, lb=-np.inf, ub=np.inf):
        lb = np.full(var.ndim, lb)
        ub = np.full(var.ndim, ub)

        def create_single_var(_lb, _ub, is_binary):
            if is_binary:
                return self.model.addIntegral(lb=0, ub=1)
            else:
                try:
                    return self.model.addVariable(lb=_lb, ub=_ub)
                except:
                    return self.model.addVariable(lb=_lb, ub=_ub)

        for k, v in var.decomposition_dict.items():
            self.vector_var_map[k] = [create_single_var(lb[i], ub[i], k.is_binary) for i in range(k.ndim)]


    def add_param_stack_var(self, param_stack_var, param_list):
        pass

    def leaf_expr_to_solver(self):
        pass

    def expression_to_solver(self, expression: LinearExpression):
        highs_var = np.zeros(expression.ndim)
        for k, v in expression.decomposition_dict.items():
            v = v.toarray() if spa.issparse(v) else v
            if self.vector_var_map.get(k, None) is None:
                self.add_var(k)

            highs_var = highs_var + v @ self.vector_var_map[k]

        return highs_var

    def constraints_to_solver(self, constraints: List[Constraint]):
        for constr in constraints:
            self.add_constr(constr)

    def update_bounds(self):
        pass

    def add_constr(self, constraint):
        def _add_constr(_lhs):
            match constraint.operator:
                case "=":
                    self.model.addConstr(_lhs == 0)
                case "<=":
                    self.model.addConstr(_lhs <= 0)

        for constr in self.expression_to_solver(constraint.lhs):
            _add_constr(constr)

    def set_objective(self, expression, sense=None):
        match sense:
            case "minimize":
                sense = ObjSense.kMinimize
            case "maximize":
                sense = ObjSense.kMaximize
            case None:
                sense = None
            case _:
                raise TypeError(f"Objective sense must be either minimize or maximize, got {sense}")

        try:
            if int(expression) != 0:
                raise ValueError(
                    f"Objective function must be either a LinearExpression object or a 0, got {expression}"
                )
            for i in range(self.model.numVariables):
                self.model.changeColCost(i, 0)
        except TypeError:
            self.model.setObjective(self.expression_to_solver(expression)[0], sense=sense)



    def optimize(self):
        self.model.optimize()

    def get_solution_value(self, expression: LinearExpression):
        out = 0
        for k, v in expression.decomposition_dict.items():
            out += v @ np.array([self.model.val(i) for i in self.vector_var_map[k]])
        return out

    def relaxed_model(self):
        pass

    def get_relaxed_variables(self, var):
        pass

    def is_optimal(self):
        return self.model.getModelStatus() == hp.HighsModelStatus.kOptimal
