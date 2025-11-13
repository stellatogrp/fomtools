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
        # Variable to keep track of number of highs variable created
        self._var_counter = -1
        self.model = hp.Highs()

    def __getitem__(self, item):
        val = {"objVal": self.model.getInfo().objective_function_value}
        return val[item]

    def set_param(self, param, val):
        pass

    def relax_model(self):
        import copy

        self._relaxed_model = hp.Highs()
        relaxed_lp = hp.HighsLp()
        lp = self.model.getLp()

        relaxed_lp.num_col_ = copy.deepcopy(lp.num_col_)
        relaxed_lp.num_row_ = copy.deepcopy(lp.num_row_)

        relaxed_lp.col_cost_ = copy.deepcopy(lp.col_cost_)
        relaxed_lp.col_lower_ = copy.deepcopy(lp.col_lower_)
        relaxed_lp.col_upper_ = copy.deepcopy(lp.col_upper_)
        relaxed_lp.row_lower_ = copy.deepcopy(lp.row_lower_)
        relaxed_lp.row_upper_ = copy.deepcopy(lp.row_upper_)

        relaxed_lp.a_matrix_.start_ = copy.deepcopy(lp.a_matrix_.start_)
        relaxed_lp.a_matrix_.index_ = copy.deepcopy(lp.a_matrix_.index_)
        relaxed_lp.a_matrix_.value_ = copy.deepcopy(lp.a_matrix_.value_)

        relaxed_lp.integrality_ = [hp.HighsVarType.kContinuous] * relaxed_lp.num_col_
        relaxed_lp.sense_ = lp.sense_
        relaxed_lp.offset_ = lp.offset_

        self._relaxed_model.passModel(relaxed_lp)
        self._relaxed_model.optimize()

        self.__dict__.pop("relaxed_model", None)

    def add_var(self, var, lb=-np.inf, ub=np.inf):
        lb = np.full(var.ndim, lb)
        ub = np.full(var.ndim, ub)

        def create_single_var(_lb, _ub, is_binary):
            self._var_counter += 1
            if is_binary:
                return self.model.addIntegral(lb=0, ub=1, name=f"v{self._var_counter}")
            else:
                return self.model.addVariable(lb=_lb, ub=_ub, name=f"v{self._var_counter}")

        for k, v in var.decomposition_dict.items():
            self.vector_var_map[k] = [
                create_single_var(lb[i], ub[i], k.is_binary) for i in range(k.ndim)
            ]

    def expression_to_solver(self, expression: LinearExpression):
        highs_var = np.zeros(expression.ndim)
        for k, v in expression.decomposition_dict.items():
            v = v.toarray() if spa.issparse(v) else np.array(v)
            if self.vector_var_map.get(k, None) is None:
                self.add_var(k)

            highs_var = highs_var + v @ self.vector_var_map[k]

        return highs_var + expression.const

    def constraints_to_solver(self, constraints: List[Constraint]):
        for i, constr in enumerate(constraints):
            self.add_constr(constr)

    def update_bounds(self, target_expr, bound_lb, bound_ub):
        for i, var in enumerate(self.expression_to_solver(target_expr)):
            var.bounds = (bound_lb[i], bound_ub[i])

    def add_constr(self, constraint):
        def _add_constr(_lhs):
            match constraint.operator:
                case "=":
                    self.model.addConstr(_lhs == 0)
                case "<=":
                    self.model.addConstr(_lhs <= 0)

        try:
            self.model.addConstr(constraint)
        except AttributeError:
            for i, constr in enumerate(self.expression_to_solver(constraint.lhs)):
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

    def get_relaxed_variables(self, expression):
        # If no relaxed model currently exists, build and solve it first.
        if self.relaxed_model is None:
            self.relax_model()

        highs_var, values = [], []
        for k in expression.decomposition_dict.keys():
            _hp_var = self.vector_var_map[k]
            highs_var.extend(_hp_var)
            values.extend(self.relaxed_model.variableValues(_hp_var))

        return highs_var, np.array(values)

    def is_optimal(self):
        return self.model.getModelStatus() == hp.HighsModelStatus.kOptimal
