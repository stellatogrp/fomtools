import logging

import gurobipy as gp
import numpy as np
from algoverify.solver.solver import Solver

log = logging.getLogger(__name__)


class GurobiSolver(Solver):
    def __init__(self, solver_params={}):
        self.vector_var_map = {}
        self.model = gp.Model()

        for key, val in solver_params.items():
            self.model.setParam(key, val)

        self.step_constr_map = {}
        self.obj_constraints = []

    def _add_term(self, term, lb=-np.inf, ub=np.inf):
        self.vector_var_map[term] = self.get_matrix_var(term.ndim, lb=lb, ub=ub)
        self.update_model()

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        self._add_term(param, lb, ub)

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self._add_term(iterate, lb, ub)

    def add_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self._add_term(iterate, lb, ub)

    def add_param_stack_var(self, param_stack_var, param_list):
        """
        param_stack_var is the VP param object and param_list is the list of previous params to stack
        """
        self.vector_var_map[param_stack_var] = gp.hstack(
            [self.vector_var_map[param] for param in param_list]
        )

    def update_vector_bounds(self, iterate, lb, ub):
        gp_var = self.vector_var_map[iterate]

        gp_var.lb = np.maximum(gp_var.lb, lb)
        gp_var.ub = np.minimum(gp_var.ub, ub)

    def leaf_expr_to_gp_var(self, expression):
        if expression.is_leaf:
            return self.vector_var_map[next(iter(expression.decomposition_dict))]

        raise TypeError(
            "Invalid input: expression must be a leaf. Received a composite one instead."
        )

    def expression_to_solver(self, expression):
        gp_expr = np.zeros(expression.ndim)
        for k, v in expression.decomposition_dict.items():
            gp_expr += v @ self.vector_var_map[k]

        return gp_expr

    def constraints_to_solver(self, step, constraints):
        self.step_constr_map[step] = [self.add_constr(constr) for constr in constraints]

    def add_constr(self, constraint):
        self.model.addConstr(constraint)
        self.update_model()

    def get_var(self, lb=-np.inf, ub=np.inf, vtype="continuous"):
        match vtype:
            case "continuous":
                vtype = gp.GRB.CONTINUOUS
            case "binary":
                vtype = gp.GRB.BINARY
            case _:
                raise TypeError(f"Variable type must be either continuous or binary, got {vtype}")

        return self.model.addVar(lb=lb, ub=ub, vtype=vtype)

    def get_matrix_var(self, n, lb=-np.inf, ub=np.inf, vtype="continuous"):
        match vtype:
            case "continuous":
                vtype = gp.GRB.CONTINUOUS
            case "binary":
                vtype = gp.GRB.BINARY
            case _:
                raise TypeError(f"Variable type must be either continuous or binary, got {vtype}")

        return self.model.addMVar(n, lb=lb, ub=ub, vtype=vtype)

    def set_objective(self, expression, type=None):
        match type:
            case "minimize":
                type = gp.GRB.MINIMIZE
            case "maximize":
                type = gp.GRB.MAXIMIZE
            case _:
                raise TypeError(f"Objective type must be either minimize or maximize, got {type}")
        self.model.setObjective(expression, sense=type)
        self.update_model()

    def relax_model(self):
        return self.model.relax()

    def optimize(self):
        self.model.optimize()

    def update_model(self):
        self.model.update()

    def get_solution_value(self, var_name):
        return self.model.getVarByName(var_name.VarName.item()).X

    def extract_vec_var_by_name(self, var, model):
        return gp.MVar.fromlist(
            [model.getVarByName(var[i].item().VarName) for i in range(var.shape[0])]
        )

    def get_solve_data(self):
        return {
            "objVal": self.model.objVal,
            "objBound": self.model.objBound,
            "MIPGap": self.model.MIPGap,
            "Runtime": self.model.Runtime,
            "rel_LP_sol": self.rel_LP_sol,
            "numBinVars": self.model.NumBinVars,
        }
