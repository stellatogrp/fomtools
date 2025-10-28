import logging

import gurobipy as gp
import numpy as np

from algoverify.canonicalizers.canonicalizer import Canonicalizer

log = logging.getLogger(__name__)


class OBBTHandler(Canonicalizer):
    def __init__(self):
        super().__init__()

    def add_var(self, var, lb=-np.inf, ub=np.inf):
        self.solver.add_var(var, lb, ub)

    def add_param_stack_var(self, param_stack_var, param_list):
        self.solver.add_param_stack_var(param_stack_var, param_list)

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        lhs_gp_expr = self.solver.expression_to_solver(lhs_expr)
        rhs_gp_expr = self.solver.expression_to_solver(rhs_expr)

        self.solver.add_constr(lhs_gp_expr == rhs_gp_expr)

    def obbt(self, linexpr, lb, ub):
        self.solver.update_model()
        lb_out = np.zeros(linexpr.ndim)
        ub_out = np.zeros(linexpr.ndim)

        gp_expr = self.solver.expression_to_solver(linexpr)

        for sense in ["minimize", "maximize"]:
            for i in range(linexpr.ndim):
                self.solver.set_objective(gp_expr[i], sense)
                self.solver.optimize()


                if self.solver.model.status != gp.GRB.OPTIMAL:
                    log.info(f'bound tighting failed, GRB model status: {self.solver.model.status}')
                    log.info(f'i = {i}, sense={sense}')
                    return

                if sense == "maximize":
                    ub_out[i] = self.solver.model.objVal
                else:
                    lb_out[i] = self.solver.model.objVal

        # reset objective
        self.solver.set_objective(0)

        assert np.all(lb_out <= ub_out)
        return np.maximum(lb, lb_out), np.minimum(ub, ub_out)
