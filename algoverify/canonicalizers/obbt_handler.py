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
        self.solver.add_constr(lhs_expr == rhs_expr)

    def obbt(self, linexpr, lb, ub):
        lb_out = np.zeros(linexpr.ndim)
        ub_out = np.zeros(linexpr.ndim)

        for sense in ["minimize", "maximize"]:
            for i, expr in enumerate(linexpr):
                self.solver.set_objective(expr, sense)
                self.solver.optimize()

                if self.solver["status"] != gp.GRB.OPTIMAL:
                    log.info(f"bound tighting failed, GRB model status: {self.solver['status']}")
                    log.info(f"i = {i}, sense={sense}")
                    return

                if sense == "maximize":
                    ub_out[i] = self.solver["objVal"]
                else:
                    lb_out[i] = self.solver["objVal"]

        # reset objective
        self.solver.set_objective(0)

        assert np.all(lb_out <= ub_out)
        return np.maximum(lb, lb_out), np.minimum(ub, ub_out)
