import logging

import numpy as np

from fomtools.canonicalizers.canonicalizer import Canonicalizer

log = logging.getLogger(__name__)


class OBBTHandler(Canonicalizer):
    def __init__(self, solver):
        super().__init__(solver=solver)

    def add_var(self, var):
        self.solver.add_var(var, var.lb, var.ub)

    def obbt(self, linexpr, lb, ub):
        lb_out = np.zeros(linexpr.ndim)
        ub_out = np.zeros(linexpr.ndim)

        for sense in ["minimize", "maximize"]:
            for i, expr in enumerate(linexpr):
                self.solver.set_objective(expr, sense)
                self.solver.optimize()

                if not self.solver.is_optimal():
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
