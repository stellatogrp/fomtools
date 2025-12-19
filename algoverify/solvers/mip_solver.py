from functools import cached_property

import numpy as np
import scipy.sparse as spa

from algoverify.solvers.solver import Solver
from algoverify.utils.huchette_cuts import create_new_constr, get_cuts


class MIPSolver(Solver):
    def __init__(self, solver_params=None):
        super(MIPSolver, self).__init__(solver_params=solver_params)

    def relax_model(self):
        raise NotImplementedError

    def get_relaxed_variables(self, var):
        raise NotImplementedError

    @cached_property
    def relaxed_model(self):
        """
        Return the relaxed version of the MIP model, in which all integrality
        restrictions, SOS conditions, semi-continuous and semi-integer
        requirements on variables have been relaxed and general constraints
        have been removed.
        """
        return self._relaxed_model

    def add_huchette_cuts(self, steps):
        self.relax_model()

        for step in steps:
            y_var, y_val = self.get_relaxed_variables(step.rhs)
            w_var, w_val = self.get_relaxed_variables(step.lhs)

            l = np.concatenate(step.rhs.lb)
            u = np.concatenate(step.rhs.ub)

            for i in range(step.lhs.ndim):
                # special case out sparse matrices as needed
                A = np.concatenate(
                    [
                        np.asarray(v.todense()).reshape(-1) if spa.issparse(v) else v.flatten()
                        for v in step.rhs[i].decomposition_dict.values()
                    ]
                )

                # TODO: Gurobi is leaked in this part
                Iint, lI, h, L_hat, U_hat = get_cuts(step, i, w_val[i], A, y_val, l, u)
                if Iint is not None:
                    self.add_constr(
                        create_new_constr(w_var[i], A, y_var, Iint, lI, h, L_hat, U_hat)
                    )
