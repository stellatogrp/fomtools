from functools import cached_property

import numpy as np

from algoverify.utils.huchette_cuts import create_new_constr, get_cuts


class Solver(object):
    def __init__(self, solver_params=None):
        self._relaxed_model = None
        self.vector_var_map = {}

    @cached_property
    def relaxed_model(self):
        """
        Return the relaxed version of the MIP model, in which all integrality
        restrictions, SOS conditions, semi-continuous and semi-integer
        requirements on variables have been relaxed and general constraints
        have been removed.
        """
        return self._relaxed_model

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

    def relax_model(self):
        raise NotImplementedError

    def get_relaxed_variables(self, var):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def add_huchette_cuts(self, steps, bounds):
        self.relax_model()

        for step in steps:
            y_var = self.get_relaxed_variables(step.rhs)
            w_var = self.get_relaxed_variables(step.lhs)
            l, u = tuple(
                np.concatenate(x) for x in zip(*[bounds[k] for k in step.rhs.decomposition_dict])
            )

            for i in range(step.lhs.ndim):
                # TODO: Gurobi is leaked in this part
                A = np.concatenate([v.flatten() for v in step.rhs[i].decomposition_dict.values()])
                Iint, lI, h, L_hat, U_hat = get_cuts(step, i, w_var.X[i], A, y_var.X, l, u)
                if Iint is not None:
                    self.add_constr(
                        create_new_constr(w_var[i], A, y_var, Iint, lI, h, L_hat, U_hat)
                    )
