import numpy as np

from algoverify.solver import gurobi
from algoverify.solver.solver import Solver


class Problem(object):
    def __init__(self, solver_params=None, solver="gurobi"):
        self.solver_params = solver_params
        self.solver = self._get_solver(solver)

    @staticmethod
    def _get_solver(solver: str) -> Solver:
        """
        Return an initialized solver instance for the requested backend.

        Parameters
        ----------
        solver: str, default 'gurobi'
            Case-insensitive identifier for the solver to use. Supported values
            (and common aliases) include:
            - ``"gurobi"``, ``"grb"``

        Returns
        -------
        Solver
            A ready-to-use instance of a subclass of :class:`Solver`
            (e.g., ``gurobi.Gurobi()``).

        Raises
        ------
        ValueError
            If `solver` is not one of the supported options.
        """
        match solver.lower():
            case "gurobi" | "grb":
                return gurobi.GurobiSolver()
            case _:
                raise ValueError(f"Solver must be 'gurobi' or 'TBD', got {solver}.")

    def add_var(self, iterate, lb=-np.inf, ub=np.inf):
        raise NotImplementedError

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
