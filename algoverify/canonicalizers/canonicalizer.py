import numpy as np

from algoverify import LinearExpression
from algoverify.solvers import gurobi
from algoverify.solvers.solver import Solver


class Canonicalizer:
    """
    A base class for mathematical-optimization methods. This class
    defines a common interface for mixed-integer programming (MIP),
    and semidefinite programming (SDP).

    Attributes
    ----------
    solver: Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).

    Methods
    -------
    ...

    """

    def __init__(self, solver_params: list[LinearExpression] = None, solver: str = "gurobi"):
        """
        Initialize an instance of :class:`Method`.

        Parameters
        ----------
        solver_params: list or None, optional
            List of solver parameters to start with.

        solver: str, default "gurobi"
            Case-insensitive identifier for the solver to use. Supported
            values (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
        """
        self.solver_params = solver_params
        self.solver = self._get_solver(solver)

    @staticmethod
    def _get_solver(solver: str) -> Solver:
        """
        Return an initialized solver instance for the requested backend.

        Parameters
        ----------
        solver: str
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
