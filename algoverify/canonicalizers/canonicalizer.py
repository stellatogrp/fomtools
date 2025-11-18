import numpy as np

from algoverify.solvers import gurobi, highs
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

    def __init__(self, solver_params: dict = None, solver: str = "gurobi"):
        """
        Initialize an instance of :class:`Method`.

        Parameters
        ----------
        solver_params: dict or None, optional
            Key-value pairs with solver parameters and their corresponding
            configuration values.

        solver: str, default "gurobi"
            Case-insensitive identifier for the solver to use. Supported
            values (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
        """
        self.solver_params = solver_params
        self.solver = self._get_solver(solver, solver_params)

    @staticmethod
    def _get_solver(solver: str, solver_params: dict) -> Solver:
        """
        Return an initialized solver instance for the requested backend.

        Parameters
        ----------
        solver: str
            Case-insensitive identifier for the solver to use. Supported values
            (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
            - ``"highs"``

        solver_params: dict or None, optional
            Key-value pairs with solver parameters and their corresponding
            configuration values.

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
                return gurobi.GurobiSolver(solver_params)
            case "highs":
                return highs.HiGHSSolver(solver_params)
            case _:
                raise ValueError(f"Solver must be 'gurobi' or 'TBD', got {solver}.")

    def add_var(self, iterate):
        raise NotImplementedError

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
