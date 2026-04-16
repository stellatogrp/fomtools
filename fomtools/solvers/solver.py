import numpy as np


class Solver(object):
    """
    Abstract base class for optimization solvers.

    This class defines the common interface for all solver backends in FOMTools,
    including MIP solvers (Gurobi, HiGHS) and SDP solvers (Clarabel). Subclasses
    must implement the abstract methods to interface with their specific solver.

    Attributes
    ----------
    vector_var_map : dict
        Mapping from LinearExpression objects to their corresponding solver variables.
    """

    def __init__(self, solver_params=None):
        """
        Initialize the solver with optional parameters.

        Parameters
        ----------
        solver_params : dict, optional
            Dictionary of solver-specific parameters to configure.
            Keys are parameter names and values are the settings.
        """
        self._relaxed_model = None
        self.vector_var_map = {}
        if solver_params is not None:
            self.set_solver_params(solver_params)

    def set_solver_params(self, solver_params):
        """
        Apply a dictionary of solver parameters.

        Parameters
        ----------
        solver_params : dict
            Dictionary mapping parameter names to their values.
            Each key-value pair is passed to ``set_param``.
        """
        for k, v in solver_params.items():
            self.set_param(k, v)

    def set_param(self, param, value):
        """
        Set an individual solver parameter.

        Parameters
        ----------
        param : str
            Name of the solver parameter.
        value : int or float
            Value to assign to the parameter.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def add_param_stack_var(self, param_stack_var, param_list):
        """
        Create a stacked variable from a list of parameter variables.

        Parameters
        ----------
        param_stack_var : LinearExpression
            The new variable to register as the stacked result.
        param_list : list[LinearExpression]
            List of existing variables to stack together.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def add_constr(self, constr):
        """
        Add a single constraint to the model.

        Parameters
        ----------
        constr : Constraint
            The constraint to add to the solver model.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def add_var(self, iterate, lb=-np.inf, ub=np.inf):
        """
        Add a decision variable to the model.

        Parameters
        ----------
        iterate : LinearExpression
            The variable to register with the solver.
        lb : float, default=-np.inf
            Lower bound for the variable.
        ub : float, default=np.inf
            Upper bound for the variable.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def constraints_to_solver(self, constraints):
        """
        Add multiple constraints to the model.

        Parameters
        ----------
        constraints : list[Constraint]
            List of constraints to add to the solver model.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def optimize(self):
        """
        Solve the optimization model.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def copy(self) -> "Solver":
        """
        Create an independent copy of this solver.

        The copy can be modified without affecting the original.

        Returns
        -------
        Solver
            A new solver instance with copied state.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError
