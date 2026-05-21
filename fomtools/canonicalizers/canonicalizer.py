from abc import abstractmethod

import numpy as np

from fomtools.constraint import Constraint
from fomtools.linearexpression import LinearExpression
from fomtools.solvers import gurobi, highs
from fomtools.solvers.solver import Solver
from fomtools.steps.step import Step


class Canonicalizer:
    """
    A base class for mathematical-optimization methods. This class
    defines a common interface for mixed-integer programming (MIP),
    and semidefinite programming (SDP).

    Subclasses declare a class-level ``_CANONICALIZERS`` dict mapping the
    lowercase step class name (e.g. ``"relu"``) to the function that
    encodes it. ``canonicalize`` looks up the function in this dict and
    raises ``NotImplementedError`` if absent.

    Attributes
    ----------
    solver: Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).

    """

    _CANONICALIZERS: dict = {}

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
        self.obbt_handler = None
        if solver is not None:
            self.solver_name = solver.lower()
            self.solver = self._get_solver(solver, solver_params)
        else:
            self.solver_name = None
            self.solver = None

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
            case "clarabel" | "cvxpy" | "cvxpy_sdp" | "cuclarabel":
                return None  # SDP solvers are instantiated at solve time
            case _:
                raise ValueError(
                    f"Unknown solver: {solver}. Supported solvers: "
                    "'gurobi', 'highs', 'clarabel', 'cvxpy', 'cvxpy_sdp', 'cuclarabel'."
                )

    def canonicalize(self, step: Step):
        """
        Dispatch the provided step into a backend-specific encoding.

        Looks up the lowercase step class name (e.g. ``"relu"``) in this
        canonicalizer's ``_CANONICALIZERS`` registry and invokes the
        corresponding encoder. Raises ``NotImplementedError`` if the step
        type is not registered.

        Parameters
        ----------
        step: Step
            The concrete step class to canonicalize. This class encapsulates
            the thresholding or activation operator.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If this canonicalizer's ``_CANONICALIZERS`` registry has no
            entry for the concrete type of ``step``.

        """
        canonicalizer = self._CANONICALIZERS.get(type(step).__name__.lower())
        if canonicalizer is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support {type(step).__name__}"
            )
        canonicalizer(self, step)

    def add_var(self, iterate, **kwargs) -> None:
        """
        Add a variable to the canonicalizer instance.

        Parameters
        ----------
        var : LinearExpression
            Linear expression object to add to the canonicalizer.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint to the canonicalizer instance.

        Parameters
        ----------
        constraint : Constraint
            Constraint object to add to the canonicalizer.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_bounds(
        self, expression: LinearExpression, lb: float | np.ndarray, ub: float | np.ndarray
    ) -> None:
        """
        Update the lower and upper bounds of the solver variables corresponding
        to the given ``LinearExpression``.

        Parameters
        ----------
        expression : LinearExpression
            A ``LinearExpression`` object whose underlying variable
            bounds are to be updated.

        lb : float or np.ndarray
            New lower bound(s).

        ub : float or np.ndarray
            New upper bound(s).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def set_two_norm_objective(
        self, expr_list: list[LinearExpression], lb_list=None, ub_list=None
    ) -> None:
        """
        Set the objective to maximize the sum of squared two-norms.

        Configures the optimization objective to compute
        sum_i ||expr_i||_2^2 over the given list of expressions.

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose squared norms are summed in the
            objective.
        lb_list : list[np.ndarray], optional
            Lower bounds for each expression (may be used by some backends).
        ub_list : list[np.ndarray], optional
            Upper bounds for each expression (may be used by some backends).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses that support this objective type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_two_norm_objective"
        )

    def set_one_norm_objective(self, expr_list: list[LinearExpression]) -> None:
        """
        Set the objective to maximize the one-norm (sum of absolute values).

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose one-norms are summed in the objective.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses that support this objective type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_one_norm_objective"
        )

    def set_infinity_norm_objective(
        self, expr_list: list[LinearExpression], lb=None, ub=None
    ) -> None:
        """
        Set the objective to maximize the infinity-norm (maximum absolute value).

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose infinity-norm maximum is the objective.
        lb : np.ndarray, optional
            Lower bounds for the expressions (may be used by some backends).
        ub : np.ndarray, optional
            Upper bounds for the expressions (may be used by some backends).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses that support this objective type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_infinity_norm_objective"
        )

    def get_expression_bounds(
        self, expr: LinearExpression
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute bounds for a linear expression.

        Uses OBBT if available for tighter bounds, otherwise falls back
        to interval bound propagation.

        Parameters
        ----------
        expr : LinearExpression
            The expression whose bounds are to be computed.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (lb, ub) of lower and upper bounds.
        """
        lb, ub = expr.bound_propagation()

        if self.obbt_handler is not None:
            result = self.obbt_handler.obbt(expr, lb, ub)
            if result is not None:
                lb, ub = result

        return lb, ub

    @abstractmethod
    def copy(self) -> "Canonicalizer":
        """
        Create an independent copy of this canonicalizer.

        The copy shares no mutable state with the original and can be
        modified (e.g., adding objectives) without affecting the original.

        Returns
        -------
        Canonicalizer
            A new canonicalizer instance with copied state.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError
