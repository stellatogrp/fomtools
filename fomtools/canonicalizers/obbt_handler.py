import logging

import numpy as np

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import Constraint
from fomtools.linearexpression import LinearExpression

log = logging.getLogger(__name__)


class OBBTHandler(Canonicalizer):
    """
    Handler for Optimization-Based Bound Tightening (OBBT).

    This class provides an interface for tightening variable bounds using
    optimization. It maintains a separate solver instance that is used to
    compute tighter lower and upper bounds on variables by solving
    minimization and maximization problems.

    Attributes
    ----------
    solver : Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).
    """

    def __init__(self, solver: str):
        """
        Initialize an instance of :class:`OBBTHandler`.

        Parameters
        ----------
        solver : str
            Case-insensitive identifier for the solver to use. Supported
            values (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
            - ``"highs"``
        """
        super().__init__(solver=solver)

    def add_var(self, var: LinearExpression) -> None:
        """
        Add a variable to the OBBT handler instance.

        This method registers a new variable with the OBBT solver so it
        can be used in bound tightening optimization problems.

        Parameters
        ----------
        var : LinearExpression
            Linear expression object to add to the OBBT instance.

        Returns
        -------
        None
        """
        self.solver.add_var(var, var.lb, var.ub)

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint to the OBBT handler instance.

        This method registers a constraint with the OBBT solver so it
        is considered during bound tightening optimization.

        Parameters
        ----------
        constraint : Constraint
            Constraint object to add to the OBBT instance.

        Returns
        -------
        None
        """
        self.solver.add_constr(constraint)

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

        Returns
        -------
        None
        """
        self.solver.update_bounds(expression, lb, ub)

    def obbt(
        self, linexpr: LinearExpression, lb: np.ndarray, ub: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Perform Optimization-Based Bound Tightening on a linear expression.

        For each element in the linear expression, this method solves
        minimization and maximization problems to compute potentially
        tighter bounds. The returned bounds are the intersection of the
        input bounds and the computed bounds.

        Parameters
        ----------
        linexpr : LinearExpression
            The linear expression whose bounds are to be tightened.

        lb : np.ndarray
            Current lower bounds for each element of the expression.

        ub : np.ndarray
            Current upper bounds for each element of the expression.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            A tuple ``(new_lb, new_ub)`` containing the tightened lower
            and upper bounds. Returns ``None`` if the optimization failed
            (e.g., the problem is infeasible or unbounded).
        """
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
