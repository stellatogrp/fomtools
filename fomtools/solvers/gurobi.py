import logging
from typing import List, Tuple

import gurobipy as gp
import numpy as np

from fomtools import LinearExpression
from fomtools.constraint import Constraint
from fomtools.solvers.mip_solver import MIPSolver

log = logging.getLogger(__name__)


class GurobiSolver(MIPSolver):
    """
    A wrapper around the Gurobi ``Model`` that bridges ``LinearExpression``
    and ``Constraint`` objects to solver-ready variables, expressions, and
    constraints.

    Attributes
    ----------
    model: gp.Model
        The underlying Gurobi model instance.

    ...

    """

    def __init__(self, solver_params=None):
        """
        Initialize an instance of the `GurobiSolver` class.

        Parameters
        ----------
        solver_params
        """
        self.model = gp.Model()
        super().__init__(solver_params)

    def __getitem__(self, item):
        return getattr(self.model, item)

    def set_param(self, param: str, val: int | float):
        """
        Configure a solver parameter that controls its behavior.

        Parameters
        ----------
        param: str
            The name of the solver parameter to set.

        val: int | float
            The value to assign to the specified parameter.
        """
        self.model.setParam(param, val)

    def relax_model(self):
        """
        Build and solve the continuous relaxation of the current
        model.

        This method creates a Gurobi model in which all integrality
        constraints are relaxed to continuous domains and optimizes
        it. It then clears any cached ``relaxed_model`` property by
        so that future accesses re-expose the freshly computed
        object.
        """
        self._relaxed_model = self.model.relax()
        self._relaxed_model.optimize()

        self.__dict__.pop("relaxed_model", None)

    def add_var(self, var: LinearExpression, lb: float = -np.inf, ub: float = np.inf) -> gp.Var:
        """
        Create and register a Gurobi matrix variable corresponding to the
        given ``LinearExpression``.

        This method wraps ``model.addVar`` to create one or more decision variables
        whose shape matches that of provided ``LinearExpression``. It also records
        the mapping between the ``LinearExpression`` object and its associated Gurobi
        variable for subsequent reference. The created variable(s) are of specified
        bounds and types.

        Parameters
        ----------
        var: LinearExpression
            LinearExpression object to be registered.

        lb: float, default=-np.inf
            Lower bound for all the elements of the matrix variable.

        ub: float, default=np.inf
            Upper bound for all the elements of the matrix variable.

        Returns
        -------
        None
        """
        for k in var.decomposition_dict.keys():
            self.vector_var_map[k] = self.model.addMVar(
                var.ndim, lb=lb, ub=ub, vtype=gp.GRB.BINARY if var.is_binary else gp.GRB.CONTINUOUS
            )
        self.model.update()

    def add_param_stack_var(
        self, param_stack_var: LinearExpression, param_list: list[LinearExpression]
    ) -> None:
        """
        Create a stacked Gurobi ``MVar`` by horizontally concatenating
        existing parameter variables.

        Registers ``param_stack_var`` in ``self.vector_var_map`` as the
        horizontal stack of the variables associated with each element of
        ``param_list``.  This is useful to building block-structured vectors
        from previously created components.

        Parameters
        ----------
        param_stack_var: LinearExpression
            LinearExpression object to be registered.

        param_list: list[LinearExpression]
            List of LinearExpression objects whose associated Gurobi variables are to be stacked.
            Each entry must already exist in ``self.vector_var_map`` and map to a compatible
            ``gp.Var`` or ``gp.MVar`` object.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any element of ``param_list`` is not already registered to the model.

        """
        try:
            self.vector_var_map[param_stack_var] = gp.hstack(
                [self.vector_var_map[param] for param in param_list]
            )
        except KeyError:
            raise KeyError(
                """One or more elements in 'param_list' are not registered with the Gurobi model.
                Ensure that all parameters in 'param_list' have been added to 'vector_var_map' before stacking."""
            )

    def leaf_expr_to_solver(self, expression: LinearExpression) -> LinearExpression:
        """
        Resolve a leaf ``LinearExpression`` into its solver-side Gurobi variable.

        Parameters
        ----------
        expression: LinearExpression
            A leaf ``LinearExpression`` object. with a single entry in
            ``expression.decomposition_dict``.

        Returns
        -------
        gp.Var or gp.MVar
            The Gurobi variable corresponding to the leaf ``expression``'s sole key.

        Raises
        ------
        TypeError
            If ``expression`` is not a leaf ``LinearExpression``.
        """
        if expression.is_leaf:
            return self.vector_var_map[next(iter(expression.decomposition_dict))]

        raise TypeError(
            "Invalid input: expression must be a leaf. Received a composite one instead."
        )

    def expression_to_solver(self, expression: LinearExpression) -> gp.MLinExpr:
        """
        Convert a ``LinearExpression`` into a Gurobi ``MLinExpr``.

        Given a ``LinearExpression`` object with a decomposition of the form
        ``expression.decomposition_dict = {key_1: matrix_1, ..., key_n: matrix_n}``,
        this method constructs the equivalent solver-side representation by multiplying each
        coefficient ``matrix_i`` with its corresponding Gurobi variable and
        summing the results:

        $$
            \sum_{i}^{N} matrix_i @ variable_i
        $$

        The resulting object is a Gurobi linear expressions with the same dimensinoality as
        the input ``expression``, and can be directly used in model constraints or
        objective functions.

        Parameters
        ----------
        expression: LinearExpression
            The LinearExpression object to be converted into its corresponding Gurobi form.

        Returns
        -------
        gp.MLinExpr
            The Gurobi linear expression corresponding to the input ``expression``, representing
            its solver-ide linear form.
        """
        gp_expr = np.zeros(expression.ndim)
        for k, v in expression.decomposition_dict.items():
            if self.vector_var_map.get(k, None) is None:
                self.add_var(k)
            gp_expr += v @ self.vector_var_map[k]

        return gp_expr + expression.const

    def constraints_to_solver(self, constraints: List[Constraint]):
        """
        Add multiple constraints to the Gurobi model.

        Parameters
        ----------
        constraints: list[Constraint]
            A list of constraint expressions, typically of the form ``lhs == rhs``,
            ``lhs <= rhs``, or ``lhs >= rhs`` where ``lhs`` and ``rhs`` are of type
            ``LinearExpression``.

        Returns
        -------
        None

        """
        for constr in constraints:
            self.add_constr(constr)

    def update_bounds(
        self, expression: LinearExpression, lb: float | np.ndarray, ub: float | np.ndarray
    ):
        """
        Update the lower and upper bounds of the solver variables corresponding
        to the given ``LinearExpression``.

        Parameters
        ----------
        expression: LinearExpression
            A ``LinearExpression`` object registered with the solver whose
            underlying variables bound are to be updated.

        lb: float or np.ndarray
            New lower bound(s).

        ub: float or np.ndarray
            New upper bound(s).

        """
        var = self.expression_to_solver(expression)
        var.LB = lb
        var.UB = ub

    def add_constr(self, constraint) -> None:
        """
        Add a single constraint to the Gurobi model.

        Parameters
        ----------
        constraint: Constraint | gp.TempConstr
            A Gurobi constraint expression compatible. Typical form includes
            comparisons like ``lhs == rhs``, ``lhs <= rhs``, where ``lhs``,
            and ``rhs`` are of type ``LinearExpression`` or ``gp.MLinExpr``.

        Returns
        -------
        None
        """
        try:
            self.model.addConstr(constraint)
        except gp.GurobiError:
            # Expression on the right hand side of the constraint object is always 0
            lhs = self.expression_to_solver(constraint.lhs)
            match constraint.operator:
                case "=":
                    self.model.addConstr(lhs == 0)
                case "<=":
                    self.model.addConstr(lhs <= 0)
            self.model.update()

    def set_objective(self, expression: LinearExpression, sense: str = None) -> None:
        """
        Set the optimization objective for the underlying Gurobi model.

        Parameters
        ----------
        expression: LinearExpression or 0
            A ``LinearExpression`` object registered with the solver repesenting the
            optimization objective, or 0 if setting a zero objective.

        sense: {"minimize", "maximize"}
            Objective sense specified as a string.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If provided objective ``sense`` is not ``"minimize"`` or ``"maximize"``.

        ValueError
            If provided expression is anything other than a ``LinearExpression`` object or a 0.

        """
        match sense:
            case "minimize":
                sense = gp.GRB.MINIMIZE
            case "maximize":
                sense = gp.GRB.MAXIMIZE
            case None:
                sense = None
            case _:
                raise TypeError(f"Objective sense must be either minimize or maximize, got {sense}")

        try:
            if int(expression) != 0:
                raise ValueError(
                    f"Objective function must be either a LinearExpression object or a 0, got {expression}"
                )

            self.model.setObjective(expression, sense=sense)
        except TypeError:
            self.model.setObjective(self.expression_to_solver(expression), sense=sense)
        self.model.update()

    def add_sum_squares_constr(self, expr, radius_sq, sense) -> None:
        """
        Add a sum-of-squares (quadratic) constraint to the Gurobi model.

        Encodes ``sum_i (a_i^T x + b_i)^2  op  radius_sq`` where ``op`` is given by
        ``sense``. The squared two-norm is built as ``gp_expr @ gp_expr`` (a dot product),
        the same construct used by :meth:`set_two_norm_objective`. The const term ``b`` is
        included via :meth:`expression_to_solver`.

        The ``>=`` / ``==`` senses are nonconvex quadratic constraints; ``NonConvex=2`` is
        set for them so Gurobi accepts the model across versions (the convex ``<=`` sense
        does not need it).

        Parameters
        ----------
        expr : LinearExpression
            The expression whose squared two-norm is constrained.
        radius_sq : float
            The nonnegative scalar right-hand side (e.g. ``r**2``).
        sense : str {"<=", ">=", "=="}
            The relational operator.
        """
        gp_expr = self.expression_to_solver(expr)
        ss = gp_expr @ gp_expr  # 0-d MQuadExpr = sum of squares (matrix API)
        if sense != "<=":
            self.model.Params.NonConvex = 2
        # The matrix-API MQuadExpr is added via addConstr (addQConstr rejects the 0-d MQuadExpr).
        match sense:
            case "<=":
                self.model.addConstr(ss <= radius_sq)
            case ">=":
                self.model.addConstr(ss >= radius_sq)
            case "==":
                self.model.addConstr(ss == radius_sq)
        self.model.update()

    def set_two_norm_objective(self, expr_list, lb_list, ub_list) -> None:
        """
        Set a quadratic two-norm objective for the underlying Gurobi model.

        The objective is to maximize sum_i ||expr_i||_2^2 = sum_i (expr_i @ expr_i).

        Parameters
        ----------
        expr_list : list[LinearExpression]
            List of expressions whose squared norms are summed in the objective.
        lb_list : list[np.ndarray]
            Lower bounds for each expression (currently unused but passed for interface consistency).
        ub_list : list[np.ndarray]
            Upper bounds for each expression (currently unused but passed for interface consistency).

        Returns
        -------
        None
        """
        quadratic_obj = 0
        for expr in expr_list:
            gp_expr = self.expression_to_solver(expr)
            quadratic_obj = quadratic_obj + gp_expr @ gp_expr

        self.model.setObjective(quadratic_obj, sense=gp.GRB.MAXIMIZE)
        self.model.update()

    def optimize(self):
        """
        Optimize the underlying Gurobi model.

        Notes
        -----
        The algorithm used depends on the model objective and
        on the parameter settings
        """
        self.model.optimize()

    def get_solution_value(self, expression: LinearExpression) -> np.ndarray:
        """
        Compute the solution value for the given ``LinearExpression``
        object.

        For each leaf expression in the decomposition of the given
        ``LinearExpression`` object, this method retrieves the corresponding
        solver variables from the model and computes the dot product between
        the coefficient matrix and the vector of variable values associated with
        that leaf.

        Parameters
        ----------
        expression: LinearExpression
            A ``LinearExpression`` object registered with the solver. Each element in
            its decomposition must have a corresponding entry in ``self.vector_var_map``.

        Returns
        -------
        np.ndarray[float]
            A 1-D array of shape ``(expression.ndim,)`` containing the evaluated values
            of the expression at the current model solution.
        """
        out = 0
        for k, v in expression.decomposition_dict.items():
            out += v @ [self.model.getVarByName(var).X for var in self.vector_var_map[k].VarName]
        return out

    def get_relaxed_variables(self, expression: LinearExpression) -> Tuple[gp.MVar, np.ndarray]:
        """
        Reconstruct solver-side variables from a ``LinearExpression`` object
        using the relaxed Gurobi model.

        Parameters
        ----------
        expression: LinearExpression
            A ``LinearExpression`` object registered with the solver. Each element in
            its decomposition must have a corresponding entry in ``self.vector_var_map``.

        Returns
        -------
        Tuple[gp.MVar, np.ndarray]
            A single Gurobi matrix variable formed by horizontally stacking the
            individual ``gp.MVar`` objects corresponding to each element of
            the input ``LinearExpression``, and a numpy array containing the
            current values of those stacked variables.
        """
        # If no relaxed model currently exists, build and solve it first.
        if self.relaxed_model is None:
            self.relax_model()

        gp_vars = []
        for k in expression.decomposition_dict.keys():
            gp_vars.append(
                gp.MVar.fromlist(
                    [self.relaxed_model.getVarByName(i) for i in self.vector_var_map[k].VarName]
                )
            )
        return gp.hstack(gp_vars), gp.hstack(gp_vars).X

    def is_optimal(self) -> bool:
        """
        Check whether the underlying Gurobi model has reached an optimal solution.
        """
        return self["status"] == gp.GRB.OPTIMAL

    def copy(self) -> "GurobiSolver":
        """
        Create an independent copy of this Gurobi solver.

        Returns a new GurobiSolver with a copied model and rebuilt
        vector_var_map pointing to variables in the copied model.

        Returns
        -------
        GurobiSolver
            A new solver instance with copied state.
        """
        new_solver = GurobiSolver.__new__(GurobiSolver)
        new_solver._relaxed_model = None
        new_solver.model = self.model.copy()

        # Rebuild vector_var_map with variables from the copied model
        new_solver.vector_var_map = {}
        for k, v in self.vector_var_map.items():
            # v is a gp.MVar - reconstruct it from variable names
            new_solver.vector_var_map[k] = gp.MVar.fromlist(
                [new_solver.model.getVarByName(name) for name in v.VarName]
            )

        return new_solver
