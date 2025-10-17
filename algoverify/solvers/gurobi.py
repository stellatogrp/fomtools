import logging

import gurobipy as gp
import numpy as np

from algoverify import LinearExpression
from algoverify.solvers.solver import Solver

log = logging.getLogger(__name__)


class GurobiSolver(Solver):
    """
    A wrapper around the Gurobi ``Model`` that bridges ``LinearExpression``
    and ``Constrain`` objects to solver-ready variables, expressions, and
    constraints.

    Attributes
    ----------
    model: gp.Model
        The underlying Gurobi model instance.

    ... 

    """

    def __init__(self, solver_params={}):
        """
        Initialize an instance of the `GurobiSolver` class.

        Parameters
        ----------
        solver_params
        """
        self.vector_var_map = {}
        self.model = gp.Model()

        for key, val in solver_params.items():
            self.model.setParam(key, val)


    def _get_vtype(self, vtype: str) -> str:
        """
        Return the variable type object corresponding to the given string name.

        Parameters
        ----------
        vtype: str
            The type of variable specified as a string. Supported values include:
            - ``"continuous"``
            - ``"integer"``
            - ``"binary"``

        Returns
        -------
        str
            The corresponding variable type object associated with the given
            variable name.

        Raises
        ------
        ValueError
            If the provided variable type string is not recognized.
        """
        match vtype:
            case "continuous":
                return gp.GRB.CONTINUOUS
            case "integer":
                return gp.GRB.INTEGER
            case "binary":
                return gp.GRB.BINARY
            case _:
                raise TypeError(f"Variable type must be either continuous or binary, got {vtype}")

    def get_var(self, lb: float = -np.inf, ub: float = np.inf, vtype: str = "continuous") -> gp.Var:
        """
        Create and add a decision variable to the underlying Gurobi model.

        This method wraps ``model.addVar`` to create a single variable with the
        specified bounds and type, and registers it with the model for use in
        objective or constraints.

        Parameters
        ----------
        lb: float, default=-np.inf
            Lower bound of the variable. Default is -inf.

        ub: float, default=np.inf
            Upper bound of the variable. Default is +inf.

        vtype: str, default="continuous
            The type of variable specified as a string. Supported values include:
            - ``"continuous"``
            - ``"integer"``
            - ``"binary"``

        Returns
        -------
        gp.Var
            The newly created Gurobi variable attached to the underlying model. (The
            variable is not automatically added to any objective or constraints.)

        Raises
        ------
        ValueError
            If the provided variable type string is not recognized.
        """
        return self.model.addVar(lb=lb, ub=ub, vtype=self._get_vtype(vtype))

    def get_matrix_var(
        self,
        shape: int | tuple[int, int],
        lb: float = -np.inf,
        ub: float = np.inf,
        vtype: str = "continuous",
    ) -> gp.MVar:
        """
        Create and add a multi-dimensional decision variable to the underlying Gurobi model.

        This method wraps ``model.addMVar`` to construct a matrix variable of shape
        ``shape`` with the specified bounds and type, and registers it with the
        model for use in objective or constraints.

        Parameters
        ----------
        shape: int or tuple[int,int]
            Number of dimensions of the variable.

        lb: float, default=-np.inf
            Lower bound for all the elements of the matrix variable.

        ub: float, default=np.inf
            Upper bound for all the elements of the matrix variable.

        vtype: str, default="continuous
            The type of variable specified as a string. Supported values include:
            - ``"continuous"``
            - ``"integer"``
            - ``"binary"``

        Returns
        -------
        gp.Var
            The newly created Gurobi variable attached to the underlying model. (The
            variable is not automatically added to any objective or constraints.)

        Raises
        ------
        ValueError
            If the provided variable type string is not recognized.
        """
        return self.model.addMVar(shape, lb=lb, ub=ub, vtype=self._get_vtype(vtype))

    def update_model(self):
        """
        Synchronize pending changes with the underlying Gurobi model.

        Calls ``model.update()`` to update any deffered modification (e.g.,
        newly added variables, constraints or parameters) so they become
        visible to the model queries and attribute reads.
        """
        self.model.update()

    def add_var(self, var: LinearExpression, lb: float = -np.inf, ub: float = np.inf) -> None:
        """
        Create and register a Gurobi matrix variable corresponding to the
        given ``LinearExpression``.


        This method allocates a Gurobi ``MVar`` whose shape matches that of the
        provided ``LinearExpression``, and stores the mapping between the
        ``LinearExpression`` object and its associated Gurobi variable for
        subsequent reference.

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
        self.vector_var_map[var] = self.get_matrix_var(var.ndim, lb=lb, ub=ub)
        self.update_model()

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
            gp_expr += v @ self.vector_var_map[k]

        return gp_expr

    def constraints_to_solver(self, constraints):
        """
        Add multiple constraints to the Gurobi model.

        Parameters
        ----------
        constraints: list[gp.TempConstr]
            A list of Gurobi constraint expressions, typically of the form ``lhs == rhs``,
            ``lhs <= rhs``, or ``lhs >= rhs`` where ``lhs`` and ``rhs`` are of type ``gp.MLinExpr``.

        Returns
        -------
        None

        """
        for constr in constraints:
            self.add_constr(constr)

    def add_constr(self, constraint: gp.TempConstr) -> None:
        """
        Add a single constraint to the Gurobi model.

        Parameters
        ----------
        constraint: gp.TempConstr
            A Gurobi constraint expression compatible. Typical form includes
            comparisons like ``lhs == rhs``, ``lhs <= rhs``, where ``lhs``,
            and ``rhs`` are of type ``gp.MLinExpr``.

        Returns
        -------
        None
        """
        self.model.addConstr(constraint)
        self.update_model()

    def set_objective(self, expression, type=None):
        """
        Set the optimization objective for the underlying Gurobi model.

        Parameters
        ----------
        expression: gp.LinExpr | gp.QuadExpr
            A Gurobi-compatible objective expression.

        type: {"minimize", "maximize"}
            Objective sense specified as a string.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If provided objective ``type`` is not ``"minimize"`` or ``"maximize"``.

        """
        match type:
            case "minimize":
                type = gp.GRB.MINIMIZE
            case "maximize":
                type = gp.GRB.MAXIMIZE
            case None:
                type = None
            case _:
                raise TypeError(f"Objective type must be either minimize or maximize, got {type}")
        self.model.setObjective(expression, sense=type)
        self.update_model()

    def relax_model(self):
        """
        Return the relaxed version of the MIP model, in which all integrality
        restrictions, SOS conditions, semi-continuous and semi-integer
        requirements on variables have been relaxed and general constraints
        have been removed.
        """
        return self.model.relax()

    def optimize(self):
        """
        Optimize the underlying Gurobi model.

        Notes
        -----
        The algorithm used depends on the model type and
        on the parameter settings
        """
        self.model.optimize()

    def get_solution_value(self, var_name):
        """
        Return the solution value for the given variable.
        """
        return self.model.getVarByName(var_name.VarName.item()).X

    def extract_vec_var_by_name(self, var: gp.MVar, model: gp.Model) -> gp.Var:
        """
        Construct a Gurobi variable by looking up scalar variables by name.

        Given a 1-D array-like ``var`` whose elements provide variable names
        (either as plain strings or as objects exposing a ``VarName`` attribute),
        this function retrieves the corresponding scalar variables from ``model``
        and assembles them into a single ``gp.MVar``.

        Parameters
        ----------
        var: gp.MVar
            A 1-D container in which each element is a scalar Gurobi variable.

        model: gp.Model
            The Gurobi model from which to extract the variables.

        Returns
        -------
        gp.MVar
            A Gurobi matrix variable composed of the scalar variables in
            var.
        """
        return gp.MVar.fromlist(
            [model.getVarByName(var[i].item().VarName) for i in range(var.shape[0])]
        )

    def get_solve_data(self):
        return {
            "status": self.model.status,
            "objVal": self.model.ObjVal,
            "objBound": self.model.objBound,
            "MIPGap": self.model.MIPGap,
            "Runtime": self.model.Runtime,
            "rel_LP_sol": self.rel_LP_sol,
            "numBinVars": self.model.NumBinVars,
        }
