from typing import List, Tuple

import highspy as hp
import numpy as np
import scipy.sparse as spa
from highspy import ObjSense

from fomtools import LinearExpression
from fomtools.constraint import Constraint
from fomtools.solvers.mip_solver import MIPSolver


class HiGHSSolver(MIPSolver):
    def __init__(self, solver_params=None):
        # Variable to keep track of number of highs variable created
        self._var_counter = -1
        self.model = hp.Highs()
        super().__init__(solver_params)

    def __getitem__(self, item):
        val = {"objVal": self.model.getInfo().objective_function_value}
        return val[item]

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
        param_to_highs = {
            "TimeLimit": "time_limit",
            "MIPGap": "mip_rel_gap",
            "OutputFlag": "output_flat",
        }
        self.model.setOptionValue(param_to_highs[param], val)

    def relax_model(self):
        """
        Build and solve the continuous relaxation of the current
        model.

        This method creates a HiGHS model in which all integrality
        constraints are relaxed to continuous domains and optimizes
        it. It then clears any cached ``relaxed_model`` property by
        so that future accesses re-expose the freshly computed
        object.
        """
        model_copy = self.model.getModel()
        self._relaxed_model = hp.Highs()
        self._relaxed_model.passModel(model_copy)

        for i in range(self._relaxed_model.getNumCol()):
            self._relaxed_model.changeColIntegrality(i, hp.HighsVarType.kContinuous)

        self._relaxed_model.optimize()

        self.__dict__.pop("relaxed_model", None)

    def add_var(self, var: LinearExpression, lb: float = -np.inf, ub: float = np.inf):
        """
        Create and register a list of HiGHS variable corresponding to the
        given ``LinearExpression``.

        This method creates a list of  HiGHS variables using  ``hp.Highs().addIntegral``
        or ``hp.Highs().addVariable`` to create one or more decision variables whose
        shape matches that of provided ``LinearExpression``. It also records
        the mapping between the ``LinearExpression`` object and its associated list of
        HiGHS variable for subsequent reference. The created variable(s) are of specified
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

        def _create_single_var(_lb, _ub, is_binary):
            """Create a HiGHS variable using the specified bounds and integrality."""
            self._var_counter += 1
            if is_binary:
                return self.model.addIntegral(lb=0, ub=1, name=f"v{self._var_counter}")
            else:
                return self.model.addVariable(lb=_lb, ub=_ub, name=f"v{self._var_counter}")

        lb = np.full(var.ndim, lb)
        ub = np.full(var.ndim, ub)
        for k, v in var.decomposition_dict.items():
            self.vector_var_map[k] = [
                _create_single_var(lb[i], ub[i], k.is_binary) for i in range(k.ndim)
            ]

    def expression_to_solver(
        self, expression: LinearExpression
    ) -> np.ndarray[hp.highs.highs_linear_expression]:
        """
        Convert a ``LinearExpression`` into a list of HiGHS  ``highs_linear_expression``.

        Given a ``LinearExpression`` object with a decomposition of the form
        ``expression.decomposition_dict = {key_1: matrix_1, ..., key_n: matrix_n}``,
        this method constructs the equivalent solver-side representation by multiplying each
        coefficient ``matrix_i`` with its corresponding HiGHS variable and
        summing the results:

        $$
            \sum_{i}^{N} matrix_i @ variable_i
        $$

        The resulting object is a numpy arary of HiGHS linear expressions with the same
        dimensinoality as the input ``expression``, the elements of which can be
        used in model constraints or objective functions.

        Parameters
        ----------
        expression: LinearExpression
            The LinearExpression object to be converted into its corresponding Gurobi form.

        Returns
        -------
        np.ndarray[hp.highs.highs_linear_expression]
            An array of HiGHS linear expression corresponding to the input ``expression``,
            representing its solver-side linear form.
        """
        highs_var = np.zeros(expression.ndim)
        for k, v in expression.decomposition_dict.items():
            v = v.toarray() if spa.issparse(v) else np.array(v)
            if self.vector_var_map.get(k, None) is None:
                self.add_var(k)

            highs_var = highs_var + v @ self.vector_var_map[k]

        return highs_var + expression.const

    def constraints_to_solver(self, constraints: List[Constraint]) -> None:
        """
        Add multiple constraints to the HiGHS model.

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
        for i, constr in enumerate(constraints):
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
        for i, var in enumerate(self.expression_to_solver(expression)):
            var.bounds = (lb[i], ub[i])

    def add_constr(self, constraint: Constraint | hp.highs.highs_linear_expression):
        """
        Add a single constraint to the HiGHS model.

        Parameters
        ----------
        constraint: Constraint | hp.highs.highs_linear_expression
            A HiGHS constraint expression compatible. Typical form includes
            comparisons like ``lhs == rhs``, ``lhs <= rhs``, where ``lhs``,
            and ``rhs`` are of type ``LinearExpression`` or ``hp.highs.highs_linear_expression``.

        Returns
        -------
        None
        """
        try:
            self.model.addConstr(constraint)
        except AttributeError:
            # Expression on the right hand side of the constraint object is always 0
            for i, _lhs in enumerate(self.expression_to_solver(constraint.lhs)):
                match constraint.operator:
                    case "=":
                        self.model.addConstr(_lhs == 0)
                    case "<=":
                        self.model.addConstr(_lhs <= 0)

    def set_objective(self, expression: LinearExpression, sense: str = None) -> None:
        """
        Set the optimization objective for the underlying HiGHS model.

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
                sense = ObjSense.kMinimize
            case "maximize":
                sense = ObjSense.kMaximize
            case None:
                sense = None
            case _:
                raise TypeError(f"Objective sense must be either minimize or maximize, got {sense}")

        try:
            if int(expression) != 0:
                raise ValueError(
                    f"Objective function must be either a LinearExpression object or a 0, got {expression}"
                )
            for i in range(self.model.numVariables):
                self.model.changeColCost(i, 0)
        except TypeError:
            self.model.setObjective(self.expression_to_solver(expression)[0], sense=sense)

    def set_two_norm_objective(self, expr_list, lb_list, ub_list) -> None:
        """
        Set two-norm objective (not supported for HiGHS).

        HiGHS does not support quadratic objectives (MIQP).

        Raises
        ------
        NotImplementedError
            Always raised since HiGHS does not support quadratic objectives.
        """
        raise NotImplementedError(
            "HiGHS does not support quadratic objectives (MIQP). "
            "Use Gurobi for two-norm objectives, or switch to one-norm/infinity-norm."
        )

    def add_sum_squares_constr(self, expr, radius_sq, sense) -> None:
        """
        Add a sum-of-squares (quadratic) constraint (not supported for HiGHS).

        HiGHS does not support quadratic constraints (MIQCP). This backstop mirrors
        :meth:`set_two_norm_objective`; the MIP canonicalizer guards on the solver name
        before reaching here, so this is normally unreached.

        Raises
        ------
        NotImplementedError
            Always raised since HiGHS does not support quadratic constraints.
        """
        raise NotImplementedError(
            "HiGHS does not support quadratic constraints (MIQCP). "
            "Use Gurobi for sum_squares constraints."
        )

    def optimize(self):
        """
        Optimize the underlying HiGHS model.

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
            out += v @ np.array([self.model.val(i) for i in self.vector_var_map[k]])
        return out

    def get_relaxed_variables(
        self, expression: LinearExpression
    ) -> Tuple[list[hp.highs.highs_var], np.ndarray]:
        """
        Reconstruct solver-side variables from a ``LinearExpression`` object
        using the relaxed HiGHS model.

        Parameters
        ----------
        expression: LinearExpression
            A ``LinearExpression`` object registered with the solver. Each element in
            its decomposition must have a corresponding entry in ``self.vector_var_map``.

        Returns
        -------
        Tuple[list[hp.highs.highs_var], np.ndarray]
            A list of HiGHS variable formed by stacking the individual
            ``hp.highs.highs_var`` objects corresponding to each element of
            the input ``LinearExpression``, and a numpy array containing
            the current value of those stacked variables.
        """
        # If no relaxed model currently exists, build and solve it first.
        if self.relaxed_model is None:
            self.relax_model()

        highs_var, values = [], []
        for k in expression.decomposition_dict.keys():
            _hp_var = self.vector_var_map[k]
            highs_var.extend(_hp_var)
            values.extend(self.relaxed_model.variableValues(_hp_var))

        return highs_var, np.array(values)

    def is_optimal(self) -> bool:
        """
        Check whether the underlying HiGHS model has reached an optimal solution.
        """
        return self.model.getModelStatus() == hp.HighsModelStatus.kOptimal

    def copy(self) -> "HiGHSSolver":
        """
        Create an independent copy of this HiGHS solver.

        Returns a new HiGHSSolver with a copied model and rebuilt
        vector_var_map pointing to variables in the copied model.

        Returns
        -------
        HiGHSSolver
            A new solver instance with copied state.
        """
        new_solver = HiGHSSolver.__new__(HiGHSSolver)
        new_solver._relaxed_model = None
        new_solver.model = hp.Highs()
        new_solver._var_counter = self._var_counter

        # Copy the model data
        model_data = self.model.getModel()
        new_solver.model.passModel(model_data)

        # Build a name-to-variable map for the new model
        new_vars = new_solver.model.getVariables()
        name_to_var = {v.name: v for v in new_vars}

        # Rebuild vector_var_map using variable names
        new_solver.vector_var_map = {}
        for k, var_list in self.vector_var_map.items():
            new_solver.vector_var_map[k] = [name_to_var[v.name] for v in var_list]

        return new_solver
