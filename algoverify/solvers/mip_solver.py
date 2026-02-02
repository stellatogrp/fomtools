from functools import cached_property

import numpy as np
import scipy.sparse as spa

from algoverify.solvers.solver import Solver
from algoverify.utils.convex_hull_cuts import create_new_constr, get_cuts


class MIPSolver(Solver):
    """
    Abstract base class for Mixed-Integer Programming (MIP) solvers.

    This class extends the base Solver class to provide MIP-specific functionality,
    including model relaxation and convex hull cut generation. It serves as the
    parent class for concrete MIP solver implementations like GurobiSolver and
    HiGHSSolver.

    The MIPSolver introduces methods for:
    - Relaxing integer variables to continuous variables
    - Retrieving relaxed variable values
    - Adding convex hull cuts to tighten the relaxation

    Subclasses must implement the abstract methods `relax_model` and
    `get_relaxed_variables` according to their specific solver backend.

    Parameters
    ----------
    solver_params : dict, optional
        Solver-specific parameters and configuration options.

    Attributes
    ----------
    relaxed_model : cached_property
        The relaxed version of the MIP model with all integrality restrictions,
        SOS conditions, and general constraints removed.
    """

    def __init__(self, solver_params=None):
        super(MIPSolver, self).__init__(solver_params=solver_params)

    def relax_model(self):
        """
        Relax the MIP model to a continuous linear program.

        This method creates a relaxed version of the model where all integer
        variables are converted to continuous variables, and special ordered
        set (SOS) constraints and general constraints are removed.

        Returns
        -------
        Model
            The relaxed model object (solver-specific).

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_relaxed_variables(self, var):
        """
        Get the relaxed variables corresponding to the given expression.

        This method retrieves the continuous (relaxed) variables and their
        current solution values from the relaxed model.

        Parameters
        ----------
        var : LinearExpression
            The linear expression whose relaxed variables should be retrieved.

        Returns
        -------
        tuple
            A tuple (variables, values) where:
            - variables: The relaxed solver variables
            - values: The current solution values of those variables

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    @cached_property
    def relaxed_model(self):
        """
        Return the relaxed version of the MIP model, in which all integrality
        restrictions, SOS conditions, semi-continuous and semi-integer
        requirements on variables have been relaxed and general constraints
        have been removed.
        """
        return self._relaxed_model

    def add_convex_hull_cuts(self, steps):
        """
        Add convex hull cuts to tighten the MIP relaxation.

        This method generates and adds cutting planes based on the convex hull
        of the feasible region defined by the step operators (e.g., ReLU, soft
        threshold). These cuts strengthen the linear programming relaxation and
        can improve solution quality.

        The method relaxes the model, then for each step it extracts variable
        values from the relaxed solution and generates cuts using the algorithm
        from Huchette & Vielma for piecewise linear functions.

        Parameters
        ----------
        steps : list of Step
            List of step objects (e.g., ReluStep, SoftThresholdStep) for which
            to generate convex hull cuts.
        """
        self.relax_model()

        for step in steps:
            y_var, y_val = self.get_relaxed_variables(step.rhs)
            w_var, w_val = self.get_relaxed_variables(step.lhs)

            l = np.concatenate(step.rhs.lb)
            u = np.concatenate(step.rhs.ub)

            for i in range(step.lhs.ndim):
                # special case out sparse matrices as needed
                A = np.concatenate(
                    [
                        np.asarray(v.todense()).reshape(-1) if spa.issparse(v) else v.flatten()
                        for v in step.rhs[i].decomposition_dict.values()
                    ]
                )

                # TODO: Gurobi is leaked in this part
                Iint, lI, h, L_hat, U_hat = get_cuts(step, i, w_val[i], A, y_val, l, u)
                if Iint is not None:
                    self.add_constr(
                        create_new_constr(w_var[i], A, y_var, Iint, lI, h, L_hat, U_hat)
                    )
