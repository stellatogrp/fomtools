import numpy as np

from algoverify import LinearExpression
from algoverify.canonicalizers.canonicalizer import Canonicalizer
from algoverify.constraint import Constraint


class MIP(Canonicalizer):
    """
    An interface for mixed-integer programming (MIP) methods.

    Attributes
    ----------
    solver: Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).

    Methods
    -------
    add_var(var, lb, ub)
        ...

    """

    def __init__(self, solver_params=None, solver="gurobi"):
        super(MIP, self).__init__(solver_params, solver)

    def add_var(self, var: LinearExpression) -> None:
        """
        Add a variable to the MIP instance. This method registers
        a new variable with the MIP program and attaches it to the
        internal model so it can be referenced in the constraints.

        Parameters
        ----------
        var: LinearExpression
            Linear expression object to add to the MIP instance. Later to
            be used in objective function, constraints or parameter.

        Returns
        -------
        None
        """
        self.solver.add_var(var=var, lb=var.lb, ub=var.ub)

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint to the MIP instance.This method registers
        a new constraint with the MIP problem and attaches it to the
        internal model.

        Parameters
        ----------
        constraint: Constraint
             Constraint object to add to the MIP instance.

        Returns
        -------
        None
        """
        self.solver.add_constr(constraint)

    def add_constraints(self, constraints: Constraint) -> None:
        self.solver.constraints_to_solver(constraints)

    def add_theory_cut(self, C, target_expr, bound_lb, bound_ub):
        if C != np.inf:
            self.solver.constraints_to_solver(
                [target_expr >= bound_lb - C, target_expr <= bound_ub + C]
            )

    def post_process(self, target_expr, bound_lb, bound_ub):
        self.solver.update_bounds(target_expr, bound_lb, bound_ub)

    def set_zero_objective(self):
        # this function is mostly just for debugging the constraints
        self.solver.set_objective(expression=0)

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
        constraints = []
        M = np.maximum(np.max(np.abs(np.hstack(ub_list))), np.max(np.abs(np.hstack(lb_list))))
        # q = self.solver.get_var(ub=M)
        q = LinearExpression.from_int(ndim=1, ub=M)

        all_gammas = []

        for expr, lb, ub in zip(expr_list, lb_list, ub_list):
            n = expr.get_output_dim()
            up = LinearExpression.from_int(ndim=n, ub=np.abs(ub))
            un = LinearExpression.from_int(ndim=n, ub=np.abs(lb))
            constraints.append(up - un == expr)

            for i in range(n):
                if lb[i] >= 0:
                    constraints.append(un[i] == 0)
                elif ub[i] < 0:
                    constraints.append(up[i] == 0)
                else:
                    vobji = LinearExpression.from_int(ndim=1, is_binary=True)
                    constraints.extend(
                        [up[i] <= np.abs(ub[i]) * vobji, un[i] <= np.abs(lb[i]) * (1 - vobji)]
                    )

            gamma = LinearExpression.from_int(ndim=n, is_binary=True)
            for i in range(n):
                constraints.extend([q >= up[i] + un[i], q <= up[i] + un[i] + M * (1 - gamma[i])])

            all_gammas.append(gamma)
        # need 1 gamma across all vectors in the list
        gamma_constr = 0
        for gamma in all_gammas:
            gamma_constr += sum([i for i in gamma])
        constraints.append(gamma_constr == 1)

        self.solver.constraints_to_solver(constraints)
        self.solver.set_objective(expression=q, sense="maximize")

    def solve_model(self, steps, **kwargs):
        if kwargs.get("huchette_cuts", False):
            self.solver.add_huchette_cuts(steps)

        self.solver.optimize()
        return self.solver["objVal"]

    def get_solution_value(self, iterate):
        return self.solver.get_solution_value(iterate)
