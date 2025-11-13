import numpy as np

from algoverify import LinearExpression
from algoverify.canonicalizers.canonicalizer import Canonicalizer
from algoverify.constraint import Constraint
from algoverify.steps import Relu


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

    add_param_stack_var()
        ...
    """

    def __init__(self, solver_params=None, solver="gurobi"):
        super(MIP, self).__init__(solver_params, solver)

    def add_var(self, var: LinearExpression, lb=-np.inf, ub=np.inf) -> None:
        """
        Add a variable to the MIP instance. This method registers
        a new variable with the MIP program and attaches it to the
        internal model so it can be referenced in the constraints.

        Parameters
        ----------
        var: LinearExpression
            Linear expression object to add to the MIP instance. Later to
            be used in objective function, constraints or parameter.

        lb: float
            Lower bound of the variable.

        ub: float
            Upper bound of the variable.

        Returns
        -------
        None
        """
        self.solver.add_var(var=var, lb=lb, ub=ub)

    def add_param_stack_var(self, param_stack_var, param_list):
        self.solver.add_param_stack_var(param_stack_var, param_list)

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

    def add_relu_constraints(self, step: Relu) -> None:
        r"""
        Construct and add the convex-hull linear constraints corresponding
        to the `Rectified Linear Unit (ReLU)` operator to the model.

        Given a ReLU step: $(\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.$

        This method derives the convex hull of the region defined by:

        $$
            \phi = \left\{ (y, w) \in \left [\underline{y}, \overline{y} \right ] \times
            \mathbb{R} | w = \textbf{ReLU}(a^{T}y) \right\},
        $$

        expresses it as a set of LinearExpression objects, and adds these constraints to
        the internal MIP model. The convex hull of the region is:


        $$
            \operatorname{conv}(\Phi) =
            \left\{
            (y, w) \in [\underline{y}, \overline{y}] \times \mathbb{R}
            \;\middle|\;
            \begin{cases}
            w = a^{\mathsf{T}} y, & \ell_J \ge 0, \\[6pt]
            w = 0, & \ell_J < 0, \\[6pt]
            (y, w) \in Q, & \text{otherwise}
            \end{cases}
            \right\},
        $$

        with

        $$
            Q =
            \left\{
            (y, w) \in \mathbb{R}^{d + p + 1}
            \;\middle|\;
            \begin{aligned}
            & w \ge a^{\mathsf{T}} y, \quad y \ge 0, \\[4pt]
            & w \le \sum_{i \in I} a_i (y_i - \ell_i^{0})
                + \frac{\ell_I}{u_o^{0} - \ell_o^{0}} (y_o - \ell_o^{0}),
                \quad \forall (I, o) \in \mathcal{I}
            \end{aligned}
            \right\},
        $$

        where:

        - vector $l^{0} \in \mathbb{R}^{d}$ with entries $l_i^{0} = \underline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \overline{y}_i$ otherwise,
        - vector $u^{0} \in \mathbb{R}^{d}$ with entries $u_i^{0} = \overline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \underline{y}_i$ otherwise,
        - Given $\textbf{I} \subseteq \textbf{J} = \left\{1, ..., n\right\}$, the lower bound is
        defined as $l_I = \sum_{i \in I}a_il_i^{0} + \sum_{i \notin I}a_iu_i^{0}$

        Parameters
        ----------
        step: Relu

        Returns
        -------

        """
        constraints = []
        new_vars = {}

        # after ranges are added, remove the range(n) and add the constraints when needed
        # make sure to enforce equality constraints between lhs and rhs for other indices
        for i, (_lhs, _rhs) in enumerate(zip(step.lhs, step.rhs)):
            if i in step.nonproj_indices:
                constraints.append(_lhs == _rhs)
            else:
                if step.ub[i] <= 0:
                    constraints.append(_lhs == 0)
                elif step.lb[i] > 0:
                    constraints.append(_lhs == _rhs)
                elif step.ub[i] > 0 or step.lb[i] <= 0:
                    if not step.relax_binary_vars:
                        new_vars[i] = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                        step.idx_with_binary_vars.add(i)
                        constraints.append(_lhs <= step.ub[i] * new_vars[i])

                    constraints.extend(
                        [
                            _lhs <= step.ub[i] / (step.ub[i] - step.lb[i]) * (_rhs - step.lb[i]),
                            _lhs >= _rhs,
                            _lhs <= _rhs - step.lb[i] * (1 - new_vars[i]),
                        ]
                    )

        self.solver.constraints_to_solver(constraints)

    def add_saturated_linear_constraints(self, step):
        constraints = []

        for i, (_lhs, _rhs) in enumerate(zip(step.lhs, step.rhs)):
            if step.lb[i] >= step.u[i]:
                constraints.append(_lhs == step.u[i])
            elif step.ub[i] <= step.l[i]:
                constraints.append(_lhs == step.l[i])
            elif step.lb[i] >= step.l[i] and step.ub[i] <= step.u[i]:
                constraints.append(_lhs == _rhs)
            else:
                if not step.relax_binary_vars:
                    w1 = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                    if step.lb[i] < step.l[i]:
                        constraints.extend(
                            [
                                _lhs <= _rhs + (step.ub[i] - step.lb[i]) * w2,
                                _rhs >= step.l[i] + (step.lb[i] - step.l[i]) * w2,
                                _rhs <= step.l[i] + (step.ub[i] - step.l[i]) * (1 - w2),
                            ]
                        )
                        if step.ub[i] > step.u[i]:
                            constraints.extend(
                                [
                                    _lhs >= step.u[i] - (step.u[i] - step.l[i]) * (1 - w1),
                                    _lhs <= step.l[i] + (step.u[i] - step.l[i]) * (1 - w2),
                                    _lhs <= _rhs + (step.ub[i] - step.lb[i]) * w2,
                                    w1 + w2 <= 1,
                                ]
                            )
                        elif step.ub[i] > step.u[i]:
                            constraints.extend(
                                [
                                    _lhs <= step.l[i] + (step.ub[i] - step.l[i]) * (1 - w2),
                                    _lhs >= step.l[i] + (step.lb[i] - step.l[i]) * w2,
                                ]
                            )

                    if step.ub[i] > step.u[i]:
                        constraints.extend(
                            [
                                _lhs >= _rhs - (step.ub[i] - step.lb[i]) * w1,
                                _rhs >= step.u[i] + (step.lb[i] - step.u[i]) * (1 - w1),
                                _rhs <= step.u[i] + (step.ub[i] - step.u[i]) * w1,
                            ]
                        )

                        if step.lb[i] > step.l[i]:
                            constraints.append(
                                _lhs >= step.u[i] - (step.u[i] - step.lb[i]) * (1 - w1)
                            )

                if step.lb[i] < step.l[i] and step.ub[i] > step.u[i]:
                    # following two are redundant due to preprocessing bounds on the left hand side var
                    expr = (step.u[i] - step.l[i]) / (step.u[i] - step.lb[i]) * (
                        _rhs - step.u[i]
                    ) + step.u[i]

                    constraints.extend([_lhs <= expr, _lhs >= expr])

                elif step.lb[i] < step.l[i] and step.ub[i] < step.u[i]:
                    constraints.extend(
                        [
                            _lhs >= _rhs,
                            _lhs
                            <= ((step.ub[i] - step.l[i]) / (step.ub[i] - step.lb[i]))
                            * (_rhs - step.ub[i])
                            + step.ub[i],
                        ]
                    )

                elif step.lb[i] > step.l[i] and step.ub[i] > step.u[i]:
                    constraints.extend(
                        [
                            _lhs <= _rhs,
                            _lhs
                            >= ((step.u[i] - step.lb[i]) / (step.ub[i] - step.lb[i]))
                            * (_rhs - step.lb[i])
                            + step.lb[i],
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

        self.solver.constraints_to_solver(constraints)

    def add_soft_threshold_constraints(self, step, lhs_lb, lhs_ub):
        constraints = []
        new_w1 = {}
        new_w2 = {}
        for i, (_lhs, _rhs) in enumerate(zip(step.lhs, step.rhs)):
            if step.lb[i] >= step.lambd:
                constraints.append(_lhs == _rhs - step.lambd)
            elif step.ub[i] <= -step.lambd:
                constraints.append(_lhs == _rhs + step.lambd)
            elif step.lb[i] >= -step.lambd and step.ub[i] <= step.lambd:
                constraints.append(_lhs == 0.0)
            else:
                if not step.relax_binary_vars:
                    new_w1[i] = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                    new_w2[i] = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                    step.idx_with_right_binary_vars.add(i)

                    if step.ub[i] > step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        # Upper right part: w1 = 1, y >= lambda_t
                        constraints.extend(
                            [
                                _rhs >= step.lambd + (step.lb[i] - step.lambd) * (1 - new_w1[i]),
                                _rhs <= step.lambd + (step.ub[i] - step.lambd) * new_w1[i],
                                _lhs <= _rhs - step.lambd + (2 * step.lambd) * (1 - new_w1[i]),
                            ]
                        )

                        if step.lb[i] < -step.lambd:
                            constraints.extend(
                                [
                                    _lhs <= lhs_ub[i] * new_w1[i],
                                    _lhs >= lhs_lb[i] * new_w2[i],
                                    new_w1[i] + new_w2[i] <= 1,
                                ]
                            )

                        if -step.lambd <= step.lb[i] <= step.lambd:
                            constraints.append(_lhs <= lhs_ub[i] * new_w1[i])

                    if step.lb[i] < -step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        # Lower left part: w2 = 1, y <= -lambda_t
                        constraints.extend(
                            [
                                _rhs <= -step.lambd + (step.ub[i] + step.lambd) * (1 - new_w2[i]),
                                _rhs >= -step.lambd + (step.lb[i] + step.lambd) * new_w2[i],
                                _lhs >= _rhs + step.lambd + (-2 * step.lambd) * (1 - new_w2[i]),
                            ]
                        )

                        if -step.lambd <= step.lb[i] <= step.lambd:
                            constraints.append(_lhs >= lhs_lb[i] * new_w2[i])

                if step.lb[i] < -step.lambd and step.ub[i] > step.lambd:
                    constraints.extend(
                        [
                            _lhs >= _rhs - step.lambd,
                            _lhs <= _rhs + step.lambd,
                            _lhs <= lhs_ub[i] / (step.ub[i] + step.lambd) * (_rhs + step.lambd),
                            _lhs >= lhs_lb[i] / (step.lb[i] - step.lambd) * (_rhs - step.lambd),
                        ]
                    )

                elif -step.lambd <= step.lb[i] <= step.lambd < step.ub[i]:
                    constraints.extend(
                        [
                            _lhs >= 0,
                            _lhs <= lhs_ub[i] / (step.ub[i] - step.lb[i]) * (_rhs - step.lb[i]),
                            _lhs >= _rhs - step.lambd,
                        ]
                    )

                elif step.lambd >= step.ub[i] >= -step.lambd >= step.lb[i]:
                    constraints.extend(
                        [
                            _lhs <= 0,
                            _lhs >= lhs_lb[i] / (step.lb[i] - step.ub[i]) * (_rhs - step.ub[i]),
                            _lhs <= _rhs + step.lambd,
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

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

    def solve_model(self, steps, bounds, **kwargs):
        if kwargs.get("huchette_cuts", False):
            self.solver.add_huchette_cuts(steps, bounds)

        self.solver.optimize()
        return self.solver["objVal"]

    def extract_sol(self, iterate):
        return self.solver.get_solution_value(iterate)
