import gurobipy as gp
import numpy as np
import scipy.sparse as spa

from algoverify import LinearExpression
from algoverify.canonicalizers.canonicalizer import Canonicalizer
from algoverify.canonicalizers.huchette_cuts.relu import relu_add_conv_cuts, relu_create_new_constr
from algoverify.canonicalizers.huchette_cuts.soft_threshold import (
    st_add_neg_conv_cuts,
    st_add_pos_conv_cuts,
    st_create_new_neg_constr,
    st_create_new_pos_constr,
)
from algoverify.constraint import Constraint
from algoverify.steps import Relu, SoftThreshold


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

    def add_var(self, var:LinearExpression, lb=-np.inf, ub=np.inf) -> None:
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

    def add_constraint(self, constraint:Constraint) -> None:
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
        # Expression on the right hand side of the constraint object is always 0
        lhs = self.solver.expression_to_solver(constraint.lhs)
        match constraint.operator:
            case "=":
                self.solver.add_constr(lhs == 0)
            case "<=":
                self.solver.add_constr(lhs <= 0)

    def add_relu_constraints(self, step:Relu) -> None:
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
        lhs = self.solver.expression_to_solver(step.lhs)
        rhs = self.solver.expression_to_solver(step.rhs)

        constraints = []
        new_vars = {}

        # after ranges are added, remove the range(n) and add the constraints when needed
        # make sure to enforce equality constraints between lhs and rhs for other indices
        for i, (_lhs, _rhs) in enumerate(zip(lhs, rhs)):
            if i in step.nonproj_indices:
                constraints.append(_lhs == _rhs)
            else:
                if step.ub[i] <= 0:
                    constraints.append(_lhs == 0)
                elif step.lb[i] > 0:
                    constraints.append(_lhs == _rhs)
                elif step.ub[i] > 0 or step.lb[i] <= 0:
                    if not step.relax_binary_vars:
                        new_vars[i] = self.solver.get_var(vtype="binary")
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
        assert step.lhs in self.solver.vector_var_map

        lhs = self.solver.expression_to_solver(step.lhs)
        rhs = self.solver.expression_to_solver(step.rhs)
        n = step.lhs.get_output_dim()

        constraints = []

        for i in range(n):
            if step.lb[i] >= step.u[i]:
                constraints.append(lhs[i] == step.u[i])
            elif step.ub[i] <= step.l[i]:
                constraints.append(lhs[i] == step.l[i])
            elif step.lb[i] >= step.l[i] and step.ub[i] <= step.u[i]:
                constraints.append(lhs[i] == rhs[i])
            else:
                if not step.relax_binary_vars:
                    w1 = self.solver.get_var(vtype="binary")
                    w2 = self.solver.get_var(vtype="binary")
                    if step.lb[i] < step.l[i]:
                        constraints.extend(
                            [
                                lhs[i] <= rhs[i] + (step.ub[i] - step.lb[i]) * w2,
                                rhs[i] >= step.l[i] + (step.lb[i] - step.l[i]) * w2,
                                rhs[i] <= step.l[i] + (step.ub[i] - step.l[i]) * (1 - w2),
                            ]
                        )
                        if step.ub[i] > step.u[i]:
                            constraints.extend(
                                [
                                    lhs[i] >= step.u[i] - (step.u[i] - step.l[i]) * (1 - w1),
                                    lhs[i] <= step.l[i] + (step.u[i] - step.l[i]) * (1 - w2),
                                    lhs[i] <= rhs[i] + (step.ub[i] - step.lb[i]) * w2,
                                    w1 + w2 <= 1,
                                ]
                            )
                        elif step.ub[i] > step.u[i]:
                            constraints.extend(
                                [
                                    lhs[i] <= step.l[i] + (step.ub[i] - step.l[i]) * (1 - w2),
                                    rhs[i] >= step.l[i] + (step.lb[i] - step.l[i]) * w2,
                                ]
                            )

                    if step.ub[i] > step.u[i]:
                        constraints.extend(
                            [
                                lhs[i] >= rhs[i] - (step.ub[i] - step.lb[i]) * w1,
                                rhs[i] >= step.u[i] + (step.lb[i] - step.u[i]) * (1 - w1),
                                rhs[i] <= step.u[i] + (step.ub[i] - step.u[i]) * w1,
                            ]
                        )

                        if step.lb[i] > step.l[i]:
                            constraints.append(
                                lhs[i] >= step.u[i] - (step.u[i] - step.lb[i]) * (1 - w1)
                            )

                if step.lb[i] < step.l[i] and step.ub[i] > step.u[i]:
                    # following two are redundant due to preprocessing bounds on the left hand side var
                    expr = (step.u[i] - step.l[i]) / (step.u[i] - step.lb[i]) * (
                        rhs[i] - step.u[i]
                    ) + step.u[i]

                    constraints.extend([lhs[i] <= expr, lhs[i] >= expr])

                elif step.lb[i] < step.l[i] and step.ub[i] < step.u[i]:
                    constraints.extend(
                        [
                            lhs[i] >= rhs[i],
                            lhs[i]
                            <= ((step.ub[i] - step.l[i]) / (step.ub[i] - step.lb[i]))
                            * (rhs[i] - step.ub[i])
                            + step.ub[i],
                        ]
                    )

                elif step.lb[i] > step.l[i] and step.ub[i] > step.u[i]:
                    constraints.extend(
                        [
                            lhs[i] <= rhs[i],
                            lhs[i]
                            >= ((step.u[i] - step.lb[i]) / (step.ub[i] - step.lb[i]))
                            * (rhs[i] - step.lb[i])
                            + step.lb[i],
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

        self.solver.constraints_to_solver(constraints)

    def add_saturated_linear_param_constraints(self, step):
        assert step.lhs in self.solver.vector_var_map
        self.solver.constraints_to_solver([])
        return None

    def add_soft_threshold_constraints(self, step, lhs_lb, lhs_ub):
        assert step.lhs in self.solver.vector_var_map

        lhs = self.solver.expression_to_solver(step.lhs)
        rhs = self.solver.expression_to_solver(step.rhs)

        constraints = []
        new_w1 = {}
        new_w2 = {}
        for i in range(step.lhs.get_output_dim()):
            if step.lb[i] >= step.lambd:
                constraints.append(lhs[i] == rhs[i] - step.lambd)
            elif step.ub[i] <= -step.lambd:
                constraints.append(lhs[i] == rhs[i] + step.lambd)
            elif step.lb[i] >= -step.lambd and step.ub[i] <= step.lambd:
                constraints.append(lhs[i] == 0.0)
            else:
                if not step.relax_binary_vars:
                    new_w1[i] = self.solver.get_var(vtype="binary")
                    new_w2[i] = self.solver.get_var(vtype="binary")
                    step.idx_with_right_binary_vars.add(i)

                    if step.ub[i] > step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        # Upper right part: w1 = 1, y >= lambda_t
                        constraints.extend(
                            [
                                rhs[i] >= step.lambd + (step.lb[i] - step.lambd) * (1 - new_w1[i]),
                                rhs[i] <= step.lambd + (step.ub[i] - step.lambd) * new_w1[i],
                                lhs[i] <= rhs[i] - step.lambd + (2 * step.lambd) * (1 - new_w1[i]),
                            ]
                        )

                        if step.lb[i] < -step.lambd:
                            constraints.extend(
                                [
                                    lhs[i] <= lhs_ub[i] * new_w1[i],
                                    lhs[i] >= lhs_lb[i] * new_w2[i],
                                    new_w1[i] + new_w2[i] <= 1,
                                ]
                            )

                        if -step.lambd <= step.lb[i] <= step.lambd:
                            constraints.append(lhs[i] <= lhs_ub[i] * new_w1[i])

                    if step.lb[i] < -step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        # Lower left part: w2 = 1, y <= -lambda_t
                        constraints.extend(
                            [
                                rhs[i] <= -step.lambd + (step.ub[i] + step.lambd) * (1 - new_w2[i]),
                                rhs[i] >= -step.lambd + (step.lb[i] + step.lambd) * new_w2[i],
                                lhs[i] >= rhs[i] + step.lambd + (-2 * step.lambd) * (1 - new_w2[i]),
                            ]
                        )

                        if -step.lambd <= step.lb[i] <= step.lambd:
                            constraints.append(lhs[i] >= lhs_lb[i] * new_w2[i])

                if step.lb[i] < -step.lambd and step.ub[i] > step.lambd:
                    constraints.extend(
                        [
                            lhs[i] >= rhs[i] - step.lambd,
                            lhs[i] <= rhs[i] + step.lambd,
                            lhs[i] <= lhs_ub[i] / (step.ub[i] + step.lambd) * (rhs[i] + step.lambd),
                            lhs[i] >= lhs_lb[i] / (step.lb[i] - step.lambd) * (rhs[i] - step.lambd),
                        ]
                    )

                elif -step.lambd <= step.lb[i] <= step.lambd < step.ub[i]:
                    constraints.extend(
                        [
                            lhs[i] >= 0,
                            lhs[i] <= lhs_ub[i] / (step.ub[i] - step.lb[i]) * (rhs[i] - step.lb[i]),
                            lhs[i] >= rhs[i] - step.lambd,
                        ]
                    )

                elif step.lambd >= step.ub[i] >= -step.lambd >= step.lb[i]:
                    constraints.extend(
                        [
                            lhs[i] <= 0,
                            lhs[i] >= lhs_lb[i] / (step.lb[i] - step.ub[i]) * (rhs[i] - step.ub[i]),
                            lhs[i] <= rhs[i] + step.lambd,
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

        self.solver.constraints_to_solver(constraints)

    def add_theory_cut(self, C, target_expr, bound_lb, bound_ub):
        if C == np.inf:
            return
        target_gp_expr = self.solver.expression_to_solver(target_expr)
        if target_expr.is_leaf:
            # directly update the lb/ub parameters if possible
            target_gp_expr.lb = bound_lb - C
            target_gp_expr.ub = bound_ub + C
        else:
            self.solver.add_constr(target_gp_expr >= bound_lb - C)
            self.solver.add_constr(target_gp_expr <= bound_ub + C)

    def post_process(self, target_expr, bound_lb, bound_ub):
        target_gp_expr = self.solver.expression_to_solver(target_expr)
        if target_expr.is_leaf:
            # directly update the lb/ub parameters if possible
            target_gp_expr.lb = bound_lb
            target_gp_expr.ub = bound_ub
        else:
            self.solver.add_constr(target_gp_expr >= bound_lb)
            self.solver.add_constr(target_gp_expr <= bound_ub)

    def set_zero_objective(self):
        # this function is mostly just for debugging the constraints
        self.solver.set_objective(expression=0)

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
        self.solver.update_model()

        constraints = []
        M = np.maximum(np.max(np.abs(np.hstack(ub_list))), np.max(np.abs(np.hstack(lb_list))))
        q = self.solver.get_var(ub=M)

        all_gammas = []

        for expr, lb, ub in zip(expr_list, lb_list, ub_list):
            n = expr.get_output_dim()
            rhs_gp_expr = self.solver.expression_to_solver(expr)

            up = self.solver.get_matrix_var(n, ub=np.abs(ub))
            un = self.solver.get_matrix_var(n, ub=np.abs(lb))

            constraints.append(up - un == rhs_gp_expr)

            for i in range(n):
                if lb[i] >= 0:
                    constraints.append(un[i] == 0)
                elif ub[i] < 0:
                    constraints.append(up[i] == 0)
                else:
                    vobji = self.solver.get_matrix_var(1, vtype="binary")
                    constraints.extend(
                        [up[i] <= np.abs(ub[i]) * vobji, un[i] <= np.abs(lb[i]) * (1 - vobji)]
                    )

            gamma = self.solver.get_matrix_var(n, vtype="binary")
            for i in range(n):
                constraints.extend([q >= up[i] + un[i], q <= up[i] + un[i] + M * (1 - gamma[i])])

            all_gammas.append(gamma)
        # need 1 gamma across all vectors in the list
        gamma_constr = 0
        for gamma in all_gammas:
            gamma_constr += gp.quicksum(gamma)
        constraints.append(gamma_constr == 1)

        for constr in constraints:
            self.solver.add_constr(constr)
        self.solver.set_objective(expression=q, type="maximize")
        self.solver.update_model()

    def solve_model(self, steps, bounds, **kwargs):
        if "huchette_cuts" in kwargs:
            if kwargs["huchette_cuts"]:
                self.add_huchette_cuts(steps, bounds, **kwargs)

        if "include_rel_LP_sol" in kwargs:
            if kwargs["include_rel_LP_sol"]:
                rel_model = self.solver.relax_model()
                rel_model.optimize()
                self.solver.rel_LP_sol = rel_model.objVal
            else:
                self.solver.rel_LP_sol = None
        else:
            self.solver.rel_LP_sol = None
        self.solver.optimize()
        return self.solver.get_solve_data()["objVal"]

    def add_huchette_cuts(self, steps, bounds, **kwargs):
        relaxed_model = self.solver.relax_model()
        relaxed_model.optimize()

        for step in steps:
            if isinstance(step, Relu):
                self.relu_huchette_cuts(relaxed_model, step, bounds, **kwargs)
            elif isinstance(step, SoftThreshold):
                self.soft_threshold_huchette_cuts(relaxed_model, step, bounds)

        self.solver.update_model()

    def relu_huchette_cuts(self, relaxed_model, step, bounds, **kwargs):
        w, A, y, l, u, w_var, y_var = self.extract_vecs_for_huchette_cuts(
            relaxed_model, step, bounds
        )

        for i in range(step.lhs.get_output_dim()):
            if i in step.proj_indices and i in step.idx_with_binary_vars:
                Iint, lI, h, Lhat, Uhat = relu_add_conv_cuts(w[i], A[i, :], y, l, u)
                if Iint is not None:
                    new_constr = relu_create_new_constr(
                        w_var[i], A[i, :], y_var, Iint, lI, h, Lhat, Uhat
                    )
                    self.solver.add_constr(new_constr)

    def soft_threshold_huchette_cuts(self, relaxed_model, step, bounds):
        w, A, y, l, u, w_var, y_var = self.extract_vecs_for_huchette_cuts(
            relaxed_model, step, bounds
        )

        for i in range(step.lhs.ndim):
            constr = None
            Iint, lI, h, p_Lhat, p_Uhat = st_add_pos_conv_cuts(w[i], A[i, :], y, step.lambd, l, u)
            Iint, uI, h, l_Lhat, l_Uhat = st_add_neg_conv_cuts(w[i], A[i, :], y, step.lambd, l, u)

            if Iint is not None:
                if i in step.idx_with_right_binary_vars and (
                    i not in step.idx_with_left_binary_vars or w[i] > 0
                ):
                    constr = st_create_new_pos_constr(
                        w_var[i], A[i, :], y_var, Iint, lI, h, p_Lhat, p_Uhat
                    )

                if i in step.idx_with_left_binary_vars and (
                    i not in step.idx_with_right_binary_vars or w[i] < 0
                ):
                    constr = st_create_new_neg_constr(
                        w_var[i], A[i, :], y_var, Iint, uI, h, l_Lhat, l_Uhat
                    )

                if constr is not None:
                    self.solver.add_constr(constr)

    def extract_vecs_for_huchette_cuts(
        self, relaxed_model, step, bounds, **kwargs
    ):
        A = []
        y_var = []
        l = []
        u = []
        for k, v in step.rhs.decomposition_dict.items():
            if spa.issparse(v):
                A.append(v.todense())
            else:
                A.append(v)
            y_var.append(
                self.solver.extract_vec_var_by_name(
                    self.solver.leaf_expr_to_solver(k), relaxed_model
                )
            )
            l.append(bounds[k][0])
            u.append(bounds[k][1])
        w_var = self.solver.extract_vec_var_by_name(
            self.solver.leaf_expr_to_solver(step.lhs), relaxed_model
        )

        return (
            w_var.X,
            np.asarray(np.hstack(A)),
            gp.hstack(y_var).X,
            np.hstack(l),
            np.hstack(u),
            w_var,
            gp.hstack(y_var),
        )

    def extract_sol(self, iterate):
        out = 0
        for k, v in iterate.decomposition_dict.items():
            out += v @ [
                self.solver.get_solution_value(var) for var in self.solver.vector_var_map[k]
            ]

        return out
