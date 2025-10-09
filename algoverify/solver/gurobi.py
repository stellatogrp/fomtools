import logging

import gurobipy as gp
import numpy as np
import scipy.sparse as spa

from algoverify.solver.solver import Solver
from algoverify.canonicalizers.huchette_cuts.relu import relu_add_conv_cuts, relu_create_new_constr
from algoverify.canonicalizers.huchette_cuts.soft_threshold import (
    st_add_neg_conv_cuts,
    st_add_pos_conv_cuts,
    st_create_new_neg_constr,
    st_create_new_pos_constr,
)
from algoverify.steps import Relu, SoftThreshold

log = logging.getLogger(__name__)


class GurobiSolver(Solver):
    def __init__(self, solver_params={}):
        self.vector_var_map = {}
        self.model = gp.Model()

        for key, val in solver_params.items():
            self.model.setParam(key, val)

        self.step_constr_map = {}
        self.obj_constraints = []

    def _add_term(self, term, lb=-np.inf, ub=np.inf):
        self.vector_var_map[term] = self.model.addMVar(term.ndim, lb=lb, ub=ub)
        self.model.update()

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        self._add_term(param, lb, ub)

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self._add_term(iterate, lb, ub)

    def add_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self._add_term(iterate, lb, ub)


    def add_param_stack_var(self, param_stack_var, param_list):
        """
        param_stack_var is the VP param object and param_list is the list of previous params to stack
        """
        self.vector_var_map[param_stack_var] = gp.hstack(
            [self.vector_var_map[param] for param in param_list]
        )

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        self.model.addConstr(
            self.expression_to_solver(lhs_expr) == self.expression_to_solver(lhs_expr)
        )
        self.model.update()

    def update_vector_bounds(self, iterate, lb, ub):
        gp_var = self.vector_var_map[iterate]

        gp_var.lb = np.maximum(gp_var.lb, lb)
        gp_var.ub = np.minimum(gp_var.ub, ub)

    def leaf_expr_to_gp_var(self, expression):
        if expression.is_leaf:
            return self.vector_var_map[next(iter(expression.decomposition_dict))]

        raise TypeError(
            "Invalid input: expression must be a leaf. Received a composite one instead."
        )

    def expression_to_solver(self, expression):
        gp_expr = np.zeros(expression.ndim)
        for k, v in expression.decomposition_dict.items():
            gp_expr += v @ self.vector_var_map[k]

        return gp_expr

    def constraints_to_solver(self, step, constraints):
        self.step_constr_map[step] = [self.model.addConstr(constr) for constr in constraints]
        self.model.update()

    def add_constr(self, constraint):
        self.model.addConstr(constraint)
        self.model.update()

    def get_binary_var(self):
        return self.model.addVar(vtype=gp.GRB.BINARY)

    def add_matrix_var(self, n, ub):
        return self.model.addMVar(n, ub)


    def add_relu_constraints(self, step):
        assert step.lhs in self.vector_var_map

        lhs = self.expression_to_solver(step.lhs)
        rhs = self.expression_to_solver(step.rhs)

        constraints = []
        new_vars = {}

        # after ranges are added, remove the range(n) and add the constraints when needed
        # make sure to enforce equality constraints between lhs and rhs for other indices
        for i, (_lhs, _rhs) in enumerate(zip(lhs, rhs)):
            if i in step.nonproj_indices:
                constraints.append(_lhs == _rhs)
            if i in step.proj_indices:
                if step.ub[i] <= 0:
                    constraints.append(_lhs == 0)
                elif step.lb[i] > 0:
                    constraints.append(_lhs == _rhs)
                elif step.ub[i] > 0 or step.lb[i] <= 0:
                    if not step.relax_binary_vars:
                        new_vars[i] = self.get_binary_var()
                        step.idx_with_binary_vars.add(i)
                        constraints.append(_lhs <= step.ub[i] * new_vars[i])

                    constraints.extend(
                        [
                            _lhs <= step.ub[i] / (step.ub[i] - step.lb[i]) * (_rhs - step.lb[i]),
                            _lhs >= _rhs,
                            _lhs <= _rhs - step.lb[i] * (1 - new_vars[i]),
                        ]
                    )

        self.constraints_to_solver(step, constraints)

    def add_saturated_linear_constraints(self, step):
        assert step.lhs in self.vector_var_map

        lhs = self.expression_to_solver(step.lhs)
        rhs = self.expression_to_solver(step.rhs)
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
                    w1 = self.get_binary_var()
                    w2 = self.get_binary_var()
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
                    # following two are redundant due to preprocessing bounds on the left hand side iterate
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

        self.constraints_to_solver(step, constraints)

    def add_saturated_linear_param_constraints(self, step):
        assert step.lhs in self.vector_var_map
        self.constraints_to_solver(step, [])
        return None

    def add_soft_threshold_constraints(self, step, lhs_lb, lhs_ub):
        assert step.lhs in self.vector_var_map

        lhs = self.expression_to_solver(step.lhs)
        rhs = self.expression_to_solver(step.rhs)

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
                    new_w1[i] = self.get_binary_var()
                    new_w2[i] = self.get_binary_var()
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

        self.constraints_to_solver(step, constraints)


    def add_theory_cut(self, C, target_expr, bound_lb, bound_ub):
        if C == np.inf:
            return
        target_gp_expr = self.expression_to_solver(target_expr)
        if target_expr.is_leaf:
            # directly update the lb/ub parameters if possible
            target_gp_expr.lb = bound_lb - C
            target_gp_expr.ub = bound_ub + C
        else:
            self.add_constr(target_gp_expr >= bound_lb - C)
            self.add_constr(target_gp_expr <= bound_ub + C)

    def post_process(self, target_expr, bound_lb, bound_ub):
        target_gp_expr = self.expression_to_solver(target_expr)
        if target_expr.is_leaf:
            # directly update the lb/ub parameters if possible
            target_gp_expr.lb = bound_lb
            target_gp_expr.ub = bound_ub
        else:
            self.add_constr(target_gp_expr >= bound_lb)
            self.add_constr(target_gp_expr <= bound_ub)


    def set_zero_objective(self):
        # this function is mostly just for debugging the constraints
        self.model.setObjective(0)
        self.model.update()

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
        self.model.update()

        constraints = []
        M = np.maximum(np.max(np.abs(np.hstack(ub_list))), np.max(np.abs(np.hstack(lb_list))))
        q = self.model.addVar(ub=M)

        all_gammas = []

        for expr, lb, ub in zip(expr_list, lb_list, ub_list):
            n = expr.get_output_dim()
            rhs_gp_expr = self.expression_to_solver(expr)

            up = self.model.addMVar(n, ub=np.abs(ub))
            un = self.model.addMVar(n, ub=np.abs(lb))

            constraints.append(up - un == rhs_gp_expr)

            for i in range(n):
                if lb[i] >= 0:
                    constraints.append(un[i] == 0)
                elif ub[i] < 0:
                    constraints.append(up[i] == 0)
                else:
                    vobji = self.model.addMVar(1, vtype=gp.GRB.BINARY)
                    constraints.extend(
                        [up[i] <= np.abs(ub[i]) * vobji, un[i] <= np.abs(lb[i]) * (1 - vobji)]
                    )

            gamma = self.model.addMVar(n, vtype=gp.GRB.BINARY)
            for i in range(n):
                constraints.extend([q >= up[i] + un[i], q <= up[i] + un[i] + M * (1 - gamma[i])])

            all_gammas.append(gamma)
        # need 1 gamma across all vectors in the list
        gamma_constr = 0
        for gamma in all_gammas:
            gamma_constr += gp.quicksum(gamma)
        constraints.append(gamma_constr == 1)

        for constr in constraints:
            self.model.addConstr(constr)
        self.model.setObjective(q, gp.GRB.MAXIMIZE)
        self.model.update()

    def solve_model(self, steps, lower_bounds, upper_bounds, **kwargs):
        if "huchette_cuts" in kwargs:
            if kwargs["huchette_cuts"]:
                self.add_huchette_cuts(steps, lower_bounds, upper_bounds, **kwargs)

        if "include_rel_LP_sol" in kwargs:
            if kwargs["include_rel_LP_sol"]:
                rel_model = self.model.relax()
                rel_model.optimize()
                self.rel_LP_sol = rel_model.objVal
            else:
                self.rel_LP_sol = None
        else:
            self.rel_LP_sol = None
        self.model.optimize()
        return self.model.objVal

    def add_huchette_cuts(self, steps, lower_bounds, upper_bounds, **kwargs):
        relaxed_model = self.model.relax()
        relaxed_model.optimize()

        for step in steps:
            if isinstance(step, Relu):
                self.relu_huchette_cuts(relaxed_model, step, lower_bounds, upper_bounds, **kwargs)
            elif isinstance(step, SoftThreshold):
                self.soft_threshold_huchette_cuts(relaxed_model, step, lower_bounds, upper_bounds)

        self.model.update()

    def relu_huchette_cuts(self, relaxed_model, step, lower_bounds, upper_bounds, **kwargs):
        w, A, y, l, u, w_var, y_var = self.extract_vecs_for_huchette_cuts(
            relaxed_model, step, lower_bounds, upper_bounds
        )

        for i in range(step.lhs.get_output_dim()):
            if i in step.proj_indices and i in step.idx_with_binary_vars:
                Iint, lI, h, Lhat, Uhat = relu_add_conv_cuts(w[i], A[i, :], y, l, u)
                if Iint is not None:
                    new_constr = relu_create_new_constr(
                        w_var[i], A[i, :], y_var, Iint, lI, h, Lhat, Uhat
                    )
                    self.model.addConstr(new_constr)

    def soft_threshold_huchette_cuts(self, relaxed_model, step, lower_bounds, upper_bounds):
        w, A, y, l, u, w_var, y_var = self.extract_vecs_for_huchette_cuts(
            relaxed_model, step, lower_bounds, upper_bounds
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
                    self.model.addConstr(constr)

    def extract_vecs_for_huchette_cuts(
        self, relaxed_model, step, lower_bounds, upper_bounds, **kwargs
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
            y_var.append(self.extract_vec_var_by_name(self.leaf_expr_to_gp_var(k), relaxed_model))
            l.append(lower_bounds[k])
            u.append(upper_bounds[k])
        w_var = self.extract_vec_var_by_name(self.leaf_expr_to_gp_var(step.lhs), relaxed_model)

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
                self.model.getVarByName(var.VarName.item()).X
                for var in self.vector_var_map[k]
            ]

        return out

    def extract_vec_var_by_name(self, var, new_model):
        return gp.MVar.fromlist(
            [new_model.getVarByName(var[i].item().VarName) for i in range(var.shape[0])]
        )

    def get_solve_data(self):
        return {
            "objVal": self.model.objVal,
            "objBound": self.model.objBound,
            "MIPGap": self.model.MIPGap,
            "Runtime": self.model.Runtime,
            "rel_LP_sol": self.rel_LP_sol,
            "numBinVars": self.model.NumBinVars,
        }
