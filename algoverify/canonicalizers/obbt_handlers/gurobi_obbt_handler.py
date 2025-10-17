import logging

import gurobipy as gp
import numpy as np

from algoverify.solvers import gurobi

log = logging.getLogger(__name__)


class GurobiOBBTHandler(object):

    def __init__(self):
        self.vector_var_map = {}
        self.solver = gurobi.GurobiSolver()


        self.step_constr_map = {}

    def add_var(self, var, lb=-np.inf, ub=np.inf):
        self.solver.add_var(var, lb, ub)

    def add_param_stack_var(self, param_stack_var, param_list):
        self.solver.add_param_stack_var(param_stack_var, param_list)

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        lhs_gp_expr = self.solver.expression_to_solver(lhs_expr)
        rhs_gp_expr = self.solver.expression_to_solver(rhs_expr)

        self.solver.add_constr(lhs_gp_expr == rhs_gp_expr)

    def obbt(self, linexpr, lb, ub):
        self.solver.update_model()
        lb_out = np.zeros(linexpr.ndim)
        ub_out = np.zeros(linexpr.ndim)

        gp_expr = self.solver.expression_to_solver(linexpr)

        for sense in ["minimize", "maximize"]:
            for i in range(linexpr.ndim):
                self.solver.set_objective(gp_expr[i], sense)
                self.solver.optimize()


                if self.solver.model.status != gp.GRB.OPTIMAL:
                    log.info(f'bound tighting failed, GRB model status: {self.solver.model.status}')
                    log.info(f'i = {i}, sense={sense}')
                    return

                if sense == "maximize":
                    ub_out[i] = self.solver.model.objVal
                else:
                    lb_out[i] = self.solver.model.objVal

        # reset objective
        self.solver.set_objective(0)

        assert np.all(lb_out <= ub_out)
        return np.maximum(lb, lb_out), np.minimum(ub, ub_out)

    def add_relu_constraints(self, step):
        lhs = self.solver.expression_to_solver(step.lhs)
        rhs = self.solver.expression_to_solver(step.rhs)

        constraints = []

        for i in step.nonproj_indices:
            constraints.append(lhs[i] == rhs[i])

        for i in step.proj_indices:
            if step.ub[i] <= 0:
                constraints.append(lhs[i] == 0)
            elif step.lb[i] > 0:
                constraints.append(lhs[i] == rhs[i])
            else:
                constraints.append(lhs[i] >= 0)
                constraints.append(lhs[i] <= step.ub[i] / (step.ub[i] - step.lb[i]) * (rhs[i] - step.lb[i]))
                constraints.append(lhs[i] >= rhs[i])

        self.solver.constraints_to_solver(constraints)

    def add_saturated_linear_constraints(self, step, lhs_lb, lhs_ub):
        lhs = step.lhs
        rhs = step.rhs
        l, u = step.l, step.u

        lhs_gp_expr = self.solver.expression_to_solver(lhs)
        rhs_gp_expr = self.solver.expression_to_solver(rhs)
        rhs_lb = step.lb
        rhs_ub = step.ub
        n = lhs.get_output_dim()

        out_constraints = []

        for i in range(n):
            if rhs_lb[i] >= u[i]:
                out_constraints.append(lhs_gp_expr[i] == u[i])
            elif rhs_ub[i] <= l[i]:
                out_constraints.append(lhs_gp_expr[i] == l[i])
            elif rhs_lb[i] >= l[i] and rhs_ub[i] <= u[i]:
                out_constraints.append(lhs_gp_expr[i] == rhs_gp_expr[i])
            else:
                if rhs_lb[i] < l[i] and rhs_ub[i] > u[i]:
                    # following two are redundant due to preprocessing bounds on the left hand side iterate
                    out_constraints.append(lhs_gp_expr[i] <= (u[i] - l[i]) / (u[i] - rhs_lb[i]) * (rhs_gp_expr[i] - u[i]) + u[i])
                    out_constraints.append(lhs_gp_expr[i] >= (u[i] - l[i]) / (rhs_ub[i] - l[i]) * (rhs_gp_expr[i] - l[i]) + l[i])

                elif rhs_lb[i] < l[i] and rhs_ub[i] < u[i]:
                    out_constraints.append(lhs_gp_expr[i] >= rhs_gp_expr[i])
                    out_constraints.append(lhs_gp_expr[i] <= (rhs_ub[i] - l[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_ub[i]) + rhs_ub[i])

                elif rhs_lb[i] > l[i] and rhs_ub[i] > u[i]:
                    out_constraints.append(lhs_gp_expr[i] <= rhs_gp_expr[i])

                    out_constraints.append(lhs_gp_expr[i] >= (u[i] - rhs_lb[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i]) + rhs_lb[i])

                else:
                    raise RuntimeError('Unreachable code')

        self.step_constr_map[step] = [self.solver.add_constr(constr) for constr in out_constraints]


    def add_saturated_linear_param_constraints(self, step, out_lb, out_ub, l_lb, l_ub, u_lb, u_ub):
        lhs = step.lhs
        rhs = step.rhs
        l, u = step.l, step.u

        lhs_gp_expr = self.solver.expression_to_solver(lhs)
        rhs_gp_expr = self.solver.expression_to_solver(rhs)
        rhs_lb = step.lb
        rhs_ub = step.ub
        n = lhs.get_output_dim()

        out_constraints = []

        l_gp_var = self.solver.expression_to_solver(l)
        u_gp_var = self.solver.expression_to_solver(u)

        for i in range(n):
            if rhs_lb[i] >= u_ub[i]:
                out_constraints.append(lhs_gp_expr[i] == u_gp_var[i])
            elif rhs_ub[i] <= l_lb[i]:
                out_constraints.append(lhs_gp_expr[i] == l_gp_var[i])
            elif rhs_lb[i] >= l_ub[i] and rhs_ub[i] <= u_lb[i]:
                out_constraints.append(lhs_gp_expr[i] == rhs_gp_expr[i])
            else:
                if rhs_lb[i] < l_ub[i] and rhs_ub[i] <= u_lb[i]:
                    out_constraints.append(lhs_gp_expr[i] >= l_gp_var[i])
                    out_constraints.append(lhs_gp_expr[i] >= rhs_gp_expr[i])
                    out_constraints.append(lhs_gp_expr[i] <= (rhs_ub[i] - l_ub[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_ub[i]) + rhs_ub[i])

                elif rhs_lb[i] >= l_ub[i] and rhs_ub[i] > u_lb[i]:
                    out_constraints.append(lhs_gp_expr[i] <= u_gp_var[i])
                    out_constraints.append(lhs_gp_expr[i] <= rhs_gp_expr[i])
                    out_constraints.append(lhs_gp_expr[i] >= (u_lb[i] - rhs_lb[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i]) + rhs_lb[i])

                else:
                    out_constraints.append(lhs_gp_expr[i] <= u_gp_var[i])
                    out_constraints.append(lhs_gp_expr[i] >= l_gp_var[i])

                    # TODO: figure out if these can be u_gp_var/l_gp_var or need to be the upper/lower bounds, resp.
                    out_constraints.append(lhs_gp_expr[i] <= (u_ub[i] - l_ub[i]) / (u_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - u_gp_var[i]) + u_gp_var[i])
                    out_constraints.append(lhs_gp_expr[i] >= (u_lb[i] - l_lb[i]) / (rhs_ub[i] - l_lb[i]) * (rhs_gp_expr[i] - l_gp_var[i]) + l_gp_var[i])

        self.step_constr_map[step] = [self.solver.add_constr(constr) for constr in out_constraints]


    def add_soft_threshold_constraints(self, step, lhs_lb, lhs_ub):
        lhs = step.lhs
        rhs = step.rhs
        lambd = step.lambd

        lhs_gp_expr = self.solver.expression_to_solver(lhs)
        rhs_gp_expr = self.solver.expression_to_solver(rhs)
        rhs_lb = step.lb
        rhs_ub = step.ub
        n = lhs.get_output_dim()

        out_constraints = []

        for i in range(n):
            if rhs_lb[i] >= lambd:
                out_constraints.append(lhs_gp_expr[i] == rhs_gp_expr[i] - lambd)
            elif rhs_ub[i] <= -lambd:
                out_constraints.append(lhs_gp_expr[i] == rhs_gp_expr[i] + lambd)
            elif rhs_lb[i] >= -lambd and rhs_ub[i] <= lambd:
                out_constraints.append(lhs_gp_expr[i] == 0.0)
            else:
                if rhs_lb[i] < -lambd and rhs_ub[i] > lambd:
                    out_constraints.append(lhs_gp_expr[i] >= rhs_gp_expr[i] - lambd)
                    out_constraints.append(lhs_gp_expr[i] <= rhs_gp_expr[i] + lambd)

                    out_constraints.append(lhs_gp_expr[i] <= lhs_ub[i] / (rhs_ub[i] + lambd) * (rhs_gp_expr[i] + lambd))
                    out_constraints.append(lhs_gp_expr[i] >= lhs_lb[i] / (rhs_lb[i] - lambd) * (rhs_gp_expr[i] - lambd))

                elif -lambd <= rhs_lb[i] <= lambd and rhs_ub[i] > lambd:
                    out_constraints.append(lhs_gp_expr[i] >= 0)
                    out_constraints.append(lhs_gp_expr[i] <= lhs_ub[i] / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i]))
                    out_constraints.append(lhs_gp_expr[i] >= rhs_gp_expr[i] - lambd)

                elif -lambd <= rhs_ub[i] <= lambd and rhs_lb[i] <= -lambd:

                    out_constraints.append(lhs_gp_expr[i] <= 0)
                    out_constraints.append(lhs_gp_expr[i] >= lhs_lb[i] / (rhs_lb[i] - rhs_ub[i]) * (rhs_gp_expr[i] - rhs_ub[i]))
                    out_constraints.append(lhs_gp_expr[i] <= rhs_gp_expr[i] + lambd)

                else:
                    raise RuntimeError('Unreachable code')

        self.step_constr_map[step] = [self.solver.add_constr(constr) for constr in out_constraints]

