import logging

import numpy as np

from algoverify.canonicalizers.gurobi_canonicalizer import GurobiCanonicalizer
from algoverify.canonicalizers.obbt_handlers import GurobiOBBTHandler
from algoverify.steps import Affine, Relu, SaturatedLinear, SoftThreshold
from algoverify.vector import Vector

log = logging.getLogger(__name__)


class Verifier(object):

    def __init__(
        self,
        n_obbt=3,
        theory_func=None,
        obbt=True,
        solver="gurobi",
        obbt_handler="gurobi",
        solver_params={},
    ):
        self.n_obbt = n_obbt
        self.objective = None
        self.theory_func = theory_func
        self.obbt = obbt

        self.params = []
        self.iterates = []
        self.steps = []
        self.constraint_sets = []

        if solver == 'gurobi':
            self.canonicalizer = GurobiCanonicalizer(solver_params=solver_params)

        if obbt_handler == 'gurobi':
            self.obbt_handler = GurobiOBBTHandler()

        self.lower_bounds = {}
        self.upper_bounds = {}

    def add_param(self, n, lb=-np.inf, ub=np.inf):
        assert np.all(lb <= ub)
        param = Vector(n)
        self.params.append(param)
        self.lower_bounds[param] = as_array(lb, n)
        self.upper_bounds[param] = as_array(ub, n)
        self.canonicalizer.add_param_var(param, lb=lb, ub=ub)

        if self.obbt:
            self.obbt_handler.add_param_var(param, lb=lb, ub=ub)

        return param

    def add_param_stack(self, param_list):
        dim = 0
        lbs = []
        ubs = []

        for p in param_list:
            assert p in self.params
            dim += p.ndim
            lbs.append(self.lower_bounds[p])
            ubs.append(self.upper_bounds[p])
        lb = np.hstack(lbs)
        ub = np.hstack(ubs)

        param_stack_var = Vector(dim)
        self.lower_bounds[param_stack_var] = lb
        self.upper_bounds[param_stack_var] = ub

        self.canonicalizer.add_param_stack_var(param_stack_var, param_list)

        if self.obbt:
            self.obbt_handler.add_param_stack_var(param_stack_var, param_list)

        return param_stack_var

    def add_initial_iterate(self, n, lb=-np.inf, ub=np.inf):
        assert np.all(lb <= ub)
        iterate = Vector(n)
        self.iterates.append(iterate)
        self.lower_bounds[iterate] = as_array(lb, n)
        self.upper_bounds[iterate] = as_array(ub, n)
        self.canonicalizer.add_initial_iterate_var(iterate, lb=lb, ub=ub)

        if self.obbt:
            self.obbt_handler.add_initial_iterate_var(iterate, lb=lb, ub=ub)

        return iterate

    # def add_iterate(self, n):
    #     return iterate

    def add_step(self, step):
        self.steps.append(step)

    def implicit_linear_step(self, lhs_mat, rhs_expr, lhs_mat_factorization=None):
        # need to think about best way to bound prop here/set up the api
        '''
            Process:
                - Construct out iterate
                - Construct affine step object
                - Do the interval bound propagations as necessary
                - Add the iterate and constraints to the canonicalizer
                - Do the OBBT (if self.num_obbt >= 0`)
                - Return the out iterate Vector object
        '''
        iterate = Vector(lhs_mat.shape[0])
        self.iterates.append(iterate)

        step = Affine(
            lhs_mat @ iterate,
            rhs_expr,
            lhs_mat=lhs_mat,
            lhs_mat_factorization=lhs_mat_factorization,
        )
        Minv_rhs = step.lhs_mat_inverse()

        iterate_lb, iterate_ub = self.linear_bound_propagation(Minv_rhs)
        assert np.all(iterate_lb <= iterate_ub)

        self.lower_bounds[iterate] = iterate_lb
        self.upper_bounds[iterate] = iterate_ub

        self.canonicalizer.add_iterate_var(iterate, lb=iterate_lb, ub=iterate_ub)
        self.canonicalizer.add_equality_constraint(lhs_mat @ iterate, rhs_expr)

        if self.obbt:
            self.obbt_handler.add_iterate_var(iterate, lb=iterate_lb, ub=iterate_ub)
            self.obbt_handler.add_equality_constraint(lhs_mat @ iterate, rhs_expr)

        self.add_step(step)

        return iterate

    def relu_step(self, rhs_expr, proj_ranges=None, relax_binary_vars=False):
        if proj_ranges is None:
            proj_ranges = [(0, rhs_expr.get_output_dim())]
        else:
            proj_ranges = list_upcast(proj_ranges)

        out_iterate = Vector(rhs_expr.get_output_dim())

        rhs_lb, rhs_ub = self.linear_bound_propagation(rhs_expr)
        relu_rhs_lb = relu(rhs_lb, proj_ranges=proj_ranges)
        relu_rhs_ub = relu(rhs_ub, proj_ranges=proj_ranges)

        if self.obbt:
            # obbt_lb, obbt_ub = self.canonicalizer.obbt(rhs_expr)
            obbt_lb, obbt_ub = self.obbt_handler.obbt(rhs_expr)
            rhs_lb, _ = update_lb(obbt_lb, rhs_lb)
            rhs_ub, _ = update_ub(obbt_ub, rhs_ub)

            relu_obbt_lb = relu(obbt_lb, proj_ranges=proj_ranges)
            relu_obbt_ub = relu(obbt_ub, proj_ranges=proj_ranges)

            out_lb = relu_obbt_lb
            out_ub = relu_obbt_ub
        else:
            out_lb = relu_rhs_lb
            out_ub = relu_rhs_ub

        step = Relu(
            lhs=out_iterate,
            rhs=rhs_expr,
            lb=rhs_lb,
            ub=rhs_ub,
            relax_binary_vars=relax_binary_vars,
            proj_ranges=proj_ranges,
        )

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_relu_constraints(step)

        if self.obbt:
            self.obbt_handler.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
            self.obbt_handler.add_relu_constraints(step, out_lb, out_ub)

        self.add_step(step)

        return out_iterate

    def saturated_linear_step(self, rhs_expr, l, u, relax_binary_vars=False):
        out_iterate = Vector(rhs_expr.get_output_dim())
        rhs_lb, rhs_ub = self.linear_bound_propagation(rhs_expr)

        sl_rhs_lb = saturated_linear(rhs_lb, l, u)
        sl_rhs_ub = saturated_linear(rhs_ub, l, u)

        if self.obbt:
            # obbt_lb, obbt_ub = self.canonicalizer.obbt(rhs_expr)
            obbt_lb, obbt_ub = self.obbt_handler.obbt(rhs_expr)
            rhs_lb, _ = update_lb(obbt_lb, rhs_lb)
            rhs_ub, _ = update_ub(obbt_ub, rhs_ub)

            sl_obbt_lb = saturated_linear(obbt_lb, l, u)
            sl_obbt_ub = saturated_linear(obbt_ub, l, u)

            out_lb = sl_obbt_lb
            out_ub = sl_obbt_ub
        else:
            out_lb = sl_rhs_lb
            out_ub = sl_rhs_ub

        step = SaturatedLinear(
            lhs=out_iterate,
            rhs=rhs_expr,
            lb=rhs_lb,
            ub=rhs_ub,
            relax_binary_vars=relax_binary_vars,
            l=l,
            u=u,
        )

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_saturated_linear_constraints(step, out_lb, out_ub)

        if self.obbt:
            self.obbt_handler.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
            self.obbt_handler.add_saturated_linear_constraints(step, out_lb, out_ub)

        self.add_step(step)

        return out_iterate

    def saturated_linear_param_step(self, rhs_expr, l, u, relax_binary_vars=False, equality_ranges=None):
        assert isinstance(l, Vector) and isinstance(u, Vector)
        out_iterate = Vector(rhs_expr.get_output_dim())
        rhs_lb, rhs_ub = self.linear_bound_propagation(rhs_expr)

        l_lb, l_ub = self.lower_bounds[l], self.upper_bounds[l]
        u_lb, u_ub = self.lower_bounds[u], self.upper_bounds[u]

        assert np.all(l_lb <= l_ub)
        assert np.all(u_lb <= u_ub)

        sl_rhs_lb = saturated_linear(rhs_lb, l_lb, u_ub)
        sl_rhs_ub = saturated_linear(rhs_ub, l_lb, u_ub)

        # print(rhs_lb, rhs_ub)
        # print(sl_rhs_lb, sl_rhs_ub)

        if self.obbt:
            obbt_lb, obbt_ub = self.obbt_handler.obbt(rhs_expr)
            rhs_lb, _ = update_lb(obbt_lb, rhs_lb)
            rhs_ub, _ = update_ub(obbt_ub, rhs_ub)

            sl_obbt_lb = saturated_linear(obbt_lb, l_lb, u_ub)
            sl_obbt_ub = saturated_linear(obbt_ub, l_lb, u_ub)

            out_lb = sl_obbt_lb
            out_ub = sl_obbt_ub
        else:
            out_lb = sl_rhs_lb
            out_ub = sl_rhs_ub

        step = SaturatedLinear(
            lhs=out_iterate,
            rhs=rhs_expr,
            lb=rhs_lb,
            ub=rhs_ub,
            relax_binary_vars=relax_binary_vars,
            l=l,
            u=u,
            equality_ranges=equality_ranges,
        )

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_saturated_linear_param_constraints(step, out_lb, out_ub, l_lb, l_ub, u_lb, u_ub)

        if self.obbt:
            self.obbt_handler.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
            self.obbt_handler.add_saturated_linear_param_constraints(step, out_lb, out_ub, l_lb, l_ub, u_lb, u_ub)

        self.add_step(step)

        return out_iterate

    def soft_threshold_step(self, rhs_expr, lambd, relax_binary_vars=False):
        out_iterate = Vector(rhs_expr.get_output_dim())

        rhs_lb, rhs_ub = self.linear_bound_propagation(rhs_expr)

        st_rhs_lb = soft_threshold(rhs_lb, lambd)
        st_rhs_ub = soft_threshold(rhs_ub, lambd)

        if self.obbt:
            # obbt_lb, obbt_ub = self.canonicalizer.obbt(rhs_expr)
            obbt_lb, obbt_ub = self.obbt_handler.obbt(rhs_expr)
            rhs_lb, _ = update_lb(obbt_lb, rhs_lb)
            rhs_ub, _ = update_ub(obbt_ub, rhs_ub)

            st_obbt_lb = soft_threshold(obbt_lb, lambd)
            st_obbt_ub = soft_threshold(obbt_ub, lambd)

            out_lb = st_obbt_lb
            out_ub = st_obbt_ub
        else:
            out_lb = st_rhs_lb
            out_ub = st_rhs_ub

        step = SoftThreshold(
            lhs=out_iterate,
            rhs=rhs_expr,
            lb=rhs_lb,
            ub=rhs_ub,
            relax_binary_vars=relax_binary_vars,
            lambd=lambd,
        )

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        # add new iterate and st constraints to canonicalizer
        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_soft_threshold_constraints(step, out_lb, out_ub)

        if self.obbt:
            self.obbt_handler.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
            self.obbt_handler.add_soft_threshold_constraints(step, out_lb, out_ub)

        self.add_step(step)

        return out_iterate

    def add_constraint_set(self, constraint_set):
        self.constraint_sets.append(constraint_set)

    def equality_constraint(self, lhs_expr, rhs_expr):
        self.canonicalizer.add_equality_constraint(lhs_expr, rhs_expr)

    # TODO: consider if we should make the theory func local to this func and not a verifier property
    def theory_bound(self, k, target_expr, bound_expr, return_improv_frac=True):
        C = self.theory_func(k)
        lb, ub = self.linear_bound_propagation(bound_expr)
        if target_expr.is_leaf:
            self.lower_bounds[target_expr], lb_improv_frac = update_lb(self.lower_bounds[target_expr], lb - C)
            self.upper_bounds[target_expr], ub_improv_frac = update_ub(self.upper_bounds[target_expr], lb + C)
            improv_frac = (lb_improv_frac + ub_improv_frac) / 2.
        self.canonicalizer.add_theory_cut(C, target_expr, lb, ub)

        # TODO: think if we might need to safeguard this with nonleaf theory bounds
        if return_improv_frac:
            return improv_frac

    def linear_bound_propagation(self, expression):
        n = expression.get_output_dim()
        out_lb, out_ub = np.zeros(n), np.zeros(n)

        for k, v in expression.decomposition_dict.items():
            Ax_lower, Ax_upper = interval_bound_prop(v, self.lower_bounds[k], self.upper_bounds[k])

            if np.any(Ax_lower > Ax_upper):
                raise ValueError("Lower bound cannot exceed upper bound .")

            out_lb += Ax_lower
            out_ub += Ax_upper

        if np.any(out_lb > out_ub):
            raise ValueError("Lower bound cannot exceed upper bound .")

        return out_lb, out_ub

    def set_zero_objective(self):
        self.canonicalizer.set_zero_objective()

    def set_infinity_norm_objective(self, expr_list):
        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        lb_list = []
        ub_list = []
        for expression in expr_list:
            lb, ub = self.linear_bound_propagation(expression)
            lb_list.append(lb)
            ub_list.append(ub)

        self.canonicalizer.set_infinity_norm_objective(expr_list, lb_list, ub_list)

    def set_one_norm_objective(self, expr_list):

        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        lb_list = []
        ub_list = []
        for single_expr in expr_list:
            expr_lb, expr_ub = self.linear_bound_prop(single_expr)
            lb_list.append(expr_lb)
            ub_list.append(expr_ub)

        self.canonicalizer.set_one_norm_objective(expr_list, lb_list, ub_list)

    def set_two_norm_objective(self, expr_list):

        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        self.canonicalizer.set_two_norm_objective(expr_list)

    def solve(self, **kwargs):
        # print(self.steps)
        res = self.canonicalizer.solve_model(self.steps, self.lower_bounds, self.upper_bounds, **kwargs)
        return res

    def extract_solver_data(self):
        return self.canonicalizer.get_solve_data()

    def extract_sol(self, iterate):
        return self.canonicalizer.extract_sol(iterate)

    def extract_bounds(self, iterate):
        if iterate.is_leaf:
            return self.lower_bounds[iterate], self.upper_bounds[iterate]
        else:
            raise NotImplementedError

    def post_process(self, var_to_bound, var_for_init_bounds, residual_values, return_improv_frac=True):
        delta_sum = np.sum(residual_values)
        lower_bounds = self.lower_bounds[var_for_init_bounds]
        upper_bounds = self.upper_bounds[var_for_init_bounds]
        new_lower_bounds = lower_bounds - delta_sum
        new_upper_bounds = upper_bounds + delta_sum

        new_lower_bounds, lb_improv_frac = update_lb(lower_bounds, new_lower_bounds)
        new_upper_bounds, ub_improv_frac = update_ub(upper_bounds, new_upper_bounds)

        improv_frac = (lb_improv_frac + ub_improv_frac) / 2.
        self.canonicalizer.post_process(var_to_bound, new_lower_bounds, new_upper_bounds)

        if return_improv_frac:
            return improv_frac


def interval_bound_prop(A, l, u):
    """
    Given :math:`x \\in [l, u]`, give bounds on :math:`Ax`.

    References
    ----------
        .. S. Gowal, K. Dvijotham, R. Stanforth, R. Bunel, C. Qin, J. Uesato, R. Arand-
           jelovi´c, T. Mann, and P. Kohli. "On the effectiveness of interval bound propa-
           gation for training verifiably robust models. arXiv:1810.12715, 2018."
    """

    absA = np.abs(A)
    return (A @ (u + l) - absA @ (u - l)) / 2, (A @ (u + l) + absA @ (u - l)) / 2


def as_array(val, n):
    """
    Ensure val is an array of length n.
    If val is a scalar, return float(val) * np.ones(n).
    If val is already an array, return it unchanged.
    """
    # TODO: This can simply return float(val) * np.ones(n) and not care about val if its an array
    if np.isscalar(val):
        return float(val) * np.ones(n)
    return np.array(val)


def relu(v, proj_ranges=None):
    if proj_ranges is None:
        return np.maximum(v, 0)
    else:
        out = v.copy()

        for left, right in proj_ranges:
            out[left:right] = np.maximum(v[left:right], 0)

        return out


def saturated_linear(x, l, u):
    return np.minimum(np.maximum(x, l), u)


def soft_threshold(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def update_lb(old_lb, new_lb):
    return np.maximum(old_lb, new_lb), (new_lb > old_lb).sum() / old_lb.shape[0]


def update_ub(old_ub, new_ub):
    return np.minimum(old_ub, new_ub), (new_ub < old_ub).sum() / old_ub.shape[0]


def list_upcast(val):
    # if val is a list then keep it as is otherwise put it in a list
    if isinstance(val, list):
        return val
    else:
        return [val]
