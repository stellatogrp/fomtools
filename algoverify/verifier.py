import logging

import numpy as np

from algoverify.canonicalizers.mip import MIP
from algoverify.canonicalizers.obbt_handler import OBBTHandler
from algoverify.linearexpression import LinearExpression
from algoverify.steps import Affine, Relu, SaturatedLinear, SoftThreshold

log = logging.getLogger(__name__)


class Verifier(object):
    def __init__(
        self,
        theory_func=None,
        method="mip",
        solver="gurobi",
        obbt_handler="gurobi",
        params=None,
    ):
        self.theory_func = theory_func
        self.iterates = []
        self.params = []
        self.steps = []

        self.canonicalizer = self._get_canonicalizer(method, params, solver)
        self.obbt_handler = OBBTHandler(solver=obbt_handler) if obbt_handler else None

    @staticmethod
    def _get_canonicalizer(method: str, solver_params: dict, solver: dict):
        """
         Return an initialized instance for the requested method.

         Parameters
         ----------
         method: str
             Case-insensitive identifier for the solver to use. Supported values include:
             - ``"mip"``
             - ``"sdp"``

        solver_params: dict
             Key-value pairs with solver parameters and their corresponding
             configuration values.

         solver: str
             Case-insensitive identifier for the solver to use. Supported values
             (and common aliases) include:
             - ``"gurobi"``, ``"grb"``

         Returns
         -------
         Canonicalizer
             A ready-to-use instance of a subclass of :class:`Canonicalizer`
             (e.g. :class:`mip.MIP()`).

         Raises
         ------
         ValueError
             If the requested canonicalizer is one of the supported options.
        """
        match method.lower():
            case "mip":
                return MIP(solver_params=solver_params, solver=solver)
            case "sdp":
                raise NotImplementedError()
            case _:
                raise ValueError(f"Method must be 'mip' or 'sdp', got {method}")

    def _add_var(
        self, ndim: int, lb: list[float] = -np.inf, ub: list[float] = np.inf
    ) -> LinearExpression:
        """
        Create and register a :class:`LinearExpression` variable with the Problem
        instance and its underlying canonicalizer instance.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the variable to create. Defines the size
            of each coefficient matrix in the associated :class:`LinearExpression`.

        lb: list[float]
            Lower bound of the variable.

        ub: list[float]
            Upper bound of the variable.


        Returns
        -------
        LinearExpression
            A new :class:`LinearExpression` of dimenstion `shape`.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """
        if not np.all(lb <= ub):
            raise ValueError("Lower bound must be less than or equal to upper bound.")

        var = LinearExpression.from_int(ndim, lb=lb, ub=ub)

        self.canonicalizer.add_var(var, lb=lb, ub=ub)
        if self.obbt_handler is not None:
            self.obbt_handler.add_var(var, lb=lb, ub=ub)

        return var

    def initial_iterate(
        self, ndim: int, lb: float = -np.inf, ub: float = np.inf
    ) -> LinearExpression:
        """
        Create and add the initial iterate to the Problem instance. This
        method creates a new :class:`LinearExpression` of dimension `shape`
        and registers it with the underlying canonicalizer instance.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the iterate to create. Defines the size
            of each coefficient matrix in the associated :class:`LinearExpression`.

        lb : float
            Lower bound of the initial iterate variable.

        ub : float
            Upper bound of the initial iterate variable.

        Returns
        -------
        LinearExpression
            A new :class:`LinearExpression` of dimension `shape`.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """
        try:
            lb = np.full(ndim, lb)
            ub = np.full(ndim, ub)
        except ValueError:
            raise ValueError(""" Lower bounds and upper bounds must either be scalars or 
                                have the same length as the dimension of the expression.""")

        self.iterates.append(iterate := self._add_var(ndim, lb, ub))
        return iterate

    def add_param(self, ndim: int, lb: float = -np.inf, ub: float = np.inf) -> LinearExpression:
        """
        Create and add a parameter to the Problem instance. This method
        constructs a new :class:`LinearExpression` of dimension `shape`
        and attaches it to the underlying canonicalizer instance as a problem
        parameter.

        Parameters
        ----------
        ndim: int
            Number of dimensions of the parameter to create. Defines the size
            of each coefficient matrix in the associated :class:`LinearExpression`.

        lb: float
            Lower bound of the parameter.

        ub: float
            Upper bound of the parameter.

        Returns
        -------
        LinearExpression
            A new :class:`LinearExpression` of dimension `shape`.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """
        try:
            lb = np.full(ndim, lb)
            ub = np.full(ndim, ub)
        except ValueError:
            raise ValueError(""" Lower bounds and upper bounds must either be scalars or 
                                have the same length as the dimension of the expression.""")

        self.params.append(param := self._add_var(ndim, lb, ub))
        return param

    def add_param_stack(self, param_list):
        """
        Add a stack of parameters to the Problem instance.

        Parameters
        ----------
        param_list: list[LinearExpression]
            List of :class:`LinearExpression` parameters.

        Returns
        -------

        """
        dim = 0
        bounds = [[], []]

        for p in param_list:
            dim += p.ndim
            tuple(map(list.extend, bounds, (p.lb, p.ub)))

        self.params.append(param := self._add_var(dim, lb=bounds[0], ub=bounds[1]))
        return param

    def _get_step(self, step):
        match step:
            case "relu":
                return Relu
            case "satlin":
                return SaturatedLinear
            case "soft_threshold":
                return SoftThreshold
            case "implicit_linear":
                return Affine
            case _:
                pass

    def step(self, rhs_expr, step, relax_binary_vars=False, **kwargs):
        out_iterate = LinearExpression.from_int(rhs_expr.ndim)

        _step = self._get_step(step)(
            lhs=out_iterate,
            rhs=rhs_expr,
            relax_binary_vars=relax_binary_vars,
            **kwargs,
        )

        rhs_lb, rhs_ub = _step.lb, _step.ub
        if self.obbt_handler is not None and step != "implicit_linear":
            rhs_lb, rhs_ub = self.obbt_handler.obbt(rhs_expr, rhs_lb, rhs_ub)

        out_lb = _step.new_bound(rhs_lb)
        out_ub = _step.new_bound(rhs_ub)

        self.iterates.append(out_iterate)
        out_iterate.update_bounds(lb=out_lb, ub=out_ub)

        self.canonicalizer.add_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_constraints(_step.get_constraints())

        if self.obbt_handler is not None:
            self.obbt_handler.add_var(out_iterate, lb=out_lb, ub=out_ub)

        self.steps.append(_step)
        return out_iterate

    def equality_constraint(self, lhs_expr, rhs_expr):
        self.canonicalizer.add_constraint(lhs_expr == rhs_expr)

    def fixed_point(self, n_dim, operator, lb=-100, ub=100):
        z_star = self.initial_iterate(n_dim, lb=lb, ub=ub)
        s_star = operator(z_star)
        self.equality_constraint(z_star, s_star)

        return z_star

    # TODO: consider if we should make the theory func local to this func and not a verifier property
    def theory_bound(self, k, target_expr, bound_expr, return_improv_frac=True):
        C = self.theory_func(k)
        lb, ub = bound_expr.bound_propagation()
        if target_expr.is_leaf:
            _lb, lb_improv_frac = update_lb(target_expr.lb, lb - C)
            _ub, ub_improv_frac = update_ub(target_expr.ub, lb + C)

            target_expr.update_bounds(lb=_lb, ub=_ub)

            improv_frac = (lb_improv_frac + ub_improv_frac) / 2.0
        self.canonicalizer.add_theory_cut(C, target_expr, lb, ub)

        # TODO: think if we might need to safeguard this with nonleaf theory bounds
        if return_improv_frac:
            return improv_frac

    def set_zero_objective(self):
        self.canonicalizer.set_zero_objective()

    def set_infinity_norm_objective(self, expr_list):
        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        lb_list = []
        ub_list = []
        for expression in expr_list:
            lb, ub = expression.bound_propagation()
            lb_list.append(lb)
            ub_list.append(ub)

        self.canonicalizer.set_infinity_norm_objective(expr_list, lb_list, ub_list)

    def solve(self, **kwargs):
        return self.canonicalizer.solve_model(self.steps, **kwargs)

    def extract_sol(self, iterate):
        return self.canonicalizer.extract_sol(iterate)

    def post_process(
        self, var_to_bound, var_for_init_bounds, residual_values, return_improv_frac=True
    ):
        delta_sum = np.sum(residual_values)

        new_lower_bounds, lb_improv_frac = update_lb(
            var_for_init_bounds.lb, var_for_init_bounds.lb - delta_sum
        )
        new_upper_bounds, ub_improv_frac = update_ub(
            var_for_init_bounds.ub, var_for_init_bounds.ub + delta_sum
        )

        self.canonicalizer.post_process(var_to_bound, new_lower_bounds, new_upper_bounds)

        if return_improv_frac:
            return (lb_improv_frac + ub_improv_frac) / 2.0


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


def update_lb(old_lb, new_lb):
    return np.maximum(old_lb, new_lb), (new_lb > old_lb).sum() / old_lb.shape[0]


def update_ub(old_ub, new_ub):
    return np.minimum(old_ub, new_ub), (new_ub < old_ub).sum() / old_ub.shape[0]
