import logging
from typing import List

import numpy as np

from algoverify.canonicalizers.mip import MIP
from algoverify.canonicalizers.obbt_handler import OBBTHandler
from algoverify.linearexpression import LinearExpression
from algoverify.steps import Affine, Relu, SaturatedLinear, SoftThreshold

log = logging.getLogger(__name__)

STEP_CLASSES = {
    "implicit_linear": Affine,
    "relu": Relu,
    "satlin": SaturatedLinear,
    "soft_threshold": SoftThreshold,
}


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
        self.steps = []

        self.canonicalizer = self._get_canonicalizer(method, params, solver)
        self.obbt_handler = OBBTHandler(solver=obbt_handler) if obbt_handler else None
        self.use_obbt_hander = True if obbt_handler else False
        self._step_proxy = _StepProxy(self)

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

    @staticmethod
    def _create_var(
        ndim: int, lb: list[float] = -np.inf, ub: list[float] = np.inf
    ) -> LinearExpression:
        """
        Create a :class:LinearExpression variable with the specified
        dimension, and lower and upper bounds

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

        ValueError
            If the lower or upper bound is neither a scalar nor equal in
            length to the dimension of the expression.
        """
        if not np.all(lb <= ub):
            raise ValueError("Lower bound must be less than or equal to upper bound.")

        try:
            lb = np.full(ndim, lb)
            ub = np.full(ndim, ub)
        except ValueError:
            raise ValueError(""" Lower bounds and upper bounds must either be scalars or 
                                have the same length as the dimension of the expression.""")

        return LinearExpression.from_int(ndim, lb=lb, ub=ub)

    def _register_var(self, var: LinearExpression):
        """
        Register the given :class:`LinearExpression` variable with
        the underlying canonicalizer and OBBT Handler instance.

        Parameters
        ----------
        var : LinearExpression
            A new :class:`LinearExpression` variable to register.

        """
        self.canonicalizer.add_var(var)
        if self.obbt_handler is not None:
            self.obbt_handler.add_var(var)

    def add_iterate(self, ndim: int, lb: float = -np.inf, ub: float = np.inf) -> LinearExpression:
        """
        Create and register an iterate to the Problem instance. This
        method creates a new :class:`LinearExpression` of dimension `ndim`
        and registers it with the underlying canonicalizer and OBBT
        Handler instance.

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

        self.iterates.append(iterate := self._create_var(ndim, lb, ub))
        self._register_var(iterate)
        return iterate

    def add_param(self, ndim: int, lb: float = -np.inf, ub: float = np.inf) -> LinearExpression:
        """
        Create and register a new parameter. This method creates a new
        :class:`LinearExpression` of dimension `ndim` and registers it with
        the underlying canonicalizer and OBBT Handler instance.

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
            A new :class:`LinearExpression` of dimension `ndim`.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.

        ValueError
            If the lower or upper bound is neither a scalar nor equal in
            length to the dimension of the expression.
        """
        var = self._create_var(ndim, lb, ub)
        self._register_var(var)
        return var

    def add_param_stack(self, param_list: List[LinearExpression]) -> LinearExpression:
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

        return self.add_param(dim, lb=bounds[0], ub=bounds[1])

    @property
    def step(self) -> "_StepProxy":
        """
        Interface for constructing and executing steps.

        This property exposes an interface for building
        steps on the ``Verifier``. Attribute access on ``step``
        (e.g. ``verifier.step.relu``) resolves to a callable
        that runs the corresponding step.

        Examples
        -------
        >>> verifier = Verifier()
        >>> expr = verifier.add_param(2)
        >>> x = verifier.step.relu(rhs=expr, relax_binary_vars=True)

        """
        return self._step_proxy

    def run_step(self, rhs, step_cls, relax_binary_vars=False, **kwargs):
        """
        Construct and execute the specified thresholding or activation step.

        This method centralizes the execution pipeline for all step-like
        operators. It instantiates ``step_cls`` with the provided arguments,
        optionally invokes the OBBT handler to refine bounds, and then
        incorporates constraints produced by the respective step object
        into the verification problem.

        Parameters
        ----------
        rhs: LinearExpression
            Right-hand side linear expression in the operator definition.

        step_cls: Step
            The concrete step class to be instantiated. This class encapsulates
            the thresholding or activation operator and is responsible
            for producing the constraints according to the verifier's step interface.

        relax_binary_vars: bool
            If true, binary decision variables associated with the operator
            are relaxed to take continuous values.

        Returns
        -------
            Left-hand side linear expression in the operator definition.
        """
        step = step_cls(rhs=rhs, relax_binary_vars=relax_binary_vars, **kwargs)

        rhs_lb, rhs_ub = step.lb, step.ub
        # TODO: figure out a better way to prevent obbt call for implicit linear step
        if self.use_obbt_hander:
            rhs_lb, rhs_ub = self.obbt_handler.obbt(step.rhs, rhs_lb, rhs_ub)

        step.update_bounds(rhs_lb, rhs_ub)

        lhs = step.lhs

        self._register_var(lhs)
        self.iterates.append(lhs)
        self.canonicalizer.add_constraints(step.get_constraints())
        self.steps.append(step)

        return lhs

    def equality_constraint(self, lhs_expr, rhs_expr):
        self.canonicalizer.add_constraint(lhs_expr == rhs_expr)

    def fixed_point(self, n_dim, operator, lb=-100, ub=100):
        z_star = self.add_iterate(n_dim, lb=lb, ub=ub)
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


def update_lb(old_lb, new_lb):
    return np.maximum(old_lb, new_lb), (new_lb > old_lb).sum() / old_lb.shape[0]


def update_ub(old_ub, new_ub):
    return np.minimum(old_ub, new_ub), (new_ub < old_ub).sum() / old_ub.shape[0]


class _StepProxy:
    """
    Internal helper used by the `Verifier.step` API.

    This proxy enables the syntax ``verifier.step.<step_name>(rhs=..., **kwargs)``
    where ``step_name`` identifies a thresholding or activation step
    class. Attribute access on the proxy (e.g., ``relu``, ``satlin``)
    is resolved dynamically and returns a callable that constructs and runs
    the corresponding step on the underlying ``Verifier` object.

    Notes
    -----
    This class is intended to be created and managed by
    the ``Verifier`` class. Users should not construct
    this class directly.
    """

    def __init__(self, verifier: Verifier):
        self._verifier = verifier

    def __getattr__(self, name):
        """
        Resolve a step name to a callable that executes
        that step. It constructs the corresponding step
        object and delegates execution to the verifier's
        step-running function.

        Parameters
        ----------
        name: str
            The identifier of the step type to execute.

        Returns
        -------
        Callable
            A function that executes the step on the underlying
            ``Verifier`` object.

        Raises
        ------
        AttributeError
            If ``name`` does not correspond to any registered step
            class.
        """
        step_cls = self._get_step_class(name)
        self._verifier.use_obbt_hander = False if name == "implicit_linear" else True

        def wrapper(rhs, relax_binary_vars=False, **kwargs):
            """Execute the specified thresholding or activation step."""
            return self._verifier.run_step(rhs, step_cls, relax_binary_vars, **kwargs)

        return wrapper

    @staticmethod
    def _get_step_class(name: str):
        """
        Return the step class associated with the given identifier.

        This method looks up ``name`` in the internal ``STEP_CLASSES``
        registry and returns the corresponding step class.
        """
        try:
            return STEP_CLASSES[name]
        except KeyError:
            raise AttributeError(f"Unknown step type: '{name}'.")
