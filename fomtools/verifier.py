import logging
from typing import List, Sequence

import numpy as np
import scipy.sparse as spa

from fomtools.canonicalizers.mip import MIP
from fomtools.canonicalizers.obbt_handler import OBBTHandler
from fomtools.canonicalizers.sdp import SDP
from fomtools.canonicalizers.simulator import Simulator
from fomtools.constraint import Constraint
from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import ProxableFunction
from fomtools.stepdecomposers import MIPStepDecomposer, SDPStepDecomposer
from fomtools.stepdecomposers.simulator_step_decomposer import SimulatorStepDecomposer
from fomtools.steps import Affine, Relu, SaturatedLinear, SoftThreshold

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
        self.params = []
        self.steps = []
        self.residuals = []

        self.canonicalizer = self._get_canonicalizer(method, params, solver)
        self.obbt_handler = OBBTHandler(solver=obbt_handler) if obbt_handler else None
        self.use_obbt_handler = True if obbt_handler else False
        if self.obbt_handler is not None:
            self.canonicalizer.obbt_handler = self.obbt_handler
        decomposer_cls = self._get_step_decomposer(method)
        self.step_decomposer = decomposer_cls(self.canonicalizer, self._register_var)
        self._step_proxy = _StepProxy(self)

    def __getitem__(self, item):
        """
        Access summary statistics.

        Parameters
        ----------
        item : str
            Key for the statistic to retrieve.

        Returns
        -------
        float or dict or jnp.ndarray
            The requested statistic.

        Raises
        ------
        KeyError
            If the item is not a valid statistic key.

        RuntimeError
            If no statistic is avaiable.

        Examples
        --------
        >>> vp.solve(K=100)
        >>> vp['mean']
        0.00123
        >>> vp['quantiles'][0.95]
        0.00456
        """
        try:
            summary_statistics = self.canonicalizer.summary_statistics
        except AttributeError:
            raise RuntimeError("No statistics available. Run solve() first.")

        if item not in summary_statistics:
            raise KeyError(
                f"'{item}' is not a valid statistic. Valid keys: {list(summary_statistics.keys())}"
            )

        return summary_statistics[item]

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
             - ``"simulator"``

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
                return SDP(solver_params=solver_params, solver=solver)
            case "simulator":
                return Simulator()
            case _:
                raise ValueError(f"Method must be 'mip', 'sdp' or 'simulator', got {method}")

    @staticmethod
    def _get_step_decomposer(method: str):
        """
        Return the step decomposer class for the specified method.

        Parameters
        ----------
        method : str
            Case-insensitive identifier for the method. Supported values:
            - ``"mip"``
            - ``"sdp"``

        Returns
        -------
        type
            The step decomposer class (MIPStepDecomposer or SDPStepDecomposer).

        Raises
        ------
        ValueError
            If the method is not one of the supported options.
        """
        match method.lower():
            case "mip":
                return MIPStepDecomposer
            case "sdp":
                return SDPStepDecomposer
            case "simulator":
                return SimulatorStepDecomposer
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

    def _register_var(self, var: LinearExpression, **kwargs):
        """
        Register the given :class:`LinearExpression` variable with
        the underlying canonicalizer and OBBT Handler instance.

        Parameters
        ----------
        var : LinearExpression
            A new :class:`LinearExpression` variable to register.

        """
        self.canonicalizer.add_var(var, **kwargs)
        if self.use_obbt_handler:
            self.obbt_handler.add_var(var)

    def _post_process(self, target_expr: LinearExpression, ref_expr: LinearExpression) -> None:
        """
        Apply post-processing bound tightening based on accumulated
        residual values.

        Parameters
        ----------
        target_expr: LinearExpression
            The expression whose bound to update.

        ref_expr: LinearExpression
            The expression that provides the initial lower and
            upper bounds.

        Returns
        -------
        None
        """
        # When delta_sum > 0 (residuals are positive / solver did not maximise), both
        # np.maximum / np.minimum collapse to the original bounds, so this becomes a
        # deliberate no-op — bound tightening is skipped rather than bounds being loosened.
        delta_sum = np.sum(self.residuals)
        _lb = np.maximum(ref_expr.lb - delta_sum, ref_expr.lb)
        _ub = np.minimum(ref_expr.ub + delta_sum, ref_expr.ub)
        self.canonicalizer.update_bounds(target_expr, _lb, _ub)

    def add_iterate(
        self, ndim: int, lb: float = -np.inf, ub: float = np.inf, **kwargs
    ) -> LinearExpression:
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
        self._register_var(iterate, var_type="iterate", **kwargs)
        return iterate

    def add_param(
        self, ndim: int, lb: float = -np.inf, ub: float = np.inf, **kwargs
    ) -> LinearExpression:
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
        self._register_var(var, var_type="param", **kwargs)
        return var

    def add_param_stack(self, param_list: List[LinearExpression]) -> LinearExpression:
        """
        Stack multiple parameters into a single composite LinearExpression.

        Unlike ``add_param``, this does not create a new leaf variable. Instead,
        it builds a composite expression that references the original parameters
        via sparse block-selection matrices. This preserves shared identity: if
        the same parameter appears in multiple stacks, the solver sees one
        variable, not independent copies.

        The constituent parameters must already be registered (via ``add_param``).

        Parameters
        ----------
        param_list: list[LinearExpression]
            List of :class:`LinearExpression` parameters to stack.

        Returns
        -------
        LinearExpression
            A composite expression of dimension ``sum(p.ndim for p in param_list)``
            whose decomposition references the leaves of the input parameters.
        """
        total_dim = sum(p.ndim for p in param_list)
        decomposition_dict = {}
        const = np.zeros(total_dim)
        offset = 0

        for p in param_list:
            # Sparse selection matrix: total_dim x p.ndim
            # Identity block at rows [offset : offset + p.ndim]
            rows = np.arange(offset, offset + p.ndim)
            cols = np.arange(p.ndim)
            data = np.ones(p.ndim)
            sel = spa.csc_matrix((data, (rows, cols)), shape=(total_dim, p.ndim))

            # Compose selection with p's decomposition dict
            for leaf, coeff in p.decomposition_dict.items():
                composed = sel @ coeff
                if leaf in decomposition_dict:
                    decomposition_dict[leaf] = decomposition_dict[leaf] + composed
                else:
                    decomposition_dict[leaf] = composed

            const[offset : offset + p.ndim] = p.const
            offset += p.ndim

        result = LinearExpression(decomposition_dict=decomposition_dict, const=const)
        # Override __init__'s list-based lb/ub with proper numpy arrays
        result.lb, result.ub = result.bound_propagation()
        return result

    def add_constraint(self, constr: Constraint):
        """
        Adds an arbitrary constraint object to the Problem instance.

        This method registers the constraint with both the main canonicalizer
        and the OBBT handler (if enabled). When OBBT is enabled, it also
        performs bound tightening on each leaf expression involved in the
        constraint and propagates the tightened bounds to all relevant
        locations.

        Parameters
        ----------
        constr : Constraint
            A :class:`Constraint` object to add to the VP.

        Returns
        -------
        None
        """
        if self.use_obbt_handler:
            self.obbt_handler.add_constraint(constr)
            self._tighten_constraint_bounds(constr)
        self.canonicalizer.add_constraint(constr)

    def _tighten_constraint_bounds(self, constr: Constraint) -> None:
        """
        Tighten bounds on leaf expressions involved in a constraint using OBBT.

        For each leaf expression in the constraint's decomposition, this method
        runs optimization-based bound tightening and propagates the tightened
        bounds to:
        1. The ``LinearExpression`` object itself
        2. The main canonicalizer's solver
        3. The OBBT handler's solver

        Parameters
        ----------
        constr : Constraint
            The constraint whose leaf expressions should have their bounds
            tightened.

        Returns
        -------
        None
        """
        for leaf_expr in constr.lhs.decomposition_dict.keys():
            # Get current bounds
            lb, ub = leaf_expr.lb, leaf_expr.ub

            # Run OBBT to get potentially tighter bounds
            result = self.obbt_handler.obbt(leaf_expr, lb, ub)
            if result is None:
                continue
            new_lb, new_ub = result

            # Update bounds in all three locations:
            # 1. LinearExpression object
            leaf_expr.update_bounds(new_lb, new_ub)
            # 2. Main canonicalizer's solver
            self.canonicalizer.update_bounds(leaf_expr, new_lb, new_ub)
            # 3. OBBT handler's solver
            self.obbt_handler.update_bounds(leaf_expr, new_lb, new_ub)

    def prox(self, function: ProxableFunction, rhs: LinearExpression, gamma: float = 1, **kwargs):
        """
        Interface for constructing and executing proxable functions.

        This method centralizes the execution pipeline for all proxable
        functions. It instantiates ``function`` with the provided
        keyword arguments, and computes the proximal objective value
        of the provided ``function`` at the current solution value of the
        model.

        Parameters
        ----------
        function: ProxableFunction
            Proxable function class to be instantiated and appplied.

        rhs: LinearExpression
            Linear Expression whose current solution value is used as
            the input to the proximal operator.

        gamma: float, optional
            Proximal step size. Default is ``1``.

        **kwargs
            Additional keyword arguments passed to the constructor of
            ``function``.

        Returns
        -------
        numpy.ndarray
            The proximal objective value corresponding to the implementation of
            ``function``.
        """
        rhs_value = self.get_solution_value(rhs)
        proxable_func = function(**kwargs)
        return proxable_func.prox(rhs_value, gamma)

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

        if self.use_obbt_handler and not step.skip_obbt:
            result = self.obbt_handler.obbt(step.rhs, step.lb, step.ub)
            if result is not None:
                step.update_bounds(*result)

        lhs = self.step_decomposer.process_step(step)
        self.steps.append(step)

        # Track lhs as iterate only if it's a leaf variable
        if lhs.is_leaf:
            self.iterates.append(lhs)

        return lhs

    def fixed_point(self, n_dim, operator, lb=-np.inf, ub=np.inf):
        z_star = self.add_iterate(n_dim, lb=lb, ub=ub)
        s_star = operator(z_star)
        self.add_constraint(z_star == s_star)

        return z_star

    # TODO: consider if we should make the theory func local to this func and not a verifier property
    def theory_bound(self, k, target_expr, ref_expr):
        """
        Apply theory-specific bound relaxation to a target expression.

        Parameters
        ----------
        k: int
            Index used by the theory function to compute the constant
                shift `C`. If it is a leaf node, its lower and upper
            bounds are tightened based on the propagated bounds of
            `bound_expr` and the theory function dependent constant `C`.

        target_expr: LinearExpression
            The expression whose bound to update.

        ref_expr: LinearExpression
            The expression from which bounds are propagated to derive
            new constraints on the target expression.

        Returns
        -------
        float
            The average fractional improvement in the target expression's bound.

        """

        if np.isclose((C := self.theory_func(k)), np.inf):
            return 0

        improv_frac = 0
        lb, ub = ref_expr.bound_propagation()
        if target_expr.is_leaf:
            _lb, _ub = lb - C, lb + C
            target_expr.update_bounds(lb=_lb, ub=_ub)

            improv_frac = (sum(_lb > lb) + sum(_ub < ub)) / (2.0 * len(lb))

        self.canonicalizer.add_theory_cut(C, target_expr, lb, ub)
        return improv_frac

    def set_zero_objective(self):
        """
        Set the optimization objective to zero.

        This method replaces the current objective with a constant
        zero objective, turning it into a feasibility problem.
        """
        self.canonicalizer.set_zero_objective()

    def set_infinity_norm_objective(self, expr_list: Sequence[LinearExpression] | LinearExpression):
        """
        Create an infinity-norm objective over the set of ``LinearExpression``
        objects.

        Parameters
        ----------
        expr_list: Sequence[LinearExpression] | LinearExpression
            A single LinearExpression or a list of LinearExpression objects for
            which the infinity-norm objective will be constructed.

        """
        try:
            len(expr_list)
        except TypeError:
            expr_list = [expr_list]

        lb, ub = zip(*(expr.bound_propagation() for expr in expr_list))
        self.canonicalizer.set_infinity_norm_objective(expr_list, list(lb), list(ub))

    def set_two_norm_objective(self, expr_list: Sequence[LinearExpression] | LinearExpression):
        """
        Create a 2-norm objective over the set of ``LinearExpression``
        objects.

        Parameters
        ----------
        expr_list: Sequence[LinearExpression] | LinearExpression
            A single LinearExpression or a list of LinearExpression objects for
            which the 2-norm objective will be constructed.

        """
        try:
            len(expr_list)
        except TypeError:
            expr_list = [expr_list]

        lb, ub = zip(*(expr.bound_propagation() for expr in expr_list))
        self.canonicalizer.set_two_norm_objective(expr_list, list(lb), list(ub))

    def set_one_norm_objective(
        self, expr_list: Sequence[LinearExpression] | LinearExpression
    ) -> None:
        """
        Create a 1-norm objective over the set of ``LinearExpression`` objects.

        Parameters
        ----------
        expr_list : Sequence[LinearExpression] | LinearExpression
            A single LinearExpression or a list of LinearExpression objects for
            which the 1-norm objective will be constructed.
        """
        try:
            len(expr_list)
        except TypeError:
            expr_list = [expr_list]

        self.canonicalizer.set_one_norm_objective(expr_list)

    def set_objective_norm(
        self,
        expr: Sequence[LinearExpression] | LinearExpression,
        ord: int | float | str,
    ) -> None:
        """
        Set the optimization objective to maximize a norm of the expression(s).

        Parameters
        ----------
        expr : LinearExpression or Sequence[LinearExpression]
            Expression(s) whose norm will be maximized.
        ord : {1, 2, np.inf, 'inf'}
            The norm order:
            - 1: One-norm (sum of absolute values)
            - 2: Two-norm (Euclidean norm squared)
            - np.inf or 'inf': Infinity-norm (maximum absolute value)

        Raises
        ------
        ValueError
            If ord is not one of the supported values.
        """
        match ord:
            case 1:
                self.set_one_norm_objective(expr)
            case 2:
                self.set_two_norm_objective(expr)
            case _ if ord == np.inf or ord == "inf":
                self.set_infinity_norm_objective(expr)
            case _:
                raise ValueError(f"Unsupported norm order: {ord}. Use 1, 2, np.inf, or 'inf'.")

    def solve(self, post_process: bool = False, **kwargs) -> float:
        """
        Solve the optimization problem.

        This method delegates to the canonicalizer to solve the
        optimization problem using the recorded thresholding or
        activation steps.

        Parameters
        ----------
        post_process: bool
            Whether to apply post-processing bound tightening based on accumulated
            residual values

        Returns
        -------
        float
            The objective value for the current solution, if the
            model was solved to optimality.

        Note
        -----
            When `post_process=True`, this function assumes a standard setup in which
            the target expression is derived from the last iterate (``iterates[-1]``)
            and the reference expression is derived from the first iterate (``iterates[0]``).
            These values are hardcoded and cannot be overridden. Callers with a
            non-standard setup (e.g., comparing two intermediate iterates, or supplying
            externally computed expressions) should set `post_process=False` and perform
            post-processing manually.
        """
        # reset residual values
        self.residuals = []
        self.residuals.append(self.canonicalizer.solve_model(**kwargs))
        if post_process:
            self._post_process(target_expr=self.iterates[-1], ref_expr=self.iterates[0])

        return self.residuals[-1]

    def get_solution_value(self, iterate: LinearExpression) -> np.ndarray:
        """
        Get the solution value for the given iterate.

        For each leaf expression in the decomposition of the
        given ``LinearExpression`` object, this method retrieves
        corresponding solution value and computes the dot product
        between the coefficient matrix and the vector of variable values
        associated with that leaf.

        Parameters
        ----------
        iterate: LinearExpression
            A ``LinearExpression`` object to retrieve solution value for.

        Returns
        -------
        np.ndarray[float]
            A 1-D array of shape ``(iterate.ndim,)`` containing the
            evaluated values of the iterate at the current model
            solution.
        """
        return self.canonicalizer.get_solution_value(iterate)


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

    USES_OBBT = {"relu", "satlin", "soft_threshold"}

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
