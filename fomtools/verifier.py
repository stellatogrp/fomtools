import logging
import time
import warnings
from contextlib import contextmanager
from typing import List, Sequence

import numpy as np

from fomtools.canonicalizers.mip import MIP
from fomtools.canonicalizers.obbt_handler import OBBTHandler
from fomtools.canonicalizers.sdp import SDP
from fomtools.canonicalizers.simulator import Simulator
from fomtools.constraint import (
    ComplementarityConstraint,
    Constraint,
    Norm,
    SumSquares,
    complementarity,
    is_binary_free,
    sum_squares,
)
from fomtools.linearexpression import LinearExpression, stack
from fomtools.nn.layers import LinearLayer, ReLULayer
from fomtools.proxable_functions import ProxableFunction
from fomtools.sets import Set
from fomtools.solvers.factory import is_mip_solver, solver_class, verbosity_flag
from fomtools.stepdecomposers import MIPStepDecomposer, SDPStepDecomposer
from fomtools.stepdecomposers.simulator_step_decomposer import SimulatorStepDecomposer
from fomtools.steps import (
    Affine,
    LinearProgram,
    QuadraticProgram,
    Relu,
    SaturatedLinear,
    SoftThreshold,
    StandardFormLinearProgram,
)

log = logging.getLogger(__name__)


def _deprecated(old: str, new: str) -> None:
    """Warn that ``old`` is deprecated, pointing at the user's call two frames up."""
    warnings.warn(f"{old} is deprecated; use {new}.", DeprecationWarning, stacklevel=3)


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
        solver="highs",
        obbt_solver="auto",
        params=None,
        rlt_cuts=False,
        triangle_cuts=False,
        ineq_product_cuts=True,
        ineq_box_cuts=True,
        ineq_square_cuts=True,
        relu_envelope_cuts=False,
        rlt_all_pairs=False,
        full_psd_cone=False,
        cross_cut=False,
        subgrad_mono_cut=False,
        obbt_params=None,
        verbose=False,
        tighten_residuals=True,
    ):
        """
        Initialize a Verifier instance.

        Parameters
        ----------
        theory_func : callable, optional
            Theoretical convergence function used by ``theory_bound``.
        method : {"mip", "sdp", "simulator"}, default "mip"
            Canonicalization method for the verification problem.
        solver : str, default "highs"
            Solver backend identifier (e.g., "highs", "gurobi", "clarabel").
        obbt_solver : str or None, default "auto"
            Solver used by the OBBT bound-tightening handler. ``"auto"``
            (the default) uses ``solver`` when it is a valid OBBT backend
            (Gurobi or HiGHS) and disables OBBT otherwise. Disable
            explicitly with ``None``.
        params : dict, optional
            Solver-specific parameters forwarded to the canonicalizer. Only the
            verbosity key (``OutputFlag`` for Gurobi, ``output_flag`` for HiGHS)
            is inherited by the OBBT solver, unless ``obbt_params`` is given. A
            verbosity key here overrides ``verbose``.
        rlt_cuts : bool, default False
            SDP-only: when True, add RLT cuts between cone-mate index
            pairs (paper §4.5 eq 18). No effect for ``method != "sdp"``.
        triangle_cuts : bool, default False
            SDP-only: when True, add lifted triangle cuts for mixed-bound
            ReLU outputs (paper §4.5 eq 25/26). No effect for
            ``method != "sdp"``.
        ineq_product_cuts : bool, default True
            SDP-only: add off-diagonal products ``g_i g_j >= 0`` for each
            inequality constraint ``Ax + b <= 0``. No effect for ``method != "sdp"``.
        ineq_box_cuts : bool, default True
            SDP-only: add RLT products of each inequality constraint with the
            variable box bounds. No effect for ``method != "sdp"``.
        ineq_square_cuts : bool, default True
            SDP-only: add the self-square upper bound ``g_i^2 <= g_i_lower * g_i``
            for each inequality constraint. No effect for ``method != "sdp"``.
        relu_envelope_cuts : bool, default False
            SDP-only: add the linear ReLU convex-hull inequalities (lower
            envelope and mixed-row upper chord) on the first-moment variables.
            No effect for ``method != "sdp"``.
        rlt_all_pairs : bool, default False
            SDP-only: RLT every variable pair by coupling each via a ``{0,p,q}``
            3x3 PSD minor (the middle tier between cone-mate ``rlt_cuts`` and the
            dense ``full_psd_cone``). No effect for ``method != "sdp"``.
        full_psd_cone : bool, default False
            SDP-only: union all PSD cones into one full moment matrix; combined
            with ``rlt_cuts`` gives the full Shor + RLT relaxation. No effect for
            ``method != "sdp"``.
        cross_cut : bool, default False
            SDP-only: add the PEP "cross" interpolation cuts ``z_a r_b <= 0``
            coupling pairs of ReLU output components across steps; for
            saturated-linear (box projection) steps it also adds the box
            normal-cone cuts. Self-registers minimal PSD cones (no need for
            ``full_psd_cone``). No effect for ``method != "sdp"``.
        subgrad_mono_cut : bool, default False
            SDP-only: add the PEP subgradient-monotonicity cuts
            ``<s^i/tau_i - s^j/tau_j, z^i> >= 0`` over pairs of prox steps; for
            saturated-linear steps it also adds the box firm-nonexpansiveness /
            monotonicity cuts. On the orthant the prox form reduces to
            ``cross_cut``; on soft-thresholding it is a distinct coupling. No effect
            for ``method != "sdp"``.
        obbt_params : dict, optional
            Parameters passed verbatim to the OBBT solver. The OBBT solver also
            inherits the verbosity setting of ``params`` (translated between
            backends) or ``verbose``, unless ``obbt_params`` sets its own.
        verbose : bool, default False
            Print the solver logs, for the main solves and for the OBBT LPs.
        tighten_residuals : bool, default True
            MIP only. After each solve whose objective is ``ft.norm(a - b, np.inf)``,
            chain the certified bound with the earlier ones, ``||a - z^0||_inf <= sum
            of the bounds from z^0 to a``, and tighten ``a`` with it (the triangle
            inequality behind `post_process`). Later steps then start from tighter
            boxes. The worst case is unchanged; only the MIPs get easier.
        """
        self.theory_func = theory_func
        self.method = method.lower()
        self.iterates = []
        self.params = []
        # Simulator parameters registered without samples (allowed, e.g. for code
        # generation, which never simulates; ``solve`` then asks for samples).
        self._unsampled_params = []
        self.steps = []
        # The most recently set objective as ``(expressions, ord)``, or ``None``
        # for none / the zero objective. Code generation reads it to stop the
        # generated solver on the verified metric (TerminationKind.OBJECTIVE).
        self.objective: tuple[tuple[LinearExpression, ...], int | float] | None = None
        # Successive-residual bounds for ``tighten_residuals``: ``(a, b)`` of the current
        # objective ``||a - b||_inf``, and, per expression counter, ``(root, radius)``
        # with a certified ``||expr - root||_inf <= radius``.
        self.tighten_residuals = tighten_residuals and self.method == "mip"
        self._residual = None
        self._residual_chain = {}

        # An explicit verbosity key in ``params`` (any case, as in Gurobi) overrides ``verbose``.
        option = self._verbosity_option(method, solver, verbose)
        if any(k.lower() == key.lower() for k in params or {} for key in option):
            option = {}
        params = {**option, **(params or {})} or None
        self._solver_name = solver
        self.canonicalizer = self._get_canonicalizer(
            method,
            params,
            solver,
            rlt_cuts=rlt_cuts,
            triangle_cuts=triangle_cuts,
            ineq_product_cuts=ineq_product_cuts,
            ineq_box_cuts=ineq_box_cuts,
            ineq_square_cuts=ineq_square_cuts,
            relu_envelope_cuts=relu_envelope_cuts,
            rlt_all_pairs=rlt_all_pairs,
            full_psd_cone=full_psd_cone,
            cross_cut=cross_cut,
            subgrad_mono_cut=subgrad_mono_cut,
        )
        if obbt_solver == "auto":
            obbt_solver = solver if is_mip_solver(solver) else None
        if obbt_solver:
            # Explicit ``obbt_params`` keys win over the inherited verbosity.
            inherited = self._obbt_params_from(params, obbt_solver, verbose) or {}
            obbt_params = {**inherited, **(obbt_params or {})} or None
        self.obbt_handler = (
            OBBTHandler(solver=obbt_solver, solver_params=obbt_params) if obbt_solver else None
        )
        self.use_obbt_handler = True if obbt_solver else False
        if self.obbt_handler is not None:
            self.canonicalizer.obbt_handler = self.obbt_handler
        decomposer_cls = self._get_step_decomposer(method)
        self.step_decomposer = decomposer_cls(
            self.canonicalizer,
            self._register_var,
            self._add_aux_iterate,
            self.add_constraint,
        )
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
        >>> vp.solve()
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
    def _get_canonicalizer(
        method: str,
        solver_params: dict,
        solver: dict,
        rlt_cuts: bool = False,
        triangle_cuts: bool = False,
        ineq_product_cuts: bool = True,
        ineq_box_cuts: bool = True,
        ineq_square_cuts: bool = True,
        relu_envelope_cuts: bool = False,
        rlt_all_pairs: bool = False,
        full_psd_cone: bool = False,
        cross_cut: bool = False,
        subgrad_mono_cut: bool = False,
    ):
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

         rlt_cuts: bool, default=False
             SDP-only: enable RLT cuts between cone-mate indices.

         triangle_cuts: bool, default=False
             SDP-only: enable lifted triangle cuts for mixed-bound ReLU outputs.

         Returns
         -------
         Canonicalizer
             A ready-to-use instance of a subclass of `Canonicalizer`
             (e.g. `mip.MIP()`).

         Raises
         ------
         ValueError
             If the requested canonicalizer is not one of the supported options.
        """
        method_lower = method.lower()
        sdp_only_flags = [
            ("rlt_cuts", rlt_cuts),
            ("triangle_cuts", triangle_cuts),
            ("relu_envelope_cuts", relu_envelope_cuts),
            ("rlt_all_pairs", rlt_all_pairs),
            ("full_psd_cone", full_psd_cone),
            ("cross_cut", cross_cut),
            ("subgrad_mono_cut", subgrad_mono_cut),
        ]
        if method_lower != "sdp" and any(v for _, v in sdp_only_flags):
            enabled = [name for name, v in sdp_only_flags if v]
            warnings.warn(
                f"{', '.join(enabled)} is SDP-only and will be ignored for method={method!r}.",
                UserWarning,
                stacklevel=3,
            )

        match method_lower:
            case "mip":
                return MIP(solver_params=solver_params, solver=solver)
            case "sdp":
                return SDP(
                    solver_params=solver_params,
                    solver=solver,
                    rlt_cuts=rlt_cuts,
                    triangle_cuts=triangle_cuts,
                    ineq_product_cuts=ineq_product_cuts,
                    ineq_box_cuts=ineq_box_cuts,
                    ineq_square_cuts=ineq_square_cuts,
                    relu_envelope_cuts=relu_envelope_cuts,
                    rlt_all_pairs=rlt_all_pairs,
                    full_psd_cone=full_psd_cone,
                    cross_cut=cross_cut,
                    subgrad_mono_cut=subgrad_mono_cut,
                )
            case "simulator":
                return Simulator()
            case _:
                raise ValueError(f"Method must be 'mip', 'sdp' or 'simulator', got {method}")

    @staticmethod
    def _verbosity_option(method: str, solver: str, verbose: bool, explicit: bool = False) -> dict:
        """
        The native solver option that turns the log on or off, if the backend has one.

        Clarabel and CVXPY are silent by default, so for the constructor (``explicit``
        False) their option is only added to turn the log on; a solve-time ``verbose``
        always sets it. CuClarabel takes no options.
        """
        if method.lower() == "mip":
            cls = solver_class(solver)
            if cls is not None and cls.VERBOSITY_KEY is not None:
                return {cls.VERBOSITY_KEY: verbose}
        elif method.lower() == "sdp" and (verbose or explicit) and solver.lower() != "cuclarabel":
            return {"verbose": verbose}
        return {}

    @staticmethod
    def _obbt_params_from(
        params: dict | None, obbt_solver: str, verbose: bool = False
    ) -> dict | None:
        """
        Extract the verbosity setting from the main solver ``params`` for the OBBT solver.

        The rest of ``params`` (time limits, gaps, SDP-solver options) is not forwarded:
        it targets the main solve and may be invalid for the OBBT backend. Without a
        verbosity key in ``params``, the OBBT solver follows ``verbose``.
        """
        cls = solver_class(obbt_solver)
        if cls is None:
            return None
        own = cls.verbosity_params(params)
        if own:
            return own
        # Translate the other MIP backend's verbosity key.
        flag = verbosity_flag(params)
        return {cls.VERBOSITY_KEY: verbose if flag is None else flag}

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
                raise ValueError(f"Method must be 'mip', 'sdp', or 'simulator', got {method}")

    @staticmethod
    def _create_var(
        ndim: int, lb: list[float] = -np.inf, ub: list[float] = np.inf
    ) -> LinearExpression:
        """
        Create a `LinearExpression` variable with the specified
        dimension, and lower and upper bounds

        Parameters
        ----------
        ndim : int
            Number of dimensions of the variable to create. Defines the size
            of each coefficient matrix in the associated `LinearExpression`.

        lb: list[float]
            Lower bound of the variable.

        ub: list[float]
            Upper bound of the variable.


        Returns
        -------
        LinearExpression
            A new `LinearExpression` of dimenstion `shape`.

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
        Register the given `LinearExpression` variable with
        the underlying canonicalizer and OBBT Handler instance.

        Parameters
        ----------
        var : LinearExpression
            A new `LinearExpression` variable to register.

        """
        self.canonicalizer.add_var(var, **kwargs)
        var._verifier = self
        if self.use_obbt_handler:
            self.obbt_handler.add_var(var)

    def iterate(
        self, ndim: int, lb: float = -np.inf, ub: float = np.inf, **kwargs
    ) -> LinearExpression:
        """
        Create an iterate, typically the initial iterate, in the box ``[lb, ub]``.

        Equal bounds fix the iterate. The iterate is a new `LinearExpression` of
        dimension ``ndim``, registered with the canonicalizer and the OBBT handler.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the iterate to create. Defines the size
            of each coefficient matrix in the associated `LinearExpression`.

        lb : float
            Lower bound of the initial iterate variable.

        ub : float
            Upper bound of the initial iterate variable.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` of dimension `shape`.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """

        iterate = self._add_aux_iterate(ndim, lb, ub, **kwargs)
        if self.method == "simulator":
            self.canonicalizer.record_iterate(iterate)
        return iterate

    def add_iterate(self, *args, **kwargs) -> LinearExpression:
        """Deprecated alias of `iterate`."""
        _deprecated("add_iterate", "Verifier.iterate")
        return self.iterate(*args, **kwargs)

    def _add_aux_iterate(self, ndim, lb=-np.inf, ub=np.inf, **kwargs):
        """Register a leaf without making an internal KKT variable a user iterate."""
        self.iterates.append(iterate := self._create_var(ndim, lb, ub))
        self._register_var(iterate, var_type="iterate", **kwargs)
        return iterate

    def _record_output(self, output):
        if output.is_leaf and not any(output is it for it in self.iterates):
            self.iterates.append(output)
        if self.method == "simulator":
            self.canonicalizer.record_iterate(output)
        return output

    def parameter(
        self, ndim: int, lb: float = -np.inf, ub: float = np.inf, **kwargs
    ) -> LinearExpression:
        """
        Create a problem parameter in the box ``[lb, ub]``.

        The parameter is a new `LinearExpression` of dimension ``ndim``, registered
        with the canonicalizer and the OBBT handler. With ``method="simulator"``,
        pass ``samples`` of shape ``(num_samples, ndim)``.

        Parameters
        ----------
        ndim: int
            Number of dimensions of the parameter to create. Defines the size
            of each coefficient matrix in the associated `LinearExpression`.

        lb: float
            Lower bound of the parameter.

        ub: float
            Upper bound of the parameter.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` of dimension `ndim`.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.

        ValueError
            If the lower or upper bound is neither a scalar nor equal in
            length to the dimension of the expression.
        """
        var = self._create_var(ndim, lb, ub)
        if self.method == "simulator":
            samples = kwargs.get("samples")
            if samples is not None:
                shape = np.shape(samples)
                if len(shape) != 2 or shape[1] != ndim:
                    raise ValueError(
                        f"parameter(ndim={ndim}): samples must have shape "
                        f"(num_samples, {ndim}), got {shape}."
                    )
            elif not (np.all(np.isfinite(var.lb)) and np.array_equal(var.lb, var.ub)):
                self._unsampled_params.append(var)
        self._register_var(var, var_type="param", **kwargs)
        return var

    def add_param(self, *args, **kwargs) -> LinearExpression:
        """Deprecated alias of `parameter`."""
        _deprecated("add_param", "Verifier.parameter")
        return self.parameter(*args, **kwargs)

    def add_param_stack(self, param_list: List[LinearExpression]) -> LinearExpression:
        """Deprecated: use ``ft.stack(param_list)``."""
        _deprecated("add_param_stack", "ft.stack")
        return stack(param_list)

    def iterate_trajectory(self, length: int, collapse_expr="auto"):
        """Create a backend-aware iterate trajectory of ``length`` slots.

        Returns an `IterateTrajectory` that
        behaves like a fixed-length list of iterates. Assigning a collapsed
        affine expression (``z[k] = M @ z[k - 1] + c``) is handled per backend:
        for MIP/simulator the expression stays collapsed (no new variable), and
        for SDP a fresh leaf linked by an equality is minted so the moment
        matrix stays sparse (smaller PSD cones).

        ``'auto'`` also stops collapsing once the assignments are seen to
        *accumulate* -- ``z[k] = z[k - 1] + ut - u``, as in Douglas-Rachford and
        ADMM, where each slot would otherwise reference every leaf the previous
        one did. That growth is what makes such a method expensive to generate
        code from, so keeping each iterate as its own variable is what lets
        `extract_unrolled_ir` handle it. See
        `IterateTrajectory` for details.

        Parameters
        ----------
        length : int
            Number of slots (e.g. ``K + 1`` for ``K`` steps).
        collapse_expr : {'auto', True, False}, default 'auto'
            Trajectory-level default collapse policy. ``'auto'`` collapses on
            MIP/simulator, keeps a separate leaf on SDP, and keeps a separate leaf
            once the assignments accumulate.

        Returns
        -------
        IterateTrajectory
            A list-like trajectory bound to this verifier.

        Examples
        --------
        >>> z = verifier.iterate_trajectory(K + 1)
        >>> z[0] = verifier.iterate(n, lb=z0, ub=z0)
        >>> for k in range(1, K + 1):
        ...     z[k] = verifier.relu(z[k - 1] - eta * A.T @ (A @ z[k - 1] - x))
        """
        from fomtools.iteratetrajectory import IterateTrajectory

        return IterateTrajectory(self, length, collapse_expr)

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
            A `Constraint` object to add to the VP.

        Returns
        -------
        None
        """
        # Only forward LINEAR, binary-free constraints to the OBBT handler so its
        # model stays a pure LP. Binary leaves would otherwise be relaxed silently,
        # and non-convex constraints (e.g. complementarity, is_linear=False) cannot
        # be represented in the LP at all.
        if self.use_obbt_handler and getattr(constr, "is_linear", True) and is_binary_free(constr):
            self.obbt_handler.add_constraint(constr)
            self._tighten_constraint_bounds(constr)
        self.canonicalizer.add_constraint(constr)

    def _add_cuts(self, cuts: List[Constraint]) -> None:
        """
        Add valid linear cuts to the model and, if OBBT is enabled, to the OBBT LP.

        Like `add_constraint` for a batch, except that OBBT runs only once on
        the union of the leaves the cuts touch, rather than once per cut.

        Parameters
        ----------
        cuts : List[Constraint]
            Linear, binary-free constraints valid for every feasible point.

        Returns
        -------
        None
        """
        for cut in cuts:
            self.canonicalizer.add_constraint(cut)
        if not self.use_obbt_handler:
            return
        self.obbt_handler.add_constraints(cuts)
        leaves = {}
        for cut in cuts:
            for leaf in cut.lhs.decomposition_dict:
                leaves.setdefault(id(leaf), leaf)
        self._tighten_leaf_bounds(leaves.values())

    def complementarity(
        self, expr1: LinearExpression, expr2: LinearExpression
    ) -> ComplementarityConstraint:
        """Deprecated: use ``ft.complementarity(expr1, expr2)``."""
        _deprecated("vp.complementarity", "ft.complementarity")
        return complementarity(expr1, expr2)

    def sum_squares(self, expr: LinearExpression) -> SumSquares:
        """Deprecated: use ``ft.sum_squares(expr)``."""
        _deprecated("vp.sum_squares", "ft.sum_squares")
        return sum_squares(expr)

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
        self._tighten_leaf_bounds(constr.lhs.decomposition_dict.keys())

    def _tighten_leaf_bounds(self, leaves) -> None:
        """
        Run OBBT once on each of ``leaves`` and propagate the tightened bounds to
        the ``LinearExpression``, the main canonicalizer's solver and the OBBT
        handler's solver.

        Parameters
        ----------
        leaves : Iterable[LinearExpression]
            The leaf expressions to tighten.

        Returns
        -------
        None
        """
        for leaf_expr in leaves:
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

    def prox(
        self,
        function: ProxableFunction,
        arg: LinearExpression,
        gamma: float = 1.0,
    ) -> LinearExpression:
        """
        Encode a proxable function as steps + auxiliary variables.

        Dispatches to the active step decomposer's prox registry, which
        lowers ``function`` into one or more steps (and any auxiliary
        variables those steps require) and returns the output expression.

        Parameters
        ----------
        function : ProxableFunction
            A pre-configured proxable function instance carrying problem
            data (e.g. ``NormL1(lambd=0.5)``, ``IndBox(l=l, u=u)``).
        arg : LinearExpression
            The input expression to which the prox is applied.
        gamma : float, default 1.0
            The proximal step size.

        Returns
        -------
        LinearExpression
            The output expression representing
            ``prox_{gamma * function}(arg)``.

        Raises
        ------
        NotImplementedError
            If the active backend's step decomposer does not have an
            entry for the concrete type of ``function`` in its
            ``_PROX_DECOMPOSERS`` registry.
        """
        return self._record_output(self.step_decomposer.process_prox(function, arg, gamma))

    def project(self, proj_set, arg: LinearExpression) -> LinearExpression:
        r"""
        Project ``arg`` onto a convex set.

        A friendly, math-named front door over `prox`: for a set $C$ with
        indicator $I_C$, the Euclidean projection is exactly
        $\Pi_C(x) = \operatorname{prox}_{I_C}(x)$. This method simply calls
        ``self.prox(proj_set.indicator(), arg)``, reusing the entire prox lowering
        (and inheriting each backend's support, including the simulator's
        ``NotImplementedError`` for KKT-based projections).

        Parameters
        ----------
        proj_set : Set or type[Set]
            The set to project onto, e.g. ``Box(l, u)`` or ``Polyhedron(A, b, dual_ub=...)``.
            A parameterless `Set` subclass may be passed as a
            bare class (e.g. ``NonNegative``), which is instantiated automatically.
        arg : LinearExpression
            The input expression to project.

        Returns
        -------
        LinearExpression
            The output expression representing $\Pi_{C}(\text{arg})$.

        Raises
        ------
        TypeError
            If ``proj_set`` is not a `Set` (or subclass) —
            penalty/loss functions are not projections, use `prox` for those; or
            if a *parameterized* Set class is passed as the bare-class shorthand (the
            shorthand only works for parameterless Sets like ``NonNegative``).

        Notes
        -----
        Projection is independent of the proximal step size, so there is no ``gamma``
        argument.
        """
        if isinstance(proj_set, type) and issubclass(proj_set, Set):
            try:
                proj_set = proj_set()
            except TypeError as exc:
                raise TypeError(
                    f"project(): the bare-class shorthand only works for parameterless "
                    f"Sets; {proj_set.__name__} requires constructor arguments — "
                    f"instantiate it explicitly, e.g. project({proj_set.__name__}(...), arg)."
                ) from exc
        if not isinstance(proj_set, Set):
            raise TypeError(
                "project expects a Set (or a parameterless Set subclass); got "
                f"{type(proj_set).__name__}. For penalties/losses use prox() directly."
            )
        return self.prox(proj_set.indicator(), arg)

    def quadratic_program(
        self,
        P,
        q=None,
        A=None,
        b=None,
        C=None,
        d=None,
        dual_ub=None,
        primal_bounds=None,
        nu_bounds=None,
        arg: LinearExpression = None,
        relax_binary_vars: bool = False,
    ) -> LinearExpression:
        r"""
        Encode the solution of an arbitrary convex QP as a step.

        Solves (in the verification model, via the KKT + complementarity system)

        $$
        x^\star = \arg\min_x \tfrac12 x^\top P x + q^\top x
        \quad\text{s.t.}\quad A x = b,\; C x \le d,
        $$

        where ``P`` is symmetric PSD (dense or scipy.sparse) and ``q``, ``b``, ``d``
        may be numeric or `LinearExpression` parameters. Unlike ``run_step``,
        this takes the parametric inputs directly (a QP has up to three) rather than a
        single positional ``arg``; the ``LinearExpression`` inputs already carry their
        dependence on prior iterates.

        Parameters
        ----------
        P, q, A, b, C, d, dual_ub, primal_bounds, nu_bounds, arg, relax_binary_vars
            See `fomtools.steps.QuadraticProgram`. ``dual_ub`` is required when
            inequality constraints ``C`` are present.

        Returns
        -------
        LinearExpression
            The primal solution ``x``.
        """
        step = QuadraticProgram(
            P=P,
            q=q,
            A=A,
            b=b,
            C=C,
            d=d,
            dual_ub=dual_ub,
            primal_bounds=primal_bounds,
            nu_bounds=nu_bounds,
            arg=arg,
            relax_binary_vars=relax_binary_vars,
        )
        # Composite step: the decomposer creates the primal x via iterate (already
        # tracked in self.iterates) and returns it; just record the step itself.
        x = self.step_decomposer.process_step(step)
        self.steps.append(step)
        return self._record_output(x)

    def linear_program(
        self,
        q,
        A=None,
        b=None,
        C=None,
        d=None,
        dual_ub=None,
        primal_bounds=None,
        nu_bounds=None,
        arg: LinearExpression = None,
        relax_binary_vars: bool = False,
    ) -> LinearExpression:
        r"""
        Encode the solution of a linear program (`QuadraticProgram` with
        ``P = 0``) as a step:

        $$
        x^\star = \arg\min_x q^\top x \quad\text{s.t.}\quad A x = b,\; C x \le d.
        $$

        See `quadratic_program`. ``q`` is required and its dimension sizes the LP;
        finite primal bounds must come from ``primal_bounds`` or OBBT (stationarity does
        not pin ``x`` when ``P = 0``).
        """
        step = LinearProgram(
            q=q,
            A=A,
            b=b,
            C=C,
            d=d,
            dual_ub=dual_ub,
            primal_bounds=primal_bounds,
            nu_bounds=nu_bounds,
            arg=arg,
            relax_binary_vars=relax_binary_vars,
        )
        x = self.step_decomposer.process_step(step)
        self.steps.append(step)
        return self._record_output(x)

    def standard_form_linear_program(
        self,
        c,
        A=None,
        b=None,
        dual_ub=None,
        primal_bounds=None,
        nu_bounds=None,
        arg: LinearExpression = None,
        relax_binary_vars: bool = False,
    ) -> LinearExpression:
        r"""
        Encode the solution of a standard-form linear program as a step:

        $$
        x^\star = \arg\min_x c^\top x \quad\text{s.t.}\quad A x = b,\; x \ge 0.
        $$

        A `LinearProgram` with ``q = c``, equality ``(A, b)`` and
        ``x \ge 0`` written as ``-I x \le 0``. See `linear_program`. ``dual_ub`` is
        required (the ``x \ge 0`` inequality is always present); finite primal bounds must
        come from ``primal_bounds`` or OBBT.
        """
        step = StandardFormLinearProgram(
            c=c,
            A=A,
            b=b,
            dual_ub=dual_ub,
            primal_bounds=primal_bounds,
            nu_bounds=nu_bounds,
            arg=arg,
            relax_binary_vars=relax_binary_vars,
        )
        x = self.step_decomposer.process_step(step)
        self.steps.append(step)
        return self._record_output(x)

    @property
    def step(self) -> "_StepProxy":
        """
        Deprecated: call the step directly, e.g. ``verifier.relu(expr)``.

        ``verifier.step.<name>(arg, **kwargs)`` runs the step of that name and warns.
        """
        return self._step_proxy

    def run_step(self, arg, step_cls, relax_binary_vars=False, **kwargs):
        """
        Construct and execute the specified thresholding or activation step.

        This method centralizes the execution pipeline for all step-like
        operators. It instantiates ``step_cls`` with the provided
        arguments and forwards it to the step decomposer, which handles
        OBBT bound tightening, decomposition, and canonicalization.

        Parameters
        ----------
        arg: LinearExpression
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
        step = step_cls(arg=arg, relax_binary_vars=relax_binary_vars, **kwargs)

        # OBBT input-tightening + relaxation building happens per-primitive in
        # StepDecomposer._process_primitive, so decomposed sub-steps are covered.
        output = self.step_decomposer.process_step(step)
        self.steps.append(step)

        return self._record_output(output)

    def relu(self, arg, *, proj_ranges=None, relax_binary_vars=False) -> LinearExpression:
        r"""
        Projection onto the nonnegative orthant, $\Pi_{\mathbf{R}^n_+}(\text{arg})$.

        See `Relu` for the parameters.
        """
        return self.run_step(arg, Relu, relax_binary_vars, proj_ranges=proj_ranges)

    def satlin(self, arg, *, l, u, relax_binary_vars=False) -> LinearExpression:
        r"""
        Projection onto the box $[l, u]$, $\Pi_{[l,u]}(\text{arg})$.

        See `SaturatedLinear` for the parameters.
        """
        return self.run_step(arg, SaturatedLinear, relax_binary_vars, l=l, u=u)

    def soft_threshold(self, arg, *, lambd, relax_binary_vars=False) -> LinearExpression:
        r"""
        Soft-thresholding $\mathbf{soft}_\lambda(\text{arg})$, the prox of $\lambda\|\cdot\|_1$.

        See `SoftThreshold` for the parameters.
        """
        return self.run_step(arg, SoftThreshold, relax_binary_vars, lambd=lambd)

    def implicit_linear(
        self, arg, *, output_mat=None, weight=None, bias=None, relax_binary_vars=False
    ) -> LinearExpression:
        """
        Implicit affine step ``output_mat @ w = weight @ arg + bias``.

        See `Affine` for the parameters.
        """
        return self.run_step(
            arg, Affine, relax_binary_vars, output_mat=output_mat, weight=weight, bias=bias
        )

    def fixed_point(self, ndim=None, operator=None, lb=-np.inf, ub=np.inf, *, n_dim=None):
        """
        Encode a fixed point of an operator as a new iterate.

        Creates a new iterate ``z_star`` of dimension `ndim`, applies
        `operator` to it, and constrains ``z_star == operator(z_star)``.
        Used to encode distance-to-optimality objectives, where `z_star`
        represents an optimal solution of the parametric problem.

        Parameters
        ----------
        ndim: int
            Dimension of the fixed-point iterate. (``n_dim`` is a deprecated alias.)

        operator: Callable[[LinearExpression], LinearExpression]
            The algorithm operator `T`, applied to the new iterate to
            build the fixed-point constraint ``z_star == T(z_star)``.

        lb: float, default -np.inf
            Lower bound of the fixed-point iterate. Must hold for every
            parameter in the declared parameter set.

        ub: float, default np.inf
            Upper bound of the fixed-point iterate. Must hold for every
            parameter in the declared parameter set.

        Returns
        -------
        LinearExpression
            The fixed-point iterate ``z_star``.
        """
        if n_dim is not None:
            _deprecated("fixed_point(n_dim=...)", "fixed_point(ndim=...)")
            ndim = n_dim
        if ndim is None or operator is None:
            raise TypeError("fixed_point needs ndim and operator.")
        z_star = self.iterate(ndim, lb=lb, ub=ub)
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
            Index passed to the theory function. Its nonnegative value C must
            uniformly bound the infinity-norm distance from target to reference.

        target_expr: LinearExpression
            The expression whose bound to update.

        ref_expr: LinearExpression
            The expression from which bounds are propagated to derive
            new constraints on the target expression.

        Returns
        -------
        float
            The fraction of target lower and upper bound entries tightened.

        """

        return self._apply_reference_bound(target_expr, ref_expr, self.theory_func(k))

    def _apply_reference_bound(self, target_expr, ref_expr, radius):
        """Intersect a target box with a caller-justified reference box expansion."""
        radius = float(radius)
        if np.isnan(radius) or radius < 0:
            raise ValueError("A reference-bound radius must be nonnegative and not NaN.")
        if target_expr.ndim != ref_expr.ndim:
            raise ValueError("Target and reference expressions must have the same dimension.")
        if np.isposinf(radius):
            return 0.0
        lb, ub = ref_expr.bound_propagation()
        old_lb, old_ub = target_expr.bound_propagation()
        new_lb, new_ub = np.maximum(old_lb, lb - radius), np.minimum(old_ub, ub + radius)
        if np.any(new_lb > new_ub):
            raise ValueError("The reference bound contradicts the target bounds.")
        # Fail unsupported backends before changing any Python or solver bounds.
        self.canonicalizer.add_theory_cut(radius, target_expr, lb, ub)
        if target_expr.is_leaf:
            target_expr.update_bounds(new_lb, new_ub)
            self.canonicalizer.update_bounds(target_expr, target_expr.lb, target_expr.ub)
            if self.use_obbt_handler:
                self.obbt_handler.update_bounds(target_expr, target_expr.lb, target_expr.ub)
        if self.use_obbt_handler:
            cuts = [target_expr >= lb - radius, target_expr <= ub + radius]
            self.obbt_handler.add_constraints([c for c in cuts if is_binary_free(c)])
        return float(
            (np.count_nonzero(new_lb > old_lb) + np.count_nonzero(new_ub < old_ub))
            / (2 * target_expr.ndim)
        )

    def _set_zero_objective(self) -> None:
        """Feasibility problem: a constant zero objective."""
        self.objective = None
        self._residual = None
        self.canonicalizer.set_zero_objective()

    def set_zero_objective(self):
        """Deprecated: use ``vp.maximize(0)``."""
        _deprecated("set_zero_objective", "vp.maximize(0)")
        self._set_zero_objective()

    def _set_norm_objective(
        self, expr_list: Sequence[LinearExpression] | LinearExpression, ord: int | float
    ) -> None:
        """Maximize ``||expr_list||_ord`` (``ord=2`` is the squared two-norm)."""
        self._residual = None
        try:
            len(expr_list)
        except TypeError:
            expr_list = [expr_list]
        if ord == 1:
            self.canonicalizer.set_one_norm_objective(expr_list)
        else:
            lb, ub = zip(*(expr.bound_propagation() for expr in expr_list))
            setter = (
                self.canonicalizer.set_two_norm_objective
                if ord == 2
                else self.canonicalizer.set_infinity_norm_objective
            )
            setter(expr_list, list(lb), list(ub))
        self.objective = (tuple(expr_list), ord)

    def set_infinity_norm_objective(self, expr_list):
        """Deprecated: use ``vp.maximize(ft.norm(expr, np.inf))``."""
        _deprecated("set_infinity_norm_objective", "vp.maximize(ft.norm(expr, np.inf))")
        self._set_norm_objective(expr_list, np.inf)

    def set_two_norm_objective(self, expr_list):
        """Deprecated: use ``vp.maximize(ft.sum_squares(expr))``."""
        _deprecated("set_two_norm_objective", "vp.maximize(ft.sum_squares(expr))")
        self._set_norm_objective(expr_list, 2)

    def set_one_norm_objective(self, expr_list):
        """Deprecated: use ``vp.maximize(ft.norm(expr, 1))``."""
        _deprecated("set_one_norm_objective", "vp.maximize(ft.norm(expr, 1))")
        self._set_norm_objective(expr_list, 1)

    def set_objective_norm(self, expr, ord) -> None:
        """
        Deprecated: use ``vp.maximize(ft.norm(expr, ord))``, or
        ``vp.maximize(ft.sum_squares(expr))`` for ``ord=2`` (the squared two-norm).
        """
        _deprecated("set_objective_norm", "vp.maximize(ft.norm(expr, ord)) or ft.sum_squares")
        if ord == "inf":
            ord = np.inf
        if ord not in (1, 2, np.inf):
            raise ValueError(f"Unsupported norm order: {ord}. Use 1, 2, np.inf, or 'inf'.")
        self._set_norm_objective(expr, ord)

    def maximize(self, objective: Norm | SumSquares | int) -> None:
        """
        Set the objective to maximize, the worst case that ``solve`` bounds.

        Parameters
        ----------
        objective : Norm, SumSquares, or 0
            ``ft.norm(expr, 1)``, ``ft.norm(expr, np.inf)``, ``0`` for a
            feasibility problem, or
            ``ft.sum_squares(expr)`` for the squared two-norm. With
            ``method="mip"``, the squared two-norm is an MIQP and needs
            ``solver="gurobi"``; ``solve`` then returns a bound on the squared norm.

        Examples
        --------
        >>> vp.maximize(ft.norm(z - z_prev, np.inf))
        """
        if isinstance(objective, Norm):
            self._set_norm_objective(list(objective.exprs), objective.p)
            if objective.p == np.inf and len(objective.exprs) == 1:
                self._residual = objective.exprs[0]._difference
        elif isinstance(objective, SumSquares):
            self._set_norm_objective(objective.expr, 2)
        elif (
            isinstance(objective, (int, float))
            and not isinstance(objective, bool)
            and objective == 0
        ):
            self._set_zero_objective()
        else:
            raise TypeError(
                "maximize takes ft.norm(expr, p), ft.sum_squares(expr), or 0, "
                f"got {type(objective).__name__}."
            )

    def solve(
        self,
        convex_hull_cuts: bool = False,
        cut_rounds: int = 1,
        tighten_objective_bounds: bool | None = None,
        warm_start: bool = True,
        verbose: bool | None = None,
        **kwargs,
    ) -> float:
        """
        Solve the optimization problem.

        This method delegates to the canonicalizer to solve the
        optimization problem using the recorded thresholding or
        activation steps.

        Parameters
        ----------
        convex_hull_cuts : bool, default False
            MIP only. Before solving, separate the root LP optimum from the convex
            hulls of the ReLU, saturated linear and soft-threshold steps and add
            the most violated hull inequality of every unstable component. The
            cuts stay in the model, so later solves (e.g. at the next ``K``) keep
            them and separate on top of them. With OBBT enabled, each round of
            cuts is also added to the OBBT LP, followed by one OBBT pass over the
            leaves the cuts touch.
        cut_rounds : int, default 1
            Maximum number of separation rounds per solve; each round re-solves
            the relaxation with the previous cuts. Stops early once no cut is
            violated. Small values (1 to 3) are recommended.
        tighten_objective_bounds : bool or None, default None
            MIP with an infinity-norm objective and OBBT only. Before solving
            (and after any cut rounds), replace the interval-propagation bounds
            of the residual, from which the objective's big-M constants are
            built, by its range over the OBBT LP, padded slightly outward. The
            optimum is unchanged; the relaxation gets tighter. Statistics are
            stored on ``objective_bound_stats``. ``None`` tightens whenever an
            OBBT solver is attached and skips silently otherwise; ``True`` also
            warns when it cannot tighten; ``False`` keeps the interval bounds.
        warm_start : bool, default True
            MIP only. Give the solver up to two starts: the worst case of the
            previous solve (normally ``K - 1``) run through the current steps,
            and the worst of the samples given to ``set_start_samples``.
            Solvers without MIP starts ignore them. Statistics are stored on
            ``canonicalizer.warm_start_stats``.
        verbose : bool or None, default None
            Print the log of this solve, including the cut separation and OBBT LPs
            it runs. ``None`` keeps the ``verbose`` setting of the constructor. The
            OBBT LPs that run while the steps are added follow the constructor.
        **kwargs
            Keyword arguments forwarded to the canonicalizer's
            ``solve_model``. Any keyword is passed **verbatim** to the active
            backend solver as a native option (e.g. ``TimeLimit=10`` or
            ``MIPGap=1e-4`` for Gurobi, ``max_iter=500`` for
            Clarabel/cvxpy). These solve-time options override matching
            entries in the construction-time ``params`` dict. A
            ``ValueError`` is raised if the backend rejects an option
            name or value.

        Returns
        -------
        float
            MIP: a finite best upper bound, including after a time limit;
            raises ``RuntimeError`` if no usable bound is available. SDP: the
            dual objective after an accurate optimal termination; inaccurate
            or early termination raises ``RuntimeError``. Bounds are subject
            to solver numerical tolerances, not exact-arithmetic certificates.
            Simulator: an array of sampled residuals, with summary statistics
            available through ``vp["max"]`` and related keys.
        """
        if verbose is not None and self.method == "mip":
            # Cut separation and objective-bound OBBT also run in this solve.
            with self._mip_verbosity(verbose):
                return self.solve(
                    convex_hull_cuts=convex_hull_cuts,
                    cut_rounds=cut_rounds,
                    tighten_objective_bounds=tighten_objective_bounds,
                    warm_start=warm_start,
                    **{**self._verbosity_option(self.method, self._solver_name, verbose), **kwargs},
                )
        if verbose is not None:
            kwargs = {
                **self._verbosity_option(self.method, self._solver_name, verbose, explicit=True),
                **kwargs,
            }
        # ``cut_stats`` has one entry per round that added cuts; ``separation_stats``
        # one per round that ran, including a final round that found nothing.
        self.cut_stats = []
        self.separation_stats = []
        if convex_hull_cuts and self.method == "mip":
            for _ in range(cut_rounds):
                start = time.perf_counter()
                cuts = self.canonicalizer.separate_hull_cuts()
                separate_time = time.perf_counter() - start
                self.separation_stats.append(dict(self.canonicalizer.last_separation_stats))
                if not cuts:
                    break
                start = time.perf_counter()
                self._add_cuts(cuts)
                self.cut_stats.append(
                    dict(
                        cuts=len(cuts),
                        separate_time=separate_time,
                        obbt_time=time.perf_counter() - start,
                        **self.canonicalizer.last_separation_stats,
                    )
                )
        self.objective_bound_stats = None
        # None (the default) tightens whenever an OBBT solver is attached and skips
        # silently otherwise; True also warns when it cannot tighten; False disables.
        if tighten_objective_bounds is not False and self.method == "mip":
            if not self.use_obbt_handler:
                if tighten_objective_bounds:
                    warnings.warn(
                        "tighten_objective_bounds needs an OBBT solver (obbt_solver=...); "
                        "keeping the interval bounds."
                    )
            else:
                self.objective_bound_stats = self.canonicalizer.tighten_objective_bounds()
        if self.method == "mip":
            bound = self.canonicalizer.solve_model(warm_start=warm_start, **kwargs)
            self._chain_residual(bound)
            return bound
        if self.method == "simulator" and self._unsampled_params:
            dims = ", ".join(str(var.ndim) for var in self._unsampled_params)
            raise ValueError(
                f"{len(self._unsampled_params)} parameter(s) (ndim {dims}) have no samples. "
                "The simulator does not sample from bounds: pass "
                "parameter(ndim, samples=<array of shape (num_samples, ndim)>), drawn "
                "from the distribution the probabilistic bound should hold for "
                "(equal finite bounds fix a constant and need no samples)."
            )
        return self.canonicalizer.solve_model(**kwargs)

    def _chain_residual(self, bound: float) -> None:
        """
        Tighten ``a`` after a solve certified ``||a - b||_inf <= bound``.

        With ``||b - root||_inf <= r_b`` from earlier solves, the triangle inequality
        gives ``||a - root||_inf <= r_b + bound``. The bound holds over the current
        family; constraints added later only shrink it, so it stays valid.
        """
        if not self.tighten_residuals or self._residual is None:
            return
        a, b = self._residual
        root, radius_b = self._residual_chain.get(b.counter, (b, 0.0))
        # Pad outward a little for solver tolerances.
        radius = (radius_b + bound) * (1 + 1e-9) + 1e-9
        known = self._residual_chain.get(a.counter)
        if known is not None and (known[0] is not root or known[1] <= radius):
            return
        self._residual_chain[a.counter] = (root, radius)
        self._apply_reference_bound(a, root, radius)

    @contextmanager
    def _mip_verbosity(self, verbose: bool):
        """Set the log of the main and OBBT MIP solvers to ``verbose``, then restore it."""
        owners = [self.canonicalizer] + ([self.obbt_handler] if self.use_obbt_handler else [])
        for owner in owners:
            owner.solver.set_param(owner.solver.VERBOSITY_KEY, verbose)
        try:
            yield
        finally:
            for owner in owners:
                key = owner.solver.VERBOSITY_KEY
                flags = [
                    v for k, v in (owner.solver_params or {}).items() if k.lower() == key.lower()
                ]
                owner.solver.set_param(key, bool(flags[-1]) if flags else False)

    def set_start_samples(self, samples: dict) -> None:
        """
        Samples of parameters (and free initial iterates) for the sample-max MIP start.

        At every MIP solve the worst of these samples, run through the current
        steps, is given to the solver as a start next to the sequential one.

        Parameters
        ----------
        samples : dict[LinearExpression, np.ndarray]
            ``(n, ndim)`` samples per parameter or initial iterate, with the same
            ``n``. Inputs fixed by their bounds need no samples.
        """
        if self.method == "mip":
            self.canonicalizer.set_start_samples(samples)

    def post_process(self, target_expr, ref_expr, residuals):
        """Tighten a target using certified successive infinity-norm residual bounds.

        The caller must supply nonnegative upper bounds on every successive
        iterate difference along the path from ``ref_expr`` to ``target_expr``,
        uniformly over the same parameter/initial-state family. The triangle
        inequality then bounds their distance by ``sum(residuals)``. Sample
        maxima, incumbents, squared norms, and a bound for only the final step
        do not justify this operation. An infinite bound leaves the model unchanged.

        Returns the fraction of target bound entries tightened. Updates leaf
        bounds in Python, the main solver, and OBBT; composite targets receive
        the corresponding constraints without changing their constituent leaves.
        """
        residuals = np.asarray(residuals, dtype=float)
        if (
            residuals.ndim != 1
            or residuals.size == 0
            or np.any(np.isnan(residuals))
            or np.any(residuals < 0)
        ):
            raise ValueError("Residual bounds must be a nonempty vector of nonnegative values.")
        return self._apply_reference_bound(target_expr, ref_expr, np.sum(residuals))

    def neural_network(self, expr: LinearExpression, layers: list) -> LinearExpression:
        """
        Thread ``expr`` through a feedforward ReLU network inside this Verifier.

        The network is unrolled into alternating affine and ReLU VP steps.
        Affine layers are applied analytically (no new solver variable); ReLU
        layers call ``run_step`` so the solver gets the appropriate constraints.

        Parameters
        ----------
        expr : LinearExpression
            Network input (e.g. a bounded iterate or parameter).
        layers : list
            Ordered sequence of ``LinearLayer`` / ``ReLULayer`` objects.
        Returns
        -------
        LinearExpression
            Network output, expressed in terms of the original leaf variables.

        """

        for i, layer in enumerate(layers):
            if isinstance(layer, LinearLayer):
                W, b = layer.weight, layer.bias

                if W.shape[1] != expr.ndim:
                    raise ValueError(
                        f"Layer {i}: weight shape {W.shape} is incompatible with "
                        f"input dimension {expr.ndim}."
                    )

                expr = self.run_step(expr, Affine, weight=W, bias=b)

            elif isinstance(layer, ReLULayer):
                expr = self.run_step(expr, Relu)

            else:
                raise ValueError(f"Layer {i}: unsupported layer type {type(layer).__name__}.")

        return expr

    def codegen(
        self,
        out_dir,
        iterates=None,
        config=None,
        driver=None,
        state=None,
        params=None,
        name="fom",
        code_optimization=False,
        profile=False,
        roll=True,
        param_values=None,
    ):
        """Generate a low-level solver for the run recorded on this verifier.

        Takes the iterate sequence your loop produced -- the same program you
        verified with, no restructuring:

        >>> z = verifier.iterate_trajectory(K + 1)
        >>> z[0] = verifier.iterate(n, lb=z0, ub=z0)
        >>> for k in range(1, K + 1):
        ...     z[k] = verifier.soft_threshold(...)
        >>> verifier.solve()
        >>> verifier.codegen("out/", iterates=z)

        Arbitrary per-iteration coefficients, a step type that changes with the
        iteration, and methods that hold a reference to an earlier iterate (a
        Halpern anchor) all work, because each iteration carries its own
        coefficient blocks. When repeating one iteration is *verified* to
        reproduce the recording, the compact rolled solver is emitted instead, so
        writing a stationary method as a loop costs nothing; pass ``roll=False``
        to unroll regardless.

        The result is checked against the recorded tape before anything is
        emitted, drawing its own parameter values, so no extra arguments are
        needed to gate it.

        Thin delegate to `fomtools.codegen.generate_code`; see that function
        for the remaining parameter semantics. To generate a rolled solver from a
        single stationary iteration instead, use `codegen_rolled`.

        Returns
        -------
        CodegenResult
            The IR, output directory, target, and agent results.
            ``result.ir.unrolled`` says whether the run was rolled back up.
        """
        from fomtools.codegen import generate_code

        return generate_code(
            self,
            None,
            out_dir,
            config=config,
            driver=driver,
            state=state,
            params=params,
            name=name,
            code_optimization=code_optimization,
            profile=profile,
            iterates=iterates,
            param_values=param_values,
            roll=roll,
        )

    def codegen_rolled(
        self,
        out_dir,
        next_state,
        config=None,
        driver=None,
        state=None,
        params=None,
        name="fom",
        code_optimization=False,
        profile=False,
        iterates=None,
        param_values=None,
    ):
        """Generate a rolled solver from one stationary iteration.

        Thin delegate to `fomtools.codegen.generate_code`. See that
        function for parameter semantics. ``next_state`` is the iterate
        ``LinearExpression`` produced after exactly one iteration, and the
        generated loop repeats it ``config.termination.max_iters`` times -- so
        unlike `codegen`, the horizon is yours to choose rather than the
        one you recorded.

        Pass ``iterates`` (the successive recorded iterate expressions) and
        ``param_values`` to have the extracted iteration checked against the
        recorded tape before anything is emitted. Without them nothing verifies
        that the method is stationary, so a coefficient that varies with the
        iteration index would be frozen at its recorded value. `codegen`
        needs no such argument, since it generates the recording itself.

        A method that *does* vary with the iteration index can go through
        `fomtools.codegen.generate_code` with ``schedules=``, which
        tabulates the declared coefficients but needs the recording as a build
        function; or through `codegen`, which takes the finished tape.

        Returns
        -------
        CodegenResult
            The IR, output directory, target, and agent results.
        """
        from fomtools.codegen import generate_code

        return generate_code(
            self,
            next_state,
            out_dir,
            config=config,
            driver=driver,
            state=state,
            params=params,
            name=name,
            code_optimization=code_optimization,
            profile=profile,
            iterates=iterates,
            param_values=param_values,
        )

    def _solution_value(self, expr: LinearExpression) -> np.ndarray:
        """Value of ``expr`` at the incumbent of the last solve (see `LinearExpression.value`)."""
        if not hasattr(self.canonicalizer, "get_solution_value"):
            raise NotImplementedError(
                f"Solution values are not available with method={self.method!r}."
            )
        return self.canonicalizer.get_solution_value(expr)

    def get_solution_value(self, iterate: LinearExpression) -> np.ndarray:
        """Deprecated: use ``expr.value``."""
        _deprecated("get_solution_value", "expr.value")
        return self._solution_value(iterate)


class _StepProxy:
    """
    Internal helper used by the `Verifier.step` API.

    This proxy enables the syntax ``verifier.step.<step_name>(arg=..., **kwargs)``
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

        def wrapper(arg, relax_binary_vars=False, **kwargs):
            """Execute the specified thresholding or activation step."""
            _deprecated(f"vp.step.{name}", f"vp.{name}")
            return self._verifier.run_step(arg, step_cls, relax_binary_vars, **kwargs)

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
