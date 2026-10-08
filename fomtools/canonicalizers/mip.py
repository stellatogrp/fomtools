from typing import List

import numpy as np

from fomtools import LinearExpression
from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import ComplementarityConstraint, Constraint, QuadraticConstraint
from fomtools.steps import Affine, Relu, SaturatedLinear, SoftThreshold
from fomtools.utils.envelopes import softthreshold_envelope


def _require_finite_bounds(kind: str, i: int, lb: float, ub: float) -> None:
    """Raise if a step input that needs binaries has an infinite bound (big-M is undefined)."""
    if not (np.isfinite(lb) and np.isfinite(ub)):
        raise ValueError(
            f"{kind} input {i} has bounds [{lb}, {ub}]: the exact MIP encoding needs finite "
            f"bounds on every {kind} input that crosses a breakpoint. Bound the iterates or "
            "parameters that feed it (for example with add_iterate/add_param bounds or OBBT)."
        )


class MIP(Canonicalizer):
    """
    An interface for mixed-integer programming (MIP) methods.

    Attributes
    ----------
    solver: Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).
    _objective_spec: tuple or None
        Stored objective specification for deferred application.
        Applied to a copy of the solver at solve time.
    _solved_copy: Solver or None
        Reference to the solved copy of the solver, used for
        extracting solution values after solve.

    """

    def __init__(self, solver_params=None, solver="gurobi"):
        self.steps = []
        super(MIP, self).__init__(solver_params, solver)
        self._objective_spec = None
        self._solved_copy = None

    def add_var(self, var: LinearExpression, **kwargs) -> None:
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
        if isinstance(constraint, ComplementarityConstraint):
            self.add_constraints(self._build_complementarity_constraints(constraint))
            return
        if isinstance(constraint, QuadraticConstraint):
            # Quadratic (sum-of-squares) constraints require a solver that supports
            # quadratic constraints (MIQCP). Gurobi does; HiGHS does not. Each solver
            # validates this in its own ``add_sum_squares_constr`` (HiGHS raises), so the
            # error still surfaces eagerly at add-time rather than at solve-time.
            self.solver.add_sum_squares_constr(
                constraint.expr, constraint.radius_sq, constraint.sense
            )
            return
        self.solver.add_constr(constraint)

    def _build_complementarity_constraints(
        self, constraint: ComplementarityConstraint
    ) -> List[Constraint]:
        r"""
        Build the big-M MIP formulation of an elementwise complementarity constraint.

        For ``e1 ⊥ e2`` (both ``>= 0``), each coordinate ``i`` gets a fresh binary
        ``z`` and, with finite big-M values ``M1 = e1.ub[i]`` and ``M2 = e2.ub[i]``:

        $$
            e_{1,i} \ge 0,\quad e_{2,i} \ge 0,\quad
            e_{1,i} \le M_{1}\,z,\quad e_{2,i} \le M_{2}\,(1 - z).
        $$

        ``z = 1`` forces ``e2[i] = 0``; ``z = 0`` forces ``e1[i] = 0``. This mirrors the
        ReLU big-M idiom in :meth:`_build_relu_constraints`.

        Parameters
        ----------
        constraint : ComplementarityConstraint
            The complementarity constraint to encode.

        Returns
        -------
        List[Constraint]
            The linear (and binary-coupled) constraints encoding the relation.

        Raises
        ------
        ValueError
            If either expression has a non-finite upper bound, so big-M is undefined.
        """
        e1, e2 = constraint.expr1, constraint.expr2
        # Use bound_propagation so the big-M is correct for composite expressions
        # (e.g. ``b - A @ u``), not just leaf variables: a composite's ``.ub`` is the
        # list of its leaves' bounds, whereas bound_propagation returns the elementwise
        # upper bound of the expression itself.
        with np.errstate(invalid="ignore"):  # inf - inf when an input is unbounded
            _, M1 = e1.bound_propagation()
            _, M2 = e2.bound_propagation()
        if not (np.all(np.isfinite(M1)) and np.all(np.isfinite(M2))):
            raise ValueError(
                "Complementarity big-M encoding requires finite upper bounds on both "
                "expressions. Provide explicit variable bounds or enable OBBT."
            )

        constraints = []
        for i in range(constraint.ndim):
            _e1 = e1[i]
            _e2 = e2[i]
            z = LinearExpression.from_int(_e1.ndim, is_binary=True)  # scalar binary
            constraints.extend(
                [
                    _e1 >= 0,
                    _e2 >= 0,
                    _e1 <= M1[i] * z,
                    _e2 <= M2[i] * (1 - z),
                ]
            )
        return constraints

    def add_constraints(self, constraints: List[Constraint]) -> None:
        """
        Add a list of constraints to the MIP instance. This method registers
        multiple constraints with the MIP problem and attaches them to
        the internal model.

        Parameters
        ----------
        constraints: List[Constraint]
            List of constraints to add to the MIP instance.

        Returns
        -------
        None
        """
        self.solver.constraints_to_solver(constraints)

    def add_theory_cut(
        self,
        margin: float,
        target_expr: LinearExpression,
        lb: float | np.ndarray,
        ub: float | np.ndarray,
    ):
        """
        Add relaxed bound constraints on the target expression up to the
        provided relaxation margin.

        Parameters
        ----------
        margin: float
            Relaxation margin for the bounds. If finite, the lower bound is
            shifted to ``lb - C`` and the upper bound to ``ub + C``. If
            ``np.inf``, no constraints are added

        target_expr: LinearExpression
            The expression whose bounds are to be relaxed.

        lb: float | np.ndarray
            Lower bound(s) for ``target_expr`` before relaxation.

        ub: float | np.ndarray
            Upper bound(s) for ``target_expr`` before relaxation.

        Returns
        -------
        None
        """
        if margin != np.inf:
            self.solver.constraints_to_solver(
                [target_expr >= lb - margin, target_expr <= ub + margin]
            )

    def update_bounds(
        self, expression: LinearExpression, lb: float | np.ndarray, ub: float | np.ndarray
    ):
        """
        Update the lower and upper bounds of the solver variables corresponding
        to the given ``LinearExpression``.

        Parameters
        ----------
        expression: LinearExpression
            A ``LinearExpression`` object whose underlying variables
            bound are to be updated.

        lb: float or np.ndarray
            New lower bound(s).

        ub: float or np.ndarray
            New upper bound(s).


        Returns
        -------
        None
        """
        self.solver.update_bounds(expression, lb, ub)

    def copy(self) -> "MIP":
        """
        Create an independent copy of this MIP canonicalizer.

        Returns a new MIP canonicalizer with a copied solver. The copy
        can be modified without affecting the original.

        Returns
        -------
        MIP
            A new canonicalizer instance with copied state.
        """
        new_canon = MIP.__new__(MIP)
        new_canon.solver_params = dict(self.solver_params) if self.solver_params else None
        new_canon.solver_name = self.solver_name
        new_canon.solver = self.solver.copy()
        new_canon.obbt_handler = self.obbt_handler
        new_canon._objective_spec = self._objective_spec
        new_canon._solved_copy = None
        return new_canon

    def set_zero_objective(self):
        """Store zero objective specification for deferred application."""
        self._objective_spec = ("zero",)

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
        """Store infinity-norm objective specification for deferred application."""
        self._objective_spec = ("infinity_norm", expr_list, lb_list, ub_list)

    def set_two_norm_objective(self, expr_list, lb_list, ub_list):
        """Store two-norm objective specification for deferred application."""
        self._objective_spec = ("two_norm", expr_list, lb_list, ub_list)

    def set_one_norm_objective(self, expr_list):
        """Store one-norm objective specification for deferred application."""
        self._objective_spec = ("one_norm", expr_list)

    def _apply_objective(self, solver):
        """
        Apply stored objective to the given solver.

        Parameters
        ----------
        solver : Solver
            The solver instance to apply the objective to.
        """
        if self._objective_spec is None:
            return

        obj_type = self._objective_spec[0]
        match obj_type:
            case "infinity_norm":
                _, expr_list, lb_list, ub_list = self._objective_spec
                self._apply_infinity_norm_objective(solver, expr_list, lb_list, ub_list)
            case "two_norm":
                _, expr_list, lb_list, ub_list = self._objective_spec
                self._apply_two_norm_objective(solver, expr_list, lb_list, ub_list)
            case "one_norm":
                _, expr_list = self._objective_spec
                self._apply_one_norm_objective(solver, expr_list)
            case "zero":
                solver.set_objective(expression=0)

    def _apply_infinity_norm_objective(self, solver, expr_list, lb_list, ub_list):
        """
        Build and apply infinity-norm objective constraints to solver.

        Parameters
        ----------
        solver : Solver
            The solver instance to apply the objective to.
        expr_list : list[LinearExpression]
            List of expressions for the infinity norm.
        lb_list : list[np.ndarray]
            Lower bounds for each expression.
        ub_list : list[np.ndarray]
            Upper bounds for each expression.
        """
        constraints = []
        M = np.max(np.abs(np.hstack([np.hstack(lb_list), np.hstack(ub_list)])))
        q = LinearExpression.from_int(ndim=1, ub=M)
        all_gammas = []

        for expr, lb, ub in zip(expr_list, lb_list, ub_list):
            n = expr.ndim

            up = LinearExpression.from_int(ndim=n, ub=np.abs(ub))
            un = LinearExpression.from_int(ndim=n, ub=np.abs(lb))
            constraints.append(up - un == expr)

            all_gammas.append(gamma := LinearExpression.from_int(ndim=n, is_binary=True))
            for _lb, _ub, _up, _un, _gamma in zip(lb, ub, up, un, gamma):
                if _lb >= 0:
                    constraints.append(_un == 0)
                elif _ub < 0:
                    constraints.append(_up == 0)
                else:
                    vobji = LinearExpression.from_int(ndim=1, is_binary=True)
                    constraints.extend(
                        [_up <= np.abs(_ub) * vobji, _un <= np.abs(_lb) * (1 - vobji)]
                    )

                constraints.extend([q >= _up + _un, q <= _up + _un + M * (1 - _gamma)])

        # need 1 gamma across all vectors in the list
        constraints.append(sum([sum(gamma) for gamma in all_gammas]) == 1)

        solver.constraints_to_solver(constraints)
        solver.set_objective(expression=q, sense="maximize")

    def _apply_two_norm_objective(self, solver, expr_list, lb_list, ub_list):
        """
        Build and apply two-norm objective to solver.

        The two-norm objective requires a quadratic program (MIQP). This is
        supported by Gurobi but not by HiGHS.

        Parameters
        ----------
        solver : Solver
            The solver instance to apply the objective to.
        expr_list : list[LinearExpression]
            List of expressions for the two-norm objective.
        lb_list : list[np.ndarray]
            Lower bounds for each expression.
        ub_list : list[np.ndarray]
            Upper bounds for each expression.

        Raises
        ------
        ValueError
            If the solver does not support quadratic objectives (e.g., HiGHS).
        """
        if self.solver_name not in ("gurobi", "grb"):
            raise ValueError(
                "Two-norm objective requires a solver that supports quadratic "
                "programming (MIQP). HiGHS does not support MIQP. Use Gurobi or "
                "switch to infinity-norm/one-norm objective."
            )

        solver.set_two_norm_objective(expr_list, lb_list, ub_list)

    def _apply_one_norm_objective(self, solver, expr_list):
        """
        Build and apply one-norm objective to solver.

        Uses the identity: ||x||_1 = 1^T @ (relu(x) + relu(-x))

        Parameters
        ----------
        solver : Solver
            The solver instance to apply the objective to.
        expr_list : list[LinearExpression]
            List of expressions for the one-norm objective.
        """
        constraints = []
        q = 0

        for expr in expr_list:
            n = expr.ndim

            # Create relu(expr) step
            step_pos = Relu(arg=expr)
            step_pos.update_bounds(*self.get_expression_bounds(expr))
            for leaf in step_pos.output_leaves:
                solver.add_var(leaf)
            constraints.extend(self._build_relu_constraints(step_pos))

            # Create relu(-expr) step
            neg_expr = -expr
            step_neg = Relu(arg=neg_expr)
            step_neg.update_bounds(*self.get_expression_bounds(neg_expr))
            for leaf in step_neg.output_leaves:
                solver.add_var(leaf)
            constraints.extend(self._build_relu_constraints(step_neg))

            # Accumulate: 1^T @ (relu(expr) + relu(-expr))
            ones = np.ones((1, n))
            q = q + ones @ step_pos.output + ones @ step_neg.output

        solver.constraints_to_solver(constraints)
        solver.set_objective(expression=q, sense="maximize")

    def separate_hull_cuts(self, tol: float = 1e-6) -> List[Constraint]:
        """
        Separate the root LP optimum from the convex hulls of the steps.

        Copies the model (which carries every cut added so far), applies the
        objective, solves the continuous relaxation and returns the most violated
        hull inequality of every unstable step component. The model itself is not
        modified; ``Verifier.solve`` adds the returned cuts.

        Parameters
        ----------
        tol: float, default 1e-6
            Minimum violation for a cut to be returned.

        Returns
        -------
        List[Constraint]
            The separated cuts.
        """
        relaxed = self.solver.copy()
        if self.solver_params:
            relaxed.set_solver_params(self.solver_params)
        self._apply_objective(relaxed)
        return relaxed.separate_convex_hull_cuts(self.steps, tol=tol)

    def solve_model(self, **solver_kwargs) -> float:
        """
        Optimize the underlying MIP problem.

        Creates a copy of the solver, applies the objective to the copy,
        solves it, and keeps a reference for solution extraction.
        This prevents objective constraints from accumulating across
        multiple solve calls. Convex hull cuts are added to the model
        beforehand by ``Verifier.solve``.

        Parameters
        ----------
        **solver_kwargs
            Native solver options forwarded verbatim to the backend
            (e.g. ``TimeLimit=10``, ``MIPGap=1e-4`` for Gurobi). Applied
            to the solve copy, so they override construction-time
            ``params`` for this call. A ``ValueError`` is raised if the
            backend rejects an option.

        Returns
        -------
        float
            The objective value for the current solution.

        """
        # Create copy to apply objective without modifying original
        self._solved_copy = self.solver.copy()

        # Re-apply construction-time options: HiGHS's copy() uses passModel,
        # which copies the model data but not solver options, so the copy
        # would otherwise reset them to defaults. (Gurobi's copy() preserves
        # them, making this a harmless no-op there.) solver_kwargs are applied
        # last so solve-time options override construction-time params.
        if self.solver_params:
            self._solved_copy.set_solver_params(self.solver_params)

        if solver_kwargs:
            self._solved_copy.set_solver_params(solver_kwargs)

        self._apply_objective(self._solved_copy)
        self._solved_copy.optimize()
        return self._solved_copy["objVal"]

    def get_solution_value(self, expression: LinearExpression) -> np.ndarray:
        """
        Return the solution value for the given ``LinearExpression``.

        Parameters
        ----------
        expression: LinearExpression
            The expression whose solution value is to be computed. Each element in
            its decomposition must have been registered with the solver.

        Returns
        -------
        np.ndarray[float]
            A 1-D array of shape ``(expression.ndim,)`` containing the
            evaluated values at the current model solution.

        Raises
        ------
        RuntimeError
            If no solution is available (solve_model() not called).
        """
        if self._solved_copy is None:
            raise RuntimeError("No solution available. Call solve_model() first.")
        return self._solved_copy.get_solution_value(expression)

    def canonicalize_affine(self, step: Affine) -> None:
        """
        Construct the linear equality constraint corresponding to the
        affine iteration step. The constraint is used to express the affine
        update step in the verification problem

        Parameters
        ----------
        step: Affine
            An instance of the ``Affine`` class to canonicalize to a MIP problem.

        Returns
        -------
        None
        """
        self.steps.append(step)
        self.add_constraint(step.output_mat @ step.output == step.arg)

    def canonicalize_relu(self, step: Relu):
        r"""
        Construct the convex-hull linear constraints corresponding
        to the `Rectified Linear Unit (ReLU)` operator.

        Given a ReLU step: $(\phi(v,x))_i = \mathbf{ReLU}(a^T y) = \max\{a^T y, 0\}.$

        This method derives the convex hull of the region defined by:

        $$
            \phi = \left\{ (y, w) \in \left [\underline{y}, \overline{y} \right ] \times
            \mathbb{R} | w = \textbf{ReLU}(a^{T}y) \right\},
        $$

        and expresses it as a set of Constraint objects. The convex hull
        of the region is:


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
            An instance of the `Relu` step to canonicalize.

        Returns
        -------
        None
        """
        constraints = self._build_relu_constraints(step)
        self.add_constraints(constraints)
        self.steps.append(step)

    def _build_relu_constraints(self, step: Relu) -> List[Constraint]:
        """
        Build MIP formulation constraints for a ReLU step.

        Parameters
        ----------
        step : Relu
            An instance of the `Relu` step.

        Returns
        -------
        List[Constraint]
            MIP constraints for the ReLU activation.
        """
        constraints = []
        # Iterate proj indices only. For partial projection, ``step.output`` is a
        # composite whose non-proj rows share decomposition with ``step.arg``;
        # those rows already satisfy ``output[i] == arg[i]`` structurally, so no
        # explicit constraint is needed. For full projection, ``proj_indices``
        # covers every row and behavior matches the previous implementation.
        for i in sorted(step.proj_indices):
            _output = step.output[i]
            _arg = step.arg[i]
            if step.ub[i] <= 0:
                constraints.append(_output == 0)
            elif step.lb[i] > 0:
                constraints.append(_output == _arg)
            else:
                _require_finite_bounds("ReLU", i, step.lb[i], step.ub[i])
                new_var = LinearExpression.from_int(_arg.ndim, is_binary=True)
                step.idx_with_binary_vars.add(i)
                constraints.append(_output <= step.ub[i] * new_var)
                constraints.extend(
                    [
                        _output <= step.ub[i] / (step.ub[i] - step.lb[i]) * (_arg - step.lb[i]),
                        _output >= _arg,
                        _output <= _arg - step.lb[i] * (1 - new_var),
                    ]
                )
        return constraints

    def canonicalize_saturatedlinear(self, step: SaturatedLinear):
        r"""
        Construct the convex-hull linear constraints corresponding to
        the saturated linear unit operator.

        Given a saturated linear unit operator, defined as:

        $$
            (\phi(v, b))_i = S_{[b, c]}(a^T y) =
            \begin{cases}
            b & a^T y < b, \\\\
            a^T y & b \leq a^T y \leq c, \\\\
            c & a^T y > c,
            \end{cases}
            \quad \text{with } y = (v, b), \text{and }
             b, c \in \mathbb{R},
        $$

        This method derives the convex hull of the region defined by:

        $$
            \Phi = \{ (y, w) \in [\underline{y}, \bar{y}] \times [b, c] \mid w =
            \mathcal{S}_{[b,c]}(a^\top y) \}.
        $$

        and expresses it as a set of Constraint objects. The convex hull of the
        region is:

        $$
            \operatorname{conv}(\Phi) =
            \left\{
            (y, w) \in [\underline{y}, \bar{y}] \times [b, c]
            \;\middle|\;
            \begin{cases}
            w = b & u_J < b, \\[4pt]
            w = c & \ell_J > c, \\[4pt]
            w = a^\top y & b \le \ell_J \le u_J \le c, \\[4pt]
            (y, w) \in Q & \text{otherwise}
            \end{cases}
            \right\},
        $$

        with

        $$
            Q =
            \left\{
            (y, w) \in \mathbb{R}^{p + d + 1}
            \;\middle|\;
            \begin{aligned}
            & b \le w \le c, \\[4pt]
            & \frac{b - c}{b - u_J}(a^\top y - u_J) + c
            \;\le\;
            w
            \;\le\;
            \frac{c - b}{c - \ell_J}(a^\top y - \ell_J) + b, \\[6pt]
            & w \le \sum_{i \in I} a_i (y_i - \ell^0_i)
            + \frac{\ell_I - b}{u^0_o - \ell^0_o}(y_o - \ell^0_o) + b,
            \qquad \forall (I, o) \in \mathcal{I}^+, \\[6pt]
            & w \ge \sum_{i \in I} a_i (y_i - u^0_i)
            + \frac{u_I - c}{\ell^0_o - u^0_o}(y_o - u^0_o) + c,
            \qquad \forall (I, o) \in \mathcal{I}^-
            \end{aligned}
            \right\},
        $$

        where:

        - vector $l^{0} \in \mathbb{R}^{d}$ with entries $l_i^{0} = \underline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \overline{y}_i$ otherwise,
        - vector $u^{0} \in \mathbb{R}^{d}$ with entries $u_i^{0} = \overline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \underline{y}_i$ otherwise,
        - Given $\textbf{I} \subseteq \textbf{J} = \left\{1, ..., n\right\}$, the lower bound is
        defined as $l_I = \sum_{i \in I}a_il_i^{0} + \sum_{i \notin I}a_iu_i^{0}$ and the
        upper bounds $u_I = \sum_{i \in I}a_iu_i^{0} + \sum_{i \notin I}a_il_i^{0}$,
        - $\mathcal{I}^+ = \{ (I, o) \in 2^J \times J \mid \ell_I \ge b,\; \ell_{I \cup \{o\}}
        < b,\; w_I \neq 0 \},$
        - $\mathcal{I}^- = \{ (I, o) \in 2^J \times J \mid u_I \le c,\; u_{I \cup \{o\}} >
        c,\; w_I \neq 0 \}.$

        Parameters
        ----------
        step: SaturatedLinear
            An instance of the `SaturatedLinear` step to canonicalize.

        Returns
        -------
        None
        """
        constraints = []

        for i, (l, u, lb, ub, output, arg) in enumerate(
            zip(step.l, step.u, step.lb, step.ub, step.output, step.arg)
        ):
            if lb >= u:
                constraints.append(output == u)
            elif ub <= l:
                constraints.append(output == l)
            elif lb >= l and ub <= u:
                constraints.append(output == arg)
            else:
                if not step.relax_binary_vars:
                    _require_finite_bounds("Saturated-linear", i, lb, ub)
                    w1 = LinearExpression.from_int(arg.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(arg.ndim, is_binary=True)

                    if lb < l:
                        constraints.extend(
                            [
                                output <= arg + (ub - lb) * w2,
                                arg >= l + (lb - l) * w2,
                                arg <= l + (ub - l) * (1 - w2),
                            ]
                        )

                        if ub > u:
                            constraints.extend(
                                [
                                    output >= u - (u - l) * (1 - w1),
                                    output <= l + (u - l) * (1 - w2),
                                    output >= l + (lb - l) * w2,
                                    w1 + w2 <= 1,
                                ]
                            )
                        else:
                            # Lower-or-middle (lb < l, ub <= u): the upper face is
                            # never active, so only w2 (lower-saturation vs interior)
                            # matters. Pin the saturated branch: w2=1 => output <= l
                            # (with the output-variable bound output >= l this forces
                            # output = l); w2=0 => output <= ub (interior, slack).
                            # Without this the lower-sat output was unpinned and the
                            # MIP admitted off-graph points (mirror of the
                            # soft-threshold left-or-middle bug).
                            constraints.append(output <= l + (ub - l) * (1 - w2))

                    if ub > u:
                        constraints.extend(
                            [
                                output >= arg - (ub - lb) * w1,
                                arg >= u + (lb - u) * (1 - w1),
                                arg <= u + (ub - u) * w1,
                            ]
                        )
                        if lb >= l:
                            constraints.append(output >= u - (u - lb) * (1 - w1))

                if lb < l and ub > u:
                    constraints.extend(
                        [
                            output <= (u - l) / (u - lb) * (arg - u) + u,
                            output >= (u - l) / (ub - l) * (arg - l) + l,
                        ]
                    )

                # Non-strict, so that the boundary cases lb == l and ub == u still pin
                # the output to arg on the unsaturated side.
                elif lb < l and ub <= u:
                    constraints.extend(
                        [output >= arg, output <= ((ub - l) / (ub - lb)) * (arg - ub) + ub]
                    )

                elif lb >= l and ub > u:
                    constraints.extend(
                        [output <= arg, output >= ((u - lb) / (ub - lb) * (arg - lb) + lb)]
                    )

        self.add_constraints(constraints)
        self.steps.append(step)

    def canonicalize_softthreshold(self, step: SoftThreshold):
        r"""
        Construct the convex-hull linear constraints corresponding to the
        soft-thresholding operator.

        Given a soft-thresholding operator for shrinkage, defined as:

        $$
            (\phi(v,x))_i=\mathcal{T}_{\lambda}(a^\top y)=
            \begin{cases}
            a^\top y-\lambda,& a^\top y>\lambda,\\[6pt]
            0,& |a^\top y|\le \lambda,\\[6pt]
            a^\top y+\lambda,& a^\top y<-\lambda,
            \end{cases}
            \quad \text{with } y=(v,x),
        $$

        where $\mathcal{T}_{\lambda}$ is the soft-thresholding operator with constant $\lambda>0$,
        and $a\in\mathbf{R}^d$.

        This method derives the convex hull of the region defined by:

        $$
            \Phi = \{ (y, w) \in [\underline{y}, \bar{y}] \times [b, c]
            \mid w = \mathcal{S}_{[b,c]}(a^\top y) \}.
        $$

        and expresses it as a set of Constraint objects. The convex hull of the
        region is:

        $$
            \operatorname{conv}(\Phi) =
            \left\{
            (y, w) \in [\underline{y}, \bar{y}] \times \mathbf{R}
            \;\middle|\;
            \begin{cases}
            w = a^\top y - \lambda & \ell_J > \lambda, \\[4pt]
            w = a^\top y  + \lambda & u_J < - \lambda, \\[4pt]
            w = 0 & - \lambda \le \ell_J \le u_J \le \lambda, \\[4pt]
            (y, w) \in Q & \text{otherwise}
            \end{cases}
            \right\},
        $$

        with

        $$
            Q = \left\{
            (y, w) \in \mathbb{R}^{p + d + 1}
            \;\middle|\;
            \begin{aligned}
            & a^\top y - \lambda \le w \le a^\top y + \lambda, \\[4pt]
            & \frac{l_J + \lambda}{l_J - \lambda}(a^\top y - \lambda)
            \;\le\;
            w
            \;\le\;
            \frac{u_J - \lambda}{u_J + \lambda}(a^\top y + \lambda), \\[6pt]
            & w \le \sum_{i \in I} a_i (y_i - \ell^0_i)
            + \frac{\ell_I - \lambda}{u^0_o - \ell^0_o}(y_o - \ell^0_o),
            \qquad \forall (I, o) \in \mathcal{I}^+, \\[6pt]
            & w \ge \sum_{i \in I} a_i (y_i - u^0_i)
            + \frac{u_I + \lambda}{u^0_o - u^0_o}(y_o - u^0_o),
            \qquad \forall (I, o) \in \mathcal{I}^- ,
            \end{aligned}
            \right\}.
        $$

        where:

        - vector $l^{0} \in \mathbb{R}^{d}$ with entries $l_i^{0} = \underline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \overline{y}_i$ otherwise,
        - vector $u^{0} \in \mathbb{R}^{d}$ with entries $u_i^{0} = \overline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \underline{y}_i$ otherwise,
        - Given $\textbf{I} \subseteq \textbf{J} = \left\{1, ..., n\right\}$, the lower bound is
        defined as $l_I = \sum_{i \in I}a_il_i^{0} + \sum_{i \notin I}a_iu_i^{0}$ and the
        upper bounds $u_I = \sum_{i \in I}a_iu_i^{0} + \sum_{i \notin I}a_il_i^{0}$,
        - $\mathcal{I}^+ = \{ (I, o) \in 2^J \times J \mid \ell_I \ge \lambda,\; \ell_{I \cup \{o\}}
        < \lambda,\; w_I \neq 0 \},$
        -  $\mathcal{I}^- = \{ (I, o) \in 2^J \times J \mid u_I \le - \lambda,\; u_{I \cup \{o\}} >
        -\lambda,\; w_I \neq 0 \}.$
        - and $w_I$ is the slicing of vector $w$ over $I$.

        Parameters
        ----------
        step: SoftThreshold
            An instance of the `SoftThreshold` step to canonicalize.

        Returns
        -------
        None
        """
        constraints = []

        i = 0
        for lb, ub, output_lb, output_ub, output, arg in zip(
            step.lb, step.ub, step.output.lb, step.output.ub, step.output, step.arg
        ):
            if lb >= step.lambd:
                constraints.append(output == arg - step.lambd)
            elif ub <= -step.lambd:
                constraints.append(output == arg + step.lambd)
            elif lb >= -step.lambd and ub <= step.lambd:
                constraints.append(output == 0)
            else:
                if not step.relax_binary_vars:
                    _require_finite_bounds("Soft-threshold", i, lb, ub)
                    w1 = LinearExpression.from_int(arg.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(arg.ndim, is_binary=True)
                    step.idx_with_right_binary_vars.add(i)

                    if ub > step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        constraints.extend(
                            [
                                arg >= step.lambd + (lb - step.lambd) * (1 - w1),
                                arg <= step.lambd + (ub - step.lambd) * w1,
                                output <= arg - step.lambd + (2 * step.lambd) * (1 - w1),
                            ]
                        )

                        if lb < -step.lambd:
                            constraints.extend(
                                [
                                    output <= output_ub * w1,
                                    output >= output_lb * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                        # Right-or-middle: the middle (zero) sub-region sits on the
                        # LOWER side, so it is reachable iff lb is inside [-lambd,
                        # lambd]. There w1=0 selects the middle and this pin forces
                        # output <= 0; combined with output >= 0 (line 918) => 0.
                        if -step.lambd <= lb <= step.lambd:
                            constraints.append(output <= output_ub * w1)

                    if lb < -step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        constraints.extend(
                            [
                                arg <= -step.lambd + (ub + step.lambd) * (1 - w2),
                                arg >= -step.lambd + (lb + step.lambd) * w2,
                                output >= arg + step.lambd + (-2 * step.lambd) * (1 - w2),
                            ]
                        )

                        # Left-or-middle: the middle (zero) sub-region sits on the
                        # UPPER side, so it is reachable iff ub is inside [-lambd,
                        # lambd] (guard on ub, mirroring the lb guard above for the
                        # right side). There w2=0 selects the middle and this pin
                        # forces output >= 0; combined with output <= 0 (line 927)
                        # => 0. Guarding on lb here was a bug: this block already
                        # requires lb < -lambd, so an ``lb`` guard is never true and
                        # the middle output was left unpinned, admitting off-graph
                        # points (output in [arg-lambd, 0] when |arg| < lambd).
                        if -step.lambd <= ub <= step.lambd:
                            constraints.append(output >= output_lb * w2)

                constraints.extend(
                    softthreshold_envelope(output, arg, lb, ub, output_lb, output_ub, step.lambd)
                )

            i += 1

        self.add_constraints(constraints)
        self.steps.append(step)

    _CANONICALIZERS = {
        "affine": canonicalize_affine,
        "relu": canonicalize_relu,
        "saturatedlinear": canonicalize_saturatedlinear,
        "softthreshold": canonicalize_softthreshold,
    }
