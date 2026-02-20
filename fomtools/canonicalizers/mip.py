from typing import List

import gurobipy as gp
import numpy as np

from fomtools import LinearExpression
from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import Constraint
from fomtools.steps import Affine, HalpernStep, Relu, SaturatedLinear, SoftThreshold, Step


class MIP(Canonicalizer):
    """
    An interface for mixed-integer programming (MIP) methods.

    Attributes
    ----------
    solver: Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).

    """

    def __init__(self, solver_params=None, solver="gurobi"):
        super(MIP, self).__init__(solver_params, solver)
        # State for Halpern y-vector (one-hot iteration counter) per sequence id
        self._halpern_sequence_state = {}
        self._halpern_iter_count = 0

    def add_var(self, var: LinearExpression) -> None:
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
        self.solver.add_constr(constraint)

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

    def set_zero_objective(self):
        # this function is mostly just for debugging the constraints
        self.solver.set_objective(expression=0)

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
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

        self.solver.constraints_to_solver(constraints)
        self.solver.set_objective(expression=q, sense="maximize")

    def solve_model(self, steps: Step, convex_hull_cuts: bool = False) -> float:
        """
        Optimize the underlying MIP problem.

        Optionally, add convex hull cuts

        Parameters
        ----------
        steps: Step


        convex_hull_cuts: bool
            If ``True`` convex hull cuts are added to the MIP problem.

        Returns
        -------
        float
            The objective value for the current solution.

        """
        if convex_hull_cuts:
            self.solver.add_convex_hull_cuts(steps)

        self.solver.optimize()
        return self.solver["objVal"]

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
        """
        return self.solver.get_solution_value(expression)

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
        self.add_constraint(step.lhs_mat @ step.lhs == step.rhs)

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
        List[Constraint]
            A list of constraint objects describing the convex hull of the feasible region
            defined by the ReLU activation.
        """
        constraints = []
        for i, (_lhs, _rhs) in enumerate(zip(step.lhs, step.rhs)):
            if i not in step.proj_indices:
                constraints.append(_lhs == _rhs)
            else:
                if step.ub[i] <= 0:
                    constraints.append(_lhs == 0)
                elif step.lb[i] > 0:
                    constraints.append(_lhs == _rhs)
                elif step.ub[i] > 0 or step.lb[i] <= 0:
                    new_var = LinearExpression.from_int(_rhs.ndim, is_binary=True)
                    step.idx_with_binary_vars.add(i)
                    constraints.append(_lhs <= step.ub[i] * new_var)

                    constraints.extend(
                        [
                            _lhs <= step.ub[i] / (step.ub[i] - step.lb[i]) * (_rhs - step.lb[i]),
                            _lhs >= _rhs,
                            _lhs <= _rhs - step.lb[i] * (1 - new_var),
                        ]
                    )
        self.add_constraints(constraints)

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

        for l, u, lb, ub, lhs, rhs in zip(step.l, step.u, step.lb, step.ub, step.lhs, step.rhs):
            # If the lower and upper saturation bounds are LinearExpression objects,
            # then the lower and upper bounds of the LinearExpression will define
            # the respective lower and upper saturation bounds
            # These objects have already been registered with the solver
            # so no further action is needed
            try:
                float(l), float(u)
            except TypeError:
                return

            if lb >= u:
                constraints.append(lhs == u)
            elif ub <= l:
                constraints.append(lhs == l)
            elif lb >= l and ub <= u:
                constraints.append(lhs == rhs)
            else:
                if not step.relax_binary_vars:
                    w1 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(rhs.ndim, is_binary=True)

                    if lb < l:
                        constraints.extend(
                            [
                                lhs <= rhs + (ub - lb) * w2,
                                rhs >= l + (lb - l) * w2,
                                rhs <= l + (ub - l) * (1 - w2),
                            ]
                        )

                        if ub > u:
                            constraints.extend(
                                [
                                    lhs >= u - (u - l) * (1 - w1),
                                    lhs <= l + (u - l) * (1 - w2),
                                    lhs >= l + (lb - l) * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                    if ub > u:
                        constraints.extend(
                            [
                                lhs >= rhs - (ub - lb) * w1,
                                rhs >= u + (lb - u) * (1 - w1),
                                rhs <= u + (ub - u) * w1,
                            ]
                        )
                        if lb > l:
                            constraints.append(lhs >= u - (u - lb) * (1 - w1))

                if lb < l and ub > u:
                    expr = (u - l) / (u - lb) * (rhs - u) + u
                    constraints.extend([lhs <= expr, lhs >= expr])

                elif lb < l and ub < u:
                    constraints.extend(
                        [lhs >= rhs, lhs <= ((ub - l) / (ub - lb)) * (rhs - ub) + ub]
                    )

                elif lb > l and ub > u:
                    constraints.extend(
                        [lhs <= rhs, lhs >= ((u - lb) / (ub - lb) * (rhs - lb) + lb)]
                    )

        self.add_constraints(constraints)

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
        for lb, ub, lhs_lb, lhs_ub, lhs, rhs in zip(
            step.lb, step.ub, step.lhs.lb, step.lhs.ub, step.lhs, step.rhs
        ):
            if lb >= step.lambd:
                constraints.append(lhs == rhs - step.lambd)
            elif ub <= -step.lambd:
                constraints.append(lhs == rhs + step.lambd)
            elif lb >= -step.lambd and ub <= step.lambd:
                constraints.append(lhs == 0)
            else:
                if not step.relax_binary_vars:
                    w1 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    step.idx_with_right_binary_vars.add(i)

                    if ub > step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        constraints.extend(
                            [
                                rhs >= step.lambd + (lb - step.lambd) * (1 - w1),
                                rhs <= step.lambd + (ub - step.lambd) * w1,
                                lhs <= rhs - step.lambd + (2 * step.lambd) * (1 - w1),
                            ]
                        )

                        if lb < -step.lambd:
                            constraints.extend(
                                [
                                    lhs <= lhs_ub * w1,
                                    lhs >= lhs_lb * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                        if -step.lambd <= lb <= step.lambd:
                            constraints.append(lhs <= lhs_ub * w1)

                    if lb < -step.lambd:
                        step.idx_with_left_binary_vars.add(i)
                        constraints.extend(
                            [
                                rhs <= -step.lambd + (ub + step.lambd) * (1 - w2),
                                rhs >= -step.lambd + (lb + step.lambd) * w2,
                                lhs >= rhs + step.lambd + (-2 * step.lambd) * (1 - w2),
                            ]
                        )

                        if -step.lambd <= lb <= step.lambd:
                            constraints.append(lhs >= lhs_lb * w2)

                if lb < -step.lambd and ub > step.lambd:
                    constraints.extend(
                        [
                            lhs >= rhs - step.lambd,
                            lhs <= rhs + step.lambd,
                            lhs <= lhs_ub / (ub + step.lambd) * (rhs + step.lambd),
                            lhs >= lhs_lb / (lb - step.lambd) * (rhs - step.lambd),
                        ]
                    )

                elif -step.lambd <= lb <= step.lambd < ub:
                    constraints.extend(
                        [
                            lhs >= 0,
                            lhs <= lhs_ub / (ub - lb) * (rhs - lb),
                            lhs >= rhs - step.lambd,
                        ]
                    )

                elif step.lambd >= ub >= -step.lambd >= lb:
                    constraints.extend(
                        [
                            lhs <= 0,
                            lhs >= lhs_lb / (lb - ub) * (rhs - ub),
                            lhs <= rhs + step.lambd,
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

            i += 1

        self.add_constraints(constraints)

    def _get_or_init_halpern_y(self, sequence_id: str, K: int):
        """
        Initialize or retrieve the Halpern one-hot y-state vector.

        The y-state vector has length K+1 and tracks the iteration counter m.
        Initially m=0, so y[0]=1 and y[1..K]=0.

        Parameters
        ----------
        sequence_id : str
            Identifier for the Halpern sequence.
        K : int
            Maximum iteration count for the step-size schedule.

        Returns
        -------
        gp.MVar
            Binary MVar of length K+1 representing the current one-hot state.
        """
        if sequence_id not in self._halpern_sequence_state:
            model = self.solver.model
            y0 = model.addMVar(K + 1, vtype=gp.GRB.BINARY, name="halpern_y_0")
            model.addConstr(y0[0] == 1)
            for m in range(1, K + 1):
                model.addConstr(y0[m] == 0)
            model.addConstr(gp.quicksum(y0) == 1)
            self._halpern_sequence_state[sequence_id] = y0
        return self._halpern_sequence_state[sequence_id]

    def canonicalize_halpernstep(self, step: HalpernStep) -> None:
        """
        If step.dynamic is True:
          - Computes ρ = ||z^k - T(z^k)||_∞ and σ = ||a^k - T(a^k)||_∞.
          - Introduces restart binary r: ρ ≤ (1/ε)σ ⟹ r=1.
          - Updates the one-hot y-state from y_k to y_{k+1} based on r.
          - Sets a^{k+1} = T(z^k) if r=1, else a^{k+1} = a^k.
          - Sets z^{k+1} = Σ_m y_{k+1}[m] * [α_m T(z^k) + (1-α_m) a^{k+1}].

        If its False: 
          - Uses a fixed step size α_k = (k+1)/(k+2).
          - Sets z^{k+1} = α_k T(z^k) + (1-α_k) a^k.
          - Sets a^{k+1} = a^k (anchor stays constant).
        """
        model = self.solver.model
        K = step.K
        eps = step.eps
        big_M = step.big_M if step.big_M is not None else 1e3

        z_k_gp = self.solver.expression_to_solver(step.z_k_expr)
        a_k_gp = self.solver.expression_to_solver(step.anchor_expr)
        t_zk_gp = self.solver.expression_to_solver(step.t_zk_expr)
        t_ak_gp = self.solver.expression_to_solver(step.t_anchor_expr)

        n = step.lhs.ndim
        lhs_gp = self.solver.expression_to_solver(step.lhs)
        a_next_gp = self.solver.expression_to_solver(step.a_next_expr)

        if step.dynamic:
            vtype = gp.GRB.CONTINUOUS if step.relax_binary_vars else gp.GRB.BINARY

            #\rho = ||z^k - T(z^k)||_inf
            diff_z = z_k_gp - t_zk_gp
            u = model.addMVar(n, lb=0.0, ub=big_M, name=f"u_rho_{step.sequence_id}")
            b_rho = model.addMVar(n, vtype=vtype, name=f"b_sign_rho_{step.sequence_id}")
            if step.relax_binary_vars:
                b_rho.lb, b_rho.ub = 0.0, 1.0
            for i in range(n):
                d = diff_z[i]
                model.addConstr(u[i] >= d)
                model.addConstr(u[i] >= -d)
                model.addConstr(u[i] <= d + big_M * (1 - b_rho[i]))
                model.addConstr(u[i] <= -d + big_M * b_rho[i])

            rho = model.addVar(lb=0.0, ub=big_M, name=f"rho_{step.sequence_id}")
            s_rho = model.addMVar(n, vtype=vtype, name=f"s_argmax_rho_{step.sequence_id}")
            if step.relax_binary_vars:
                s_rho.lb, s_rho.ub = 0.0, 1.0
            model.addConstr(gp.quicksum(s_rho) == 1)
            for i in range(n):
                model.addConstr(rho >= u[i])
                model.addConstr(rho <= u[i] + big_M * (1 - s_rho[i]))

            #\sigma = ||a^k - T(a^k)||_inf
            diff_a = a_k_gp - t_ak_gp
            v = model.addMVar(n, lb=0.0, ub=big_M, name=f"u_sigma_{step.sequence_id}")
            b_sig = model.addMVar(n, vtype=vtype, name=f"b_sign_sigma_{step.sequence_id}")
            if step.relax_binary_vars:
                b_sig.lb, b_sig.ub = 0.0, 1.0
            for i in range(n):
                d = diff_a[i]
                model.addConstr(v[i] >= d)
                model.addConstr(v[i] >= -d)
                model.addConstr(v[i] <= d + big_M * (1 - b_sig[i]))
                model.addConstr(v[i] <= -d + big_M * b_sig[i])

            sigma = model.addVar(lb=0.0, ub=big_M, name=f"sigma_{step.sequence_id}")
            s_sig = model.addMVar(n, vtype=vtype, name=f"s_argmax_sigma_{step.sequence_id}")
            if step.relax_binary_vars:
                s_sig.lb, s_sig.ub = 0.0, 1.0
            model.addConstr(gp.quicksum(s_sig) == 1)
            for i in range(n):
                model.addConstr(sigma >= v[i])
                model.addConstr(sigma <= v[i] + big_M * (1 - s_sig[i]))

            #Restart binary r: r=1 if \hro ≤ (1/\epsilon)\sigma
            r = model.addVar(vtype=vtype, name=f"r_{step.sequence_id}")
            if step.relax_binary_vars:
                r.lb, r.ub = 0.0, 1.0
            model.addConstr(rho - (1.0 / eps) * sigma <= big_M * (1 - r))
            model.addConstr((1.0 / eps) * sigma - rho <= big_M * r)

            #one hot transition
            y_k = self._get_or_init_halpern_y(step.sequence_id, K)
            y_kp1 = model.addMVar(K + 1, vtype=vtype, name="halpern_y_next")
            if step.relax_binary_vars:
                y_kp1.lb, y_kp1.ub = 0.0, 1.0
            model.addConstr(gp.quicksum(y_kp1) == 1)
            #y_{k+1}[0] = r  (reset on restart)
            model.addConstr(y_kp1[0] == r)
            #y_{k+1}[m+1] = y_k[m] and (1-r)  for m = 0..K-1
            for m in range(K):
                model.addConstr(y_kp1[m + 1] <= y_k[m])
                model.addConstr(y_kp1[m + 1] <= 1 - r)
                model.addConstr(y_kp1[m + 1] >= y_k[m] - r)
            self._halpern_sequence_state[step.sequence_id] = y_kp1

            # update anchor
            for i in range(n):
                model.addConstr(a_next_gp[i] - t_zk_gp[i] <= big_M * (1 - r))
                model.addConstr(t_zk_gp[i] - a_next_gp[i] <= big_M * (1 - r))
                model.addConstr(a_next_gp[i] - a_k_gp[i] <= big_M * r)
                model.addConstr(a_k_gp[i] - a_next_gp[i] <= big_M * r)
            alpha = step.alpha_array
            tilde_t = []
            tilde_a = []
            for m in range(K + 1):
                tt = model.addMVar(n, lb=-big_M, ub=big_M, name=f"tt_{step.sequence_id}_{m}")
                ta = model.addMVar(n, lb=-big_M, ub=big_M, name=f"ta_{step.sequence_id}_{m}")
                tilde_t.append(tt)
                tilde_a.append(ta)
                model.addConstr(tt <= big_M * y_kp1[m])
                model.addConstr(tt >= -big_M * y_kp1[m])
                model.addConstr(ta <= big_M * y_kp1[m])
                model.addConstr(ta >= -big_M * y_kp1[m])
                for i in range(n):
                    model.addConstr(t_zk_gp[i] - tt[i] <= big_M * (1 - y_kp1[m]))
                    model.addConstr(tt[i] - t_zk_gp[i] <= big_M * (1 - y_kp1[m]))
                    model.addConstr(a_next_gp[i] - ta[i] <= big_M * (1 - y_kp1[m]))
                    model.addConstr(ta[i] - a_next_gp[i] <= big_M * (1 - y_kp1[m]))

            for i in range(n):
                expr = gp.LinExpr()
                for m in range(K + 1):
                    expr += float(alpha[m]) * tilde_t[m][i]
                    expr += float(1.0 - alpha[m]) * tilde_a[m][i]
                model.addConstr(lhs_gp[i] == expr)

        else:
            # Non-dynamic
            k_plus_1 = self._halpern_iter_count + 1
            alpha_k = k_plus_1 / (k_plus_1 + 1)

            for i in range(n):
                model.addConstr(
                    lhs_gp[i] == alpha_k * t_zk_gp[i] + (1 - alpha_k) * a_k_gp[i]
                )
            for i in range(n):
                model.addConstr(a_next_gp[i] == a_k_gp[i])

            self._halpern_iter_count += 1

        model.update()
