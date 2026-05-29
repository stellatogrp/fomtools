import logging
from typing import ClassVar, List

import numpy as np

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import Constraint, is_binary_free
from fomtools.linearexpression import LinearExpression
from fomtools.steps import Affine, Relu, SaturatedLinear, SoftThreshold

log = logging.getLogger(__name__)


class OBBTHandler(Canonicalizer):
    """
    Handler for Optimization-Based Bound Tightening (OBBT).

    This class provides an interface for tightening variable bounds using
    optimization. It maintains a separate solver instance that is used to
    compute tighter lower and upper bounds on variables by solving
    minimization and maximization problems.

    Attributes
    ----------
    solver : Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).
    """

    def __init__(self, solver: str):
        """
        Initialize an instance of :class:`OBBTHandler`.

        Parameters
        ----------
        solver : str
            Case-insensitive identifier for the solver to use. Supported
            values (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
            - ``"highs"``
        """
        super().__init__(solver=solver)

    def add_var(self, var: LinearExpression) -> None:
        """
        Add a variable to the OBBT handler instance.

        This method registers a new variable with the OBBT solver so it
        can be used in bound tightening optimization problems.

        Parameters
        ----------
        var : LinearExpression
            Linear expression object to add to the OBBT instance.

        Returns
        -------
        None
        """
        self.solver.add_var(var, var.lb, var.ub)

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint to the OBBT handler instance.

        This method registers a constraint with the OBBT solver so it
        is considered during bound tightening optimization.

        Parameters
        ----------
        constraint : Constraint
            Constraint object to add to the OBBT instance.

        Returns
        -------
        None
        """
        self.solver.add_constr(constraint)

    def add_constraints(self, constraints: List[Constraint]) -> None:
        """
        Add a list of constraints to the OBBT handler instance.

        Parameters
        ----------
        constraints : List[Constraint]
            Constraint objects to add to the OBBT instance. Every constraint
            must be binary-free (the OBBT model is a pure linear program).

        Returns
        -------
        None
        """
        self.solver.constraints_to_solver(constraints)

    def update_bounds(
        self, expression: LinearExpression, lb: float | np.ndarray, ub: float | np.ndarray
    ) -> None:
        """
        Update the lower and upper bounds of the solver variables corresponding
        to the given ``LinearExpression``.

        Parameters
        ----------
        expression : LinearExpression
            A ``LinearExpression`` object whose underlying variable
            bounds are to be updated.

        lb : float or np.ndarray
            New lower bound(s).

        ub : float or np.ndarray
            New upper bound(s).

        Returns
        -------
        None
        """
        self.solver.update_bounds(expression, lb, ub)

    def obbt(
        self, linexpr: LinearExpression, lb: np.ndarray, ub: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Perform Optimization-Based Bound Tightening on a linear expression.

        For each element in the linear expression, this method solves
        minimization and maximization problems to compute potentially
        tighter bounds. The returned bounds are the intersection of the
        input bounds and the computed bounds.

        Parameters
        ----------
        linexpr : LinearExpression
            The linear expression whose bounds are to be tightened.

        lb : np.ndarray
            Current lower bounds for each element of the expression.

        ub : np.ndarray
            Current upper bounds for each element of the expression.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            A tuple ``(new_lb, new_ub)`` containing the tightened lower
            and upper bounds. Returns ``None`` if the optimization failed
            (e.g., the problem is infeasible or unbounded).
        """
        lb_out = np.zeros(linexpr.ndim)
        ub_out = np.zeros(linexpr.ndim)

        for sense in ["minimize", "maximize"]:
            for i, expr in enumerate(linexpr):
                self.solver.set_objective(expr, sense)
                self.solver.optimize()

                if not self.solver.is_optimal():
                    log.info("bound tightening failed: OBBT model not solved to optimality")
                    log.info(f"i = {i}, sense={sense}")
                    return

                if sense == "maximize":
                    ub_out[i] = self.solver["objVal"]
                else:
                    lb_out[i] = self.solver["objVal"]

        # reset objective
        self.solver.set_objective(0)

        # Tolerance guards against minor LP-solver inconsistency between the
        # min and max solves for the same index (e.g. lb_out slightly above
        # ub_out); the intersection below clamps the bounds either way.
        assert np.all(lb_out <= ub_out + 1e-7)
        return np.maximum(lb, lb_out), np.minimum(ub, ub_out)

    # ------------------------------------------------------------------
    # Linear (binary-free) canonicalization of steps into the OBBT LP.
    #
    # These methods mirror the *linear* equality and convex-envelope rows of
    # the MIP canonicalizer (``fomtools/canonicalizers/mip.py``) with all
    # binary-indicator (big-M) rows and Huchette cuts removed. They are an
    # independent reimplementation -- they do not import or call MIP code --
    # so the OBBT model is a pure LP. A small amount of duplication of the row
    # expressions is intentional. Every method must be invoked with the step's
    # *already-tightened* bounds, since the envelope coefficients bake in
    # ``lb``/``ub``.
    # ------------------------------------------------------------------

    def _add_binary_free(self, constraints: List[Constraint]) -> None:
        """Check the constraints are binary-free, then add them to the LP."""
        if not all(is_binary_free(c) for c in constraints):
            raise ValueError(
                "OBBT handler received a constraint involving a binary variable"
            )
        self.add_constraints(constraints)

    def canonicalize_affine(self, step: Affine) -> None:
        """
        Add the affine equality ``lhs_mat @ lhs == rhs`` to the OBBT LP.

        ``lhs_mat`` is a documented attribute of the :class:`Affine` step
        interface; this mirrors ``MIP.canonicalize_affine`` exactly (both rely
        on the same equality), so the two stay consistent if ``Affine`` is
        refactored.
        """
        self._add_binary_free([step.lhs_mat @ step.lhs == step.rhs])

    def canonicalize_relu(self, step: Relu) -> None:
        """
        Add the binary-free ReLU relaxation rows to the OBBT LP.

        Mirrors ``MIP._build_relu_constraints`` keeping only the equality
        cases and, for the uncertain region, the triangle envelope
        ``{ lhs >= rhs, lhs <= ub/(ub-lb) * (rhs - lb) }``. The big-M rows are
        dropped.
        """
        constraints = []
        for i, (_lhs, _rhs) in enumerate(zip(step.lhs, step.rhs)):
            if i not in step.proj_indices:
                constraints.append(_lhs == _rhs)
            elif step.ub[i] <= 0:
                constraints.append(_lhs == 0)
            elif step.lb[i] > 0:
                constraints.append(_lhs == _rhs)
            else:
                constraints.extend(
                    [
                        _lhs <= step.ub[i] / (step.ub[i] - step.lb[i]) * (_rhs - step.lb[i]),
                        _lhs >= _rhs,
                    ]
                )
        self._add_binary_free(constraints)

    def canonicalize_saturatedlinear(self, step: SaturatedLinear) -> None:
        """
        Add the binary-free saturated-linear relaxation rows to the OBBT LP.

        Mirrors ``MIP.canonicalize_saturatedlinear`` keeping only the
        saturated/pass-through equalities and the convex-envelope slope rows;
        all big-M rows are dropped.
        """
        constraints = []
        for l, u, lb, ub, lhs, rhs in zip(
            step.l, step.u, step.lb, step.ub, step.lhs, step.rhs
        ):
            if lb >= u:
                constraints.append(lhs == u)
            elif ub <= l:
                constraints.append(lhs == l)
            elif lb >= l and ub <= u:
                constraints.append(lhs == rhs)
            elif lb < l and ub > u:
                constraints.extend(
                    [
                        lhs <= (u - l) / (u - lb) * (rhs - u) + u,
                        lhs >= (u - l) / (ub - l) * (rhs - l) + l,
                    ]
                )
            elif lb < l and ub < u:
                constraints.extend(
                    [lhs >= rhs, lhs <= ((ub - l) / (ub - lb)) * (rhs - ub) + ub]
                )
            elif lb > l and ub > u:
                constraints.extend(
                    [lhs <= rhs, lhs >= ((u - lb) / (ub - lb) * (rhs - lb) + lb)]
                )
            # No else: boundary cases (e.g. lb == l with ub > u, or lb < l with
            # ub == u) intentionally add no envelope row — they are measure-zero
            # and the variable's box bounds remain a valid relaxation. This
            # mirrors MIP.canonicalize_saturatedlinear, which has the same gap.
        self._add_binary_free(constraints)

    def canonicalize_softthreshold(self, step: SoftThreshold) -> None:
        """
        Add the binary-free soft-threshold relaxation rows to the OBBT LP.

        Mirrors ``MIP.canonicalize_softthreshold`` keeping only the fixed-region
        equalities and the convex-envelope rows; all big-M rows are dropped.
        """
        constraints = []
        for lb, ub, lhs_lb, lhs_ub, lhs, rhs in zip(
            step.lb, step.ub, step.lhs.lb, step.lhs.ub, step.lhs, step.rhs
        ):
            if lb >= step.lambd:
                constraints.append(lhs == rhs - step.lambd)
            elif ub <= -step.lambd:
                constraints.append(lhs == rhs + step.lambd)
            elif lb >= -step.lambd and ub <= step.lambd:
                constraints.append(lhs == 0)
            elif lb < -step.lambd and ub > step.lambd:
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
        self._add_binary_free(constraints)

    _CANONICALIZERS: ClassVar[dict] = {
        "affine": canonicalize_affine,
        "relu": canonicalize_relu,
        "saturatedlinear": canonicalize_saturatedlinear,
        "softthreshold": canonicalize_softthreshold,
    }
