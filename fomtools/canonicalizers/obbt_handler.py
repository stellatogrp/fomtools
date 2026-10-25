import logging
from typing import ClassVar, List

import numpy as np

from fomtools.canonicalizers.canonicalizer import Canonicalizer
from fomtools.constraint import Constraint, is_binary_free
from fomtools.linearexpression import LinearExpression
from fomtools.steps import Affine, Relu, SaturatedLinear, SoftThreshold
from fomtools.utils.envelopes import relu_upper_envelope, softthreshold_envelope
from fomtools.utils.math_utils import repair_crossed_bounds

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
        A ready-to-use instance of a subclass of `Solver`
        (e.g., ``gurobi.Gurobi()``).
    crossing_tol : float
        Largest relative crossing ``min > max`` of an OBBT bound pair treated as
        LP solver noise; such entries are swapped and buffered outward by the
        same amount. Defaults to Gurobi's default feasibility and optimality
        tolerances.
    crossing_repairs : int
        Number of bound entries repaired this way so far.
    lp_retries : int
        Number of OBBT LPs not solved to optimality warm-started but solved
        from scratch (``Solver.optimize_from_scratch``) so far.
    lp_failures : int
        Number of OBBT LPs not solved to optimality even from scratch so far;
        each keeps the input bound of its entry and sense.
    """

    def __init__(self, solver: str, solver_params: dict = None, crossing_tol: float = 1e-6):
        """
        Initialize an instance of `OBBTHandler`.

        Parameters
        ----------
        solver : str
            Case-insensitive identifier for the solver to use. Supported
            values (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
            - ``"highs"``
        solver_params : dict or None, optional
            Parameters passed verbatim to the OBBT solver (e.g.
            ``{"OutputFlag": 0}`` for Gurobi).
        crossing_tol : float, default 1e-6
            See the class attribute of the same name.
        """
        super().__init__(solver_params=solver_params, solver=solver)
        self.crossing_tol = crossing_tol
        self.crossing_repairs = 0
        self.lp_retries = 0
        self.lp_failures = 0

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
        input bounds and the computed bounds, padded outward by
        ``1e-6 * max(1, abs(bound))``. This does not certify rounding errors.

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
            and upper bounds. An LP not solved to optimality is solved once
            more from scratch; if that fails too, its entry keeps the input
            bound for that sense, and the other entries are still tightened.
            Returns ``None`` if a bound stays nonfinite.
        """
        # An LP that is not solved to optimality leaves the input bound in place.
        lb_out = np.array(lb, dtype=float, copy=True)
        ub_out = np.array(ub, dtype=float, copy=True)

        for sense in ["minimize", "maximize"]:
            for i, expr in enumerate(linexpr):
                self.solver.set_objective(expr, sense)
                self.solver.optimize()

                if not self.solver.is_optimal():
                    # A warm-started solve can end with a wrong status (on della, a
                    # feasible LP came back INFEASIBLE); retry once from scratch.
                    status = self.solver["status"]
                    self.solver.optimize_from_scratch()
                    if self.solver.is_optimal():
                        self.lp_retries += 1
                        log.info(
                            "OBBT: LP for entry %d (%s) ended with status %s; solved from scratch",
                            i,
                            sense,
                            status,
                        )
                    else:
                        self.lp_failures += 1
                        log.info(
                            "OBBT: LP for entry %d (%s) not solved to optimality (status %s, "
                            "%s from scratch); keeping its input bound",
                            i,
                            sense,
                            status,
                            self.solver["status"],
                        )
                        continue

                if sense == "maximize":
                    ub_out[i] = self.solver["objVal"]
                else:
                    lb_out[i] = self.solver["objVal"]

        # reset objective
        self.solver.set_objective(0)

        # Both the raw LP bounds (min/max solves for the same index) and their
        # intersection with the input bounds can cross by solver noise (e.g.
        # input ub = 0, LP lb = 2e-17), up to the LP tolerances. Swap such
        # entries and buffer them outward by the tolerance so the interval
        # stays valid. Crossings beyond the tolerance raise (genuine bug).
        if not (np.isfinite(lb_out).all() and np.isfinite(ub_out).all()):
            log.info("bound tightening failed: nonfinite LP objective")
            return None
        lb_out, ub_out = self._repair(lb_out, ub_out)
        # Pad the LP bounds outward to cover small LP solver errors, matching the
        # objective-bound padding policy. This is a numerical safeguard, not a
        # directed-rounding certificate.
        lb_out = lb_out - 1e-6 * np.maximum(1.0, np.abs(lb_out))
        ub_out = ub_out + 1e-6 * np.maximum(1.0, np.abs(ub_out))
        lb_new, ub_new = self._repair(np.maximum(lb, lb_out), np.minimum(ub, ub_out))
        # The buffer must not loosen the input bounds (e.g. a ReLU output's
        # lb = 0). The repaired interval always meets [lb, ub], so this clip
        # cannot cross it again.
        return np.maximum(lb_new, lb), np.minimum(ub_new, ub)

    def _repair(self, lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``repair_crossed_bounds`` at the LP tolerance, counting and logging repairs."""
        lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
        crossed = lb > ub
        if np.any(crossed):
            self.crossing_repairs += int(crossed.sum())
            log.info(
                "OBBT: repaired %d crossed bound(s), largest crossing %.3e",
                int(crossed.sum()),
                float(np.max(lb - ub)),
            )
        return repair_crossed_bounds(lb, ub, tol=self.crossing_tol, buffer=self.crossing_tol)

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
        """Check that the constraints are binary-free, then add them to the LP."""
        if not all(is_binary_free(c) for c in constraints):
            raise ValueError("OBBT handler received a constraint involving a binary variable")
        self.add_constraints(constraints)

    def canonicalize_affine(self, step: Affine) -> None:
        """
        Add the affine equality ``output_mat @ output == arg`` to the OBBT LP.

        ``output_mat`` is a documented attribute of the `Affine` step
        interface; this mirrors ``MIP.canonicalize_affine`` exactly (both rely
        on the same equality), so the two stay consistent if ``Affine`` is
        refactored.
        """
        self._add_binary_free([step.output_mat @ step.output == step.arg])

    def canonicalize_relu(self, step: Relu) -> None:
        """
        Add the binary-free ReLU relaxation rows to the OBBT LP.

        Mirrors ``MIP._build_relu_constraints`` keeping only the equality
        cases and, for the uncertain region, the triangle envelope
        ``{ output >= arg, output <= ub/(ub-lb) * (arg - lb) }``. The big-M rows are
        dropped. Iterates ``proj_indices`` only — for partial-projection
        ReLU the non-proj rows of ``step.output`` share decomposition with
        ``step.arg`` (composite passthrough), so emitting ``output[i] == arg[i]``
        would reduce to a degenerate ``0 == 0`` that the solver rejects.
        """
        constraints = []
        for i in sorted(step.proj_indices):
            _output = step.output[i]
            _arg = step.arg[i]
            if step.ub[i] <= 0:
                constraints.append(_output == 0)
            elif step.lb[i] > 0:
                constraints.append(_output == _arg)
            else:
                constraints.append(_output >= _arg)
                constraints.extend(relu_upper_envelope(_output, _arg, step.lb[i], step.ub[i]))
        self._add_binary_free(constraints)

    def canonicalize_saturatedlinear(self, step: SaturatedLinear) -> None:
        """
        Add the binary-free saturated-linear relaxation rows to the OBBT LP.

        Mirrors ``MIP.canonicalize_saturatedlinear`` keeping only the
        saturated/pass-through equalities and the convex-envelope slope rows;
        all big-M rows are dropped.
        """
        constraints = []
        for l, u, lb, ub, output, arg in zip(
            step.l, step.u, step.lb, step.ub, step.output, step.arg
        ):
            if lb >= u:
                constraints.append(output == u)
            elif ub <= l:
                constraints.append(output == l)
            elif lb >= l and ub <= u:
                constraints.append(output == arg)
            elif lb < l and ub > u:
                constraints.extend(
                    [
                        output <= (u - l) / (u - lb) * (arg - u) + u,
                        output >= (u - l) / (ub - l) * (arg - l) + l,
                    ]
                )
            elif lb < l and ub <= u:
                constraints.extend(
                    [output >= arg, output <= ((ub - l) / (ub - lb)) * (arg - ub) + ub]
                )
            elif lb >= l and ub > u:
                constraints.extend(
                    [output <= arg, output >= ((u - lb) / (ub - lb) * (arg - lb) + lb)]
                )
        self._add_binary_free(constraints)

    def canonicalize_softthreshold(self, step: SoftThreshold) -> None:
        """
        Add the binary-free soft-threshold relaxation rows to the OBBT LP.

        Mirrors ``MIP.canonicalize_softthreshold`` keeping only the fixed-region
        equalities and the convex-envelope rows; all big-M rows are dropped.
        """
        constraints = []
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
                constraints.extend(
                    softthreshold_envelope(output, arg, lb, ub, output_lb, output_ub, step.lambd)
                )
        self._add_binary_free(constraints)

    _CANONICALIZERS: ClassVar[dict] = {
        "affine": canonicalize_affine,
        "relu": canonicalize_relu,
        "saturatedlinear": canonicalize_saturatedlinear,
        "softthreshold": canonicalize_softthreshold,
    }
