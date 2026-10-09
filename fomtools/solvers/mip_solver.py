from functools import cached_property

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.solvers.solver import Solver
from fomtools.utils.convex_hull_cuts import separate_relu, separate_satlin, separate_soft_threshold


class MIPSolver(Solver):
    """
    Abstract base class for Mixed-Integer Programming (MIP) solvers.

    This class extends the base Solver class to provide MIP-specific functionality,
    including model relaxation and convex hull cut generation. It serves as the
    parent class for concrete MIP solver implementations like GurobiSolver and
    HiGHSSolver.

    The MIPSolver introduces methods for:
    - Relaxing integer variables to continuous variables
    - Retrieving relaxed variable values
    - Adding convex hull cuts to tighten the relaxation

    Subclasses must implement the abstract methods `relax_model` and
    `get_relaxed_variables` according to their specific solver backend.

    Parameters
    ----------
    solver_params : dict, optional
        Solver-specific parameters and configuration options.

    Attributes
    ----------
    relaxed_model : cached_property
        The relaxed version of the MIP model with all integrality restrictions,
        SOS conditions, and general constraints removed.
    """

    def __init__(self, solver_params=None):
        super(MIPSolver, self).__init__(solver_params=solver_params)

    def add_sum_squares_constr(self, expr, radius_sq, sense) -> None:
        """
        Add a sum-of-squares (quadratic) constraint to the model.

        Default implementation: reject it. Most MIP backends (e.g. HiGHS) cannot
        represent quadratic constraints (MIQCP); quadratic-capable backends (Gurobi)
        override this. The MIP canonicalizer forwards every ``QuadraticConstraint``
        straight here, so the rejection surfaces eagerly at ``add_constraint`` time and
        names the actual solver rather than hardcoding one.

        Raises
        ------
        ValueError
            Always, unless overridden by a quadratic-capable solver.
        """
        raise ValueError(
            f"Solver '{type(self).__name__}' does not support quadratic constraints "
            f"(MIQCP). Use a quadratic-capable solver such as solver='gurobi'."
        )

    def set_two_norm_objective(self, expr_list, lb_list, ub_list) -> None:
        """
        Maximize the sum of squares of ``expr_list`` (a two-norm objective).

        Default implementation: reject it, since most MIP backends cannot represent
        a quadratic objective (MIQP); quadratic-capable backends override this.

        Raises
        ------
        ValueError
            Always, unless overridden by a quadratic-capable solver.
        """
        raise ValueError(
            f"Solver '{type(self).__name__}' does not support MIQP, which the two-norm "
            "objective requires. Use a quadratic-capable solver, or switch to the "
            "infinity-norm or one-norm objective."
        )

    def relax_model(self):
        """
        Relax the MIP model to a continuous linear program.

        This method creates a relaxed version of the model where all integer
        variables are converted to continuous variables, and special ordered
        set (SOS) constraints and general constraints are removed.

        Returns
        -------
        Model
            The relaxed model object (solver-specific).

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_relaxed_variables(self, var):
        """
        Get the relaxed variables corresponding to the given expression.

        This method retrieves the continuous (relaxed) variables and their
        current solution values from the relaxed model.

        Parameters
        ----------
        var : LinearExpression
            The linear expression whose relaxed variables should be retrieved.

        Returns
        -------
        tuple
            A tuple (variables, values) where:
            - variables: The relaxed solver variables
            - values: The current solution values of those variables

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    @cached_property
    def relaxed_model(self):
        """
        Return the relaxed version of the MIP model, in which all integrality
        restrictions, SOS conditions, semi-continuous and semi-integer
        requirements on variables have been relaxed and general constraints
        have been removed.
        """
        return self._relaxed_model

    def relaxed_is_optimal(self) -> bool:
        """Whether the last ``relax_model`` solve ended optimal."""
        raise NotImplementedError

    def separate_convex_hull_cuts(self, steps, tol: float = 1e-6) -> list:
        """
        Separate the relaxed point from the convex hulls of the piecewise affine steps.

        Solves the continuous relaxation of the current model (which should already
        carry the objective) and, for every unstable output component of each ReLU,
        saturated linear and soft-threshold step, returns the most violated hull
        inequality, if any. Other steps are skipped. Nothing is added to the model:
        the caller decides where the cuts go.

        Parameters
        ----------
        steps : list of Step
            The canonicalized steps.
        tol : float, default 1e-6
            Minimum violation for a cut to be returned.

        Returns
        -------
        list[Constraint]
            One constraint per separated component, on the step's input and
            output leaves. ``last_separation_stats`` records the number of
            components examined per step type and the cuts per family.
        """
        from fomtools.steps import Relu, SaturatedLinear, SoftThreshold

        self.last_separation_stats = {"candidates": {}, "cuts_by_family": {}}
        self.relax_model()
        if not self.relaxed_is_optimal():
            return []

        candidates = self.last_separation_stats["candidates"]
        by_family = self.last_separation_stats["cuts_by_family"]
        cuts = []
        for step in steps:
            if isinstance(step, Relu):
                rows = step.idx_with_binary_vars & step.proj_indices
            elif isinstance(step, SaturatedLinear):
                rows = _unstable_satlin_rows(step)
            elif isinstance(step, SoftThreshold):
                rows = _unstable_soft_threshold_rows(step)
            else:
                continue
            if not rows:
                continue
            kind = type(step).__name__
            candidates[kind] = candidates.get(kind, 0) + len(rows)

            leaves = list(step.arg.decomposition_dict)
            L, U = _stacked_leaf_bounds(leaves)
            _, y_val = self.get_relaxed_variables(step.arg)
            # ``step.output_leaves[0]`` is the fresh output variable: the full output
            # leaf, or the smaller proj-leaf for a partial ReLU, whose row for arg
            # row ``i`` is ``arg_to_pos[i]``.
            w_leaf = step.output_leaves[0]
            _, w_val = self.get_relaxed_variables(w_leaf)
            arg_to_pos = {i: j for j, i in enumerate(sorted(step.proj_indices))}

            for i in sorted(rows):
                pos = arg_to_pos[i]
                a = np.concatenate([_coeff_row(v, i) for v in step.arg.decomposition_dict.values()])
                used = a != 0
                if not (np.isfinite(L[used]).all() and np.isfinite(U[used]).all()):
                    continue
                k = float(np.atleast_1d(step.arg.const)[i])
                if isinstance(step, Relu):
                    cut = separate_relu(a, k, L, U, y_val, w_val[pos], tol)
                elif isinstance(step, SaturatedLinear):
                    l, u = _satlin_limits(step)
                    cut = separate_satlin(a, k, L, U, y_val, w_val[pos], l[i], u[i], tol)
                else:
                    lambd = np.broadcast_to(np.asarray(step.lambd, dtype=float), (step.arg.ndim,))
                    cut = separate_soft_threshold(a, k, L, U, y_val, w_val[pos], lambd[i], tol)
                if cut is not None:
                    cuts.append(_cut_to_constraint(cut, w_leaf[pos], leaves))
                    by_family[cut.family] = by_family.get(cut.family, 0) + 1
        return cuts


def _coeff_row(coeff, i: int) -> np.ndarray:
    """Row ``i`` of a dense or sparse coefficient matrix, as a flat array."""
    if spa.issparse(coeff):
        return coeff.tocsr()[i].toarray().ravel()
    return np.atleast_2d(np.asarray(coeff))[i].ravel()


def _stacked_leaf_bounds(leaves) -> tuple[np.ndarray, np.ndarray]:
    """Current bounds of ``leaves``, stacked in order."""
    L = np.concatenate([np.broadcast_to(np.asarray(f.lb, dtype=float), (f.ndim,)) for f in leaves])
    U = np.concatenate([np.broadcast_to(np.asarray(f.ub, dtype=float), (f.ndim,)) for f in leaves])
    return L, U


def _satlin_limits(step) -> tuple[np.ndarray, np.ndarray] | None:
    """Numeric saturation limits ``(l, u)`` as arrays, or ``None`` if either is parametric."""
    if isinstance(step.l, LinearExpression) or isinstance(step.u, LinearExpression):
        return None
    n = step.arg.ndim
    l = np.broadcast_to(np.asarray(step.l, dtype=float), (n,))
    u = np.broadcast_to(np.asarray(step.u, dtype=float), (n,))
    return l, u


def _unstable_satlin_rows(step) -> set[int]:
    """Rows whose saturation is undecided by the input bounds (the ones with binaries)."""
    limits = _satlin_limits(step)
    if limits is None:
        return set()
    l, u = limits
    lb, ub = np.asarray(step.lb, dtype=float), np.asarray(step.ub, dtype=float)
    stable = (lb >= u) | (ub <= l) | ((lb >= l) & (ub <= u)) | (l >= u)
    return set(np.flatnonzero(~stable).tolist())


def _unstable_soft_threshold_rows(step) -> set[int]:
    """Rows whose soft-threshold branch is undecided by the input bounds."""
    lambd = np.broadcast_to(np.asarray(step.lambd, dtype=float), (step.arg.ndim,))
    lb, ub = np.asarray(step.lb, dtype=float), np.asarray(step.ub, dtype=float)
    stable = (lb >= lambd) | (ub <= -lambd) | ((lb >= -lambd) & (ub <= lambd))
    return set(np.flatnonzero(~stable).tolist())


def _cut_to_constraint(cut, w_expr, leaves):
    """The constraint ``cut`` on the output row ``w_expr`` and the input ``leaves``."""
    decomposition, start = {}, 0
    for leaf in leaves:
        row = cut.alpha[start : start + leaf.ndim]
        start += leaf.ndim
        if np.any(row != 0):
            decomposition[leaf] = row.reshape(1, -1)
    rhs = LinearExpression(decomposition_dict=decomposition, const=np.array([cut.beta]))
    return w_expr <= rhs if cut.sense > 0 else w_expr >= rhs
