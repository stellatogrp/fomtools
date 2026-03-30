from functools import cached_property

from fomtools import LinearExpression


class Step:
    """
    A base class for thresholding and activation steps. This class
    defines a common interface for step-like operators that transform
    inputs element wise.

    A step represents the relation: lhs = f(rhs), where f is the step
    function (e.g., ReLU, soft threshold, saturated linear).

    Attributes
    ----------
    rhs : LinearExpression
        Right-hand side (input) linear expression.
    lhs : LinearExpression
        Left-hand side (output) linear expression. Bounds on the output
        are stored in ``lhs.lb`` and ``lhs.ub``.
    lb : np.ndarray
        Lower bounds on the INPUT (rhs). These are the bounds before
        the step function is applied.
    ub : np.ndarray
        Upper bounds on the INPUT (rhs). These are the bounds before
        the step function is applied.
    relax_binary_vars : bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    skip_obbt : bool
        If true, OBBT bound tightening is skipped for this step type.
        Override in subclasses that compute bounds analytically.

    Notes
    -----
    Input vs output bounds:
        - ``step.lb``, ``step.ub``: bounds on the INPUT (rhs)
        - ``step.lhs.lb``, ``step.lhs.ub``: bounds on the OUTPUT (lhs)

    When ``update_bounds(lb, ub)`` is called with tightened input bounds,
    subclasses should update both ``self.lb``/``self.ub`` (input) and
    ``self.lhs.lb``/``self.lhs.ub`` (output, via applying the step function).
    """

    skip_obbt: bool = False

    def __init__(self, rhs: LinearExpression, relax_binary_vars: bool = False):
        self.rhs = rhs
        self.lb, self.ub = self.rhs.bound_propagation()
        self.relax_binary_vars = relax_binary_vars

    @cached_property
    def lhs(self) -> LinearExpression:
        """Left-hand side linear expression in the operator definition."""
        return LinearExpression.from_int(self.rhs.ndim)

    def update_bounds(self, lb, ub):
        pass

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.rhs, self.lhs))
