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
        self.relax_binary_vars = relax_binary_vars
        # Default: every output coordinate is "projected" (i.e., the step
        # transformation applies to all rows). Subclasses that support
        # partial projection (e.g. ``Relu`` with ``proj_ranges``) override
        # this in their own ``__init__``.
        self.proj_indices = set(range(rhs.ndim))

    @cached_property
    def lhs(self) -> LinearExpression:
        """Left-hand side linear expression in the operator definition."""
        return LinearExpression.from_int(self.rhs.ndim)

    @property
    def lhs_leaves(self) -> list[LinearExpression]:
        """Leaves this step introduces and that must be registered with the
        canonicalizer / step-decomposer.

        Default: ``[self.lhs]`` (the lhs is itself a leaf). Steps that expose
        a non-leaf composite ``lhs`` (currently only partial-projection
        ``Relu``) override this to return the smaller fresh leaf actually
        introduced.
        """
        return [self.lhs]

    def update_bounds(self, lb, ub):
        raise NotImplementedError

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.rhs, self.lhs))
