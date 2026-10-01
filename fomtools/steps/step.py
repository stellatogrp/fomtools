from functools import cached_property

from fomtools import LinearExpression


class Step:
    """
    A base class for thresholding and activation steps. This class
    defines a common interface for step-like operators that transform
    inputs element wise.

    A step represents the relation: output = f(arg), where f is the step
    function (e.g., ReLU, soft threshold, saturated linear).

    Attributes
    ----------
    arg : LinearExpression
        Right-hand side (input) linear expression.
    output : LinearExpression
        Left-hand side (output) linear expression. Bounds on the output
        are stored in ``output.lb`` and ``output.ub``.
    relax_binary_vars : bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    skip_obbt : bool
        If true, OBBT bound tightening is skipped for this step type.
        Override in subclasses that compute bounds analytically.
    obbt_output : bool
        If true, OBBT is also run on the step's output leaves once the step's
        relaxation is in the OBBT LP. Enabled for :class:`Affine` with non-identity
        ``output_mat``, whose output bounds cannot otherwise see the tightened
        input box without ``M^{-1}``.

    Notes
    -----
    Input vs output bounds:
        - ``step.lb``, ``step.ub``: bounds on the INPUT (arg)
        - ``step.output.lb``, ``step.output.ub``: bounds on the OUTPUT (output)

    When ``update_bounds(lb, ub)`` is called with tightened input bounds,
    subclasses should update both ``self.lb``/``self.ub`` (input) and
    ``self.output.lb``/``self.output.ub`` (output, via applying the step function).
    """

    skip_obbt: bool = False
    obbt_output: bool = False

    def __init__(self, arg: LinearExpression, relax_binary_vars: bool = False):
        self.arg = arg
        self.relax_binary_vars = relax_binary_vars
        # Default: every output coordinate is "projected" (i.e., the step
        # transformation applies to all rows). Subclasses that support
        # partial projection (e.g. ``Relu`` with ``proj_ranges``) override
        # this in their own ``__init__``.
        self.proj_indices = set(range(arg.ndim))

    @cached_property
    def output(self) -> LinearExpression:
        """Left-hand side linear expression in the operator definition."""
        return LinearExpression.from_int(self.arg.ndim)

    @property
    def output_leaves(self) -> list[LinearExpression]:
        """Leaves this step introduces and that must be registered with the
        canonicalizer / step-decomposer.

        Default: ``[self.output]`` (the output is itself a leaf). Steps that expose
        a non-leaf composite ``output`` (currently only partial-projection
        ``Relu``) override this to return the smaller fresh leaf actually
        introduced.
        """
        return [self.output]

    def update_bounds(self, lb, ub):
        raise NotImplementedError

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.arg, self.output))
