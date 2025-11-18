from functools import cached_property

from algoverify import LinearExpression


class Step:
    """
    A base class for thresholding and activation steps. This class
    defines a common interface for step-like operators that transform
    inputs element wise.

    Attributes
    ----------
    lhs: LinearExpression
        Left-hand side linear expression in the operator definition.
    rhs: LinearExpression
        Right-hand side linear expression in the operator definition.
    lb: float
        Lower bound associated with the transformed values.
    ub: float
        Lower bound associated with the transformed values.
    relax_binary_vars: bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    """

    def __init__(self, rhs, relax_binary_vars=False):
        self.rhs = rhs
        self.lb, self.ub = self.rhs.bound_propagation()
        self.relax_binary_vars = relax_binary_vars

    @cached_property
    def lhs(self):
        return LinearExpression.from_int(self.rhs.ndim)

    def update_bounds(self, lb, ub):
        pass

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.rhs, self.lhs))
