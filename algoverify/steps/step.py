from algoverify.utils.steps_utils import linear_bound_propagation


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

    def __init__(self, lhs, rhs, bounds=None, relax_binary_vars=False):
        self.lhs = lhs
        self.rhs = rhs
        if bounds:
            self.lb, self.ub = linear_bound_propagation(expression=self.rhs, bounds=bounds)
        self.bounds = bounds
        self.relax_binary_vars = relax_binary_vars

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.rhs, self.lhs))
