from dataclasses import dataclass

import numpy as np

from algoverify.linearexpression import LinearExpression


@dataclass(frozen=True)
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

    lhs: LinearExpression
    rhs: LinearExpression
    lb: float = -np.inf
    ub: float = np.inf
    relax_binary_vars: bool = False

    # TODO: figure out a better way to hash these objects
    def __hash__(self):
        return hash((self.rhs, self.lhs))
