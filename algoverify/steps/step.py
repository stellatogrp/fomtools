import numpy as np
from dataclasses import dataclass
from algoverify.linearexpression import LinearExpression


@dataclass(frozen=True)
class Step:
    """
    A base class for thresholding and activation steps. This class
    defines a common interface for step-like operators that transform
    inputs element wise.

    Attributes:
        lhs: LinearExpression
            ...

        rhs: LinearExpression
            ...

        lb: float
            ...

        ub: float
            ...

        relax_binary_vars: bool
                If true,
    """

    lhs: LinearExpression
    rhs: LinearExpression
    lb: float = -np.inf
    ub: float = np.inf
    relax_binary_vars: bool = False

    def __hash__(self):
        return hash((self.rhs, self.lhs))
