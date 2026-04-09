import numpy as np

from fomtools import LinearExpression
from fomtools.steps.step import Step


class SaturatedLinear(Step):
    r"""
    Saturated Linear Unit step operator, defined as:

    $$
        (\phi(v, x))_i = S_{[x, c]}(a^T y) =
        \begin{cases}
            x & a^T y < x, \\\\
            a^T y & x \leq a^T y \leq c, \\\\
            c & a^T y > c,
        \end{cases}
        \quad \text{with } y = (v, x),
    $$

    where $x, c \in \mathbb{R}$.

    Attributes
    ----------
    rhs : LinearExpression
        Right-hand side (input) linear expression.
    lhs : LinearExpression
        Left-hand side (output) linear expression.
    lb : np.ndarray
        Lower bounds on the INPUT (rhs), before saturation is applied.
    ub : np.ndarray
        Upper bounds on the INPUT (rhs), before saturation is applied.
    relax_binary_vars : bool
        If true, binary decision variables associated with the operator
        are relaxed to take continuous values.
    l : np.ndarray[float] | LinearExpression
        The lower saturation bound of the saturated linear operator.
        Can be either a numeric value or a LinearExpression object whose
        lower bounds acts as the lower saturation bound.
    u : np.ndarray[float] | LinearExpression
        The upper saturation bound of the saturated linear operator.
        Can be either a numeric value or a LinearExpression object whose
        upper bounds acts as the upper saturation bound.
    """

    def __init__(
        self,
        rhs: LinearExpression,
        l: LinearExpression | np.ndarray[float],
        u: LinearExpression | np.ndarray[float],
        relax_binary_vars: bool = False,
    ):
        super().__init__(rhs, relax_binary_vars)
        self.l = l
        self.u = u
        self.update_bounds(*self.rhs.bound_propagation())

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    def satlin(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Saturated Linear Unit operator to the input
        arrray.

        Parameters
        ----------
        x: np.ndarray
            A 1-D array of values on which to apply the activation.

        Returns
        -------
        np.ndarray
            A new array representing saturated output, obtained by
            clipping `x` to the range [lower bound, upper bound].
        """
        try:
            return np.minimum(np.maximum(x, self.l.lb), self.u.ub)
        except AttributeError:
            return np.minimum(np.maximum(x, self.l), self.u)

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update the bounds of the left-hand side after applying
        Saturated Linear Unit activation.

        Parameters
        ----------
        lb: float or np.ndarray
            The new lower bound(s) before applying Saturated Linear
            Unit.

        ub: float or np.ndarray
            The new upper bound(s) before applying Saturated Linear
            Unit.

        """
        self.lb, self.ub = lb, ub
        self.lhs.update_bounds(lb=self.satlin(lb), ub=self.satlin(ub))
