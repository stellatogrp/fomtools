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
    arg : LinearExpression
        Right-hand side (input) linear expression.
    output : LinearExpression
        Left-hand side (output) linear expression.
    lb : np.ndarray
        Lower bounds on the INPUT (arg), before saturation is applied.
    ub : np.ndarray
        Upper bounds on the INPUT (arg), before saturation is applied.
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
        arg: LinearExpression,
        l: LinearExpression | np.ndarray[float],
        u: LinearExpression | np.ndarray[float],
        relax_binary_vars: bool = False,
    ):
        super().__init__(arg, relax_binary_vars)
        self.l = l
        self.u = u
        self.update_bounds(*self.arg.bound_propagation())

    def __hash__(self):
        return hash((self.arg, self.output))

    def satlin_prop(self, x: np.ndarray, bound: str) -> np.ndarray:
        """
        Propagate a bound through the saturated linear operator.

        Since clip(x, l, u) is monotonically non-decreasing in all three
        arguments, the correct output bounds are:
            lower: clip(x_lb, l_lb, u_lb)
            upper: clip(x_ub, l_ub, u_ub)

        Parameters
        ----------
        x: np.ndarray
            A 1-D array of input bound values to propagate.
        bound: str
            Either "lb" or "ub", indicating which bound is being propagated.

        Returns
        -------
        np.ndarray
            The propagated bound on the output of the saturated linear operator.
        """
        try:
            l_bound = self.l.lb if bound == "lb" else self.l.ub
        except AttributeError:
            l_bound = self.l

        try:
            u_bound = self.u.lb if bound == "lb" else self.u.ub
        except AttributeError:
            u_bound = self.u
        return np.minimum(np.maximum(x, l_bound), u_bound)

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
        self.output.update_bounds(
            lb=self.satlin_prop(lb, "lb"),
            ub=self.satlin_prop(ub, "ub"),
        )
