import numpy as np

from algoverify import LinearExpression
from algoverify.steps.step import Step


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
        \quad \text{with } y = (v, x).
    $$

    where $x, c \in \mathbb{R}$.

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
    l: np.ndarray[float] | LinearExpression
        The lower saturation bound of the saturated linear operator.
        Can be either a numeric value or a LinearExpression object whose
        lower boudnds acts as the lower saturation bound.
    u: np.ndarray[float] | LinearExpression
        The lower saturation bound of the saturated linear operator.
        Can be either a numeric value or a LinearExpression object whose
        lower boudnds acts as the upper saturation bound.
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
        Update the bounds of the left-land side after applying
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
        self.lhs.update_bounds(lb=self.satlin(lb), ub=self.satlin(ub))

    def get_constraints(self):
        r"""
        Construct the convex-hull linear constraints corresponding to
        the saturated linear unit operator.

        Given a saturated linear unit operator, defined as:

        $$
            (\phi(v, b))_i = S_{[b, c]}(a^T y) =
            \begin{cases}
            b & a^T y < b, \\\\
            a^T y & b \leq a^T y \leq c, \\\\
            c & a^T y > c,
            \end{cases}
            \quad \text{with } y = (v, b), \text{and }
             b, c \in \mathbb{R}.
        $$

        This method derives the convex hull of the region defined by:

        $$
            \Phi = \{ (y, w) \in [\underline{y}, \bar{y}] \times [b, c] \mid w =
            \mathcal{S}_{[b,c]}(a^\top y) \}.
        $$

        and expresses it as a set of Constraint objects. The convex hull of the
        region is:

        $$
            \operatorname{conv}(\Phi) =
            \left\{
            (y, w) \in [\underline{y}, \bar{y}] \times [b, c]
            \;\middle|\;
            \begin{cases}
            w = b & u_J < b, \\[4pt]
            w = c & \ell_J > c, \\[4pt]
            w = a^\top y & b \le \ell_J \le u_J \le c, \\[4pt]
            (y, w) \in Q & \text{otherwise}
            \end{cases}
            \right\},
        $$

        with

        $$
            Q =
            \left\{
            (y, w) \in \mathbb{R}^{p + d + 1}
            \;\middle|\;
            \begin{aligned}
            & b \le w \le c, \\[4pt]
            & \frac{b - c}{b - u_J}(a^\top y - u_J) + c
            \;\le\;
            w
            \;\le\;
            \frac{c - b}{c - \ell_J}(a^\top y - \ell_J) + b, \\[6pt]
            & w \le \sum_{i \in I} a_i (y_i - \ell^0_i)
            + \frac{\ell_I - b}{u^0_o - \ell^0_o}(y_o - \ell^0_o) + b,
            \qquad \forall (I, o) \in \mathcal{I}^+, \\[6pt]
            & w \ge \sum_{i \in I} a_i (y_i - u^0_i)
            + \frac{u_I - c}{\ell^0_o - u^0_o}(y_o - u^0_o) + c,
            \qquad \forall (I, o) \in \mathcal{I}^-
            \end{aligned}
            \right\}.
        $$

        where:

        - vector $l^{0} \in \mathbb{R}^{d}$ with entries $l_i^{0} = \underline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \overline{y}_i$ otherwise,
        - vector $u^{0} \in \mathbb{R}^{d}$ with entries $u_i^{0} = \overline{y}_i \text{ if }
        a_i \geq 0 \text{ and } \underline{y}_i$ otherwise,
        - Given $\textbf{I} \subseteq \textbf{J} = \left\{1, ..., n\right\}$, the lower bound is
        defined as $l_I = \sum_{i \in I}a_il_i^{0} + \sum_{i \notin I}a_iu_i^{0}$ and the
        upper bounds $u_I = \sum_{i \in I}a_iu_i^{0} + \sum_{i \notin I}a_il_i^{0}$,
        - $\mathcal{I}^+ = \{ (I, o) \in 2^J \times J \mid \ell_I \ge b,\; \ell_{I \cup \{o\}}
        < b,\; w_I \neq 0 \},$
        - $\mathcal{I}^- = \{ (I, o) \in 2^J \times J \mid u_I \le c,\; u_{I \cup \{o\}} >
        c,\; w_I \neq 0 \}.$

        Returns
        -------
        List[Constraint]
            A list of constraint objects describing the convex hull of the
            feasible region defined by the saturated linear unit activation.

        """
        constraints = []

        for l, u, lb, ub, lhs, rhs in zip(self.l, self.u, self.lb, self.ub, self.lhs, self.rhs):
            try:
                float(l), float(u)
            except TypeError:
                return constraints

            if lb >= u:
                constraints.append(lhs == u)
            elif ub <= l:
                constraints.append(lhs == l)
            elif lb >= l and ub <= u:
                constraints.append(lhs == rhs)
            else:
                if not self.relax_binary_vars:
                    w1 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(rhs.ndim, is_binary=True)

                    if lb < l:
                        constraints.extend(
                            [
                                lhs <= rhs + (ub - lb) * w2,
                                rhs >= l + (lb - l) * w2,
                                rhs <= l + (ub - l) * (1 - w2),
                            ]
                        )

                        if ub > u:
                            constraints.extend(
                                [
                                    lhs >= u - (u - l) * (1 - w1),
                                    lhs <= l + (u - l) * (1 - w2),
                                    lhs >= l + (lb - l) * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                    if ub > u:
                        constraints.extend(
                            [
                                lhs >= rhs - (ub - lb) * w1,
                                rhs >= u + (lb - u) * (1 - w1),
                                rhs <= u + (ub - u) * w1,
                            ]
                        )
                        if lb > l:
                            constraints.append(lhs >= u - (u - lb) * (1 - w1))

                if lb < l and ub > u:
                    expr = (u - l) / (u - lb) * (rhs - u) + u
                    constraints.extend([lhs <= expr, lhs >= expr])

                elif lb < l and ub < u:
                    constraints.extend(
                        [lhs >= rhs, lhs <= ((ub - l) / (ub - lb)) * (rhs - ub) + ub]
                    )

                elif lb > l and ub > u:
                    constraints.extend(
                        [lhs <= rhs, lhs >= ((u - lb) / (ub - lb) * (rhs - lb) + lb)]
                    )

        return constraints
