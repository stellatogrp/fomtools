import numpy as np

from algoverify.linearexpression import LinearExpression
from algoverify.steps.step import Step


class SoftThreshold(Step):
    r"""
    A soft-thresholding operator for shrinkage, defined as:

    $$
        (\phi(v,x))_i=\mathcal{T}_{\lambda}(a^\top y)=
        \begin{cases}
        a^\top y-\lambda,& a^\top y>\lambda,\\[6pt]
        0,& |a^\top y|\le \lambda,\\[6pt]
        a^\top y+\lambda,& a^\top y<-\lambda,
        \end{cases}
        \quad \text{with } y=(v,x).
    $$

    where, $\mathcal{T}_{\lambda}$ is the soft-thresholding operator with constant $\lambda>0$,
    and $a\in\mathbf{R}^d$.

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
    lambd : float
        The threshold ($\lambda$) controlling shrinkage. Must be non-negative.
    idx_with_right_binary_vars : set[int]
        Indices tied to “right-branch” indicators in a MILP reformulation (e.g., $t>\lambda$).
    idx_with_left_binary_vars : set[int]
        Indices tied to “left-branch” indicators (e.g., $t<-\lambda$).
    """

    def __init__(self, rhs: LinearExpression, lambd: float, relax_binary_vars: bool = False):
        super().__init__(rhs, relax_binary_vars)
        self.lambd = lambd
        self.idx_with_right_binary_vars = set()
        self.idx_with_left_binary_vars = set()

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    def soft_threshold(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the soft-thresholding operator to the input array.

        Parameters
        ----------
        x: np.ndarray
            A 1-D array of values on which to apply the activation.

        Returns
        -------
        np.ndarray
            A new array representing obtained by applying the soft-thresholding
            operator element-wise. Values with magnitude less than `self.lambd`
            are mapped to zero, while others are shrunk toward zero by
            `self.lambd`.

        """
        return np.sign(x) * np.maximum(np.abs(x) - self.lambd, 0)

    def update_bounds(self, lb: float | np.ndarray, ub: float | np.ndarray):
        """
        Update the bounds of the left-land side after applying
        soft-thresholding operator.

        Parameters
        ----------
        lb: float or np.ndarray
            The new lower bound(s) before applying soft-thresholding.

        ub: float or np.ndarray
            The new upper bound(s) before applying soft-thresholding.
        """
        self.lhs.update_bounds(lb=self.soft_threshold(lb), ub=self.soft_threshold(ub))

    def get_constraints(self):
        r"""
        Construct the convex-hull linear constraints corresponding to the
        soft-thresholding operator.

        Given a soft-thresholding operator for shrinkage, defined as:

        $$
            (\phi(v,x))_i=\mathcal{T}_{\lambda}(a^\top y)=
            \begin{cases}
            a^\top y-\lambda,& a^\top y>\lambda,\\[6pt]
            0,& |a^\top y|\le \lambda,\\[6pt]
            a^\top y+\lambda,& a^\top y<-\lambda,
            \end{cases}
            \quad \text{with } y=(v,x).
        $$

        where, $\mathcal{T}_{\lambda}$ is the soft-thresholding operator with constant $\lambda>0$,
        and $a\in\mathbf{R}^d$.

        This method derives the convex hull of the region defined by:

        $$
            \Phi = \{ (y, w) \in [\underline{y}, \bar{y}] \times [b, c]
            \mid w = \mathcal{S}_{[b,c]}(a^\top y) \}.
        $$

        and expresses it as a set of Constraint objects. The convex hull of the
        region is:

        $$
            \operatorname{conv}(\Phi) =
            \left\{
            (y, w) \in [\underline{y}, \bar{y}] \times \mathbf{R}
            \;\middle|\;
            \begin{cases}
            w = a^\top y - \lambda & \ell_J > \lambda, \\[4pt]
            w = a^\top y  + \lambda & u_J < - \lambda, \\[4pt]
            w = 0 & - \lambda \le \ell_J \le u_J \le \lambda, \\[4pt]
            (y, w) \in Q & \text{otherwise}
            \end{cases}
            \right\},
        $$

        with

        $$
            Q = \left\{
            (y, w) \in \mathbb{R}^{p + d + 1}
            \;\middle|\;
            \begin{aligned}
            & a^\top y - \lambda \le w \le a^\top y + \lambda, \\[4pt]
            & \frac{l_J + \lambda}{l_J - \lambda}(a^\top y - \lambda)
            \;\le\;
            w
            \;\le\;
            \frac{u_J - \lambda}{u_J + \lambda}(a^\top y + \lambda), \\[6pt]
            & w \le \sum_{i \in I} a_i (y_i - \ell^0_i)
            + \frac{\ell_I - \lambda}{u^0_o - \ell^0_o}(y_o - \ell^0_o),
            \qquad \forall (I, o) \in \mathcal{I}^+, \\[6pt]
            & w \ge \sum_{i \in I} a_i (y_i - u^0_i)
            + \frac{u_I + \lambda}{u^0_o - u^0_o}(y_o - u^0_o),
            \qquad \forall (I, o) \in \mathcal{I}^- .
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
        - $\mathcal{I}^+ = \{ (I, o) \in 2^J \times J \mid \ell_I \ge \lambda,\; \ell_{I \cup \{o\}}
        < \lambda,\; w_I \neq 0 \},$
        -  $\mathcal{I}^- = \{ (I, o) \in 2^J \times J \mid u_I \le - \lambda,\; u_{I \cup \{o\}} >
        -\lambda,\; w_I \neq 0 \}.$
        - and $w_I$ is the slicing of vector $w$ over $I$.


        Returns
        -------
        List[Constraint]
            A list of constraint objects describing the convex hull of
            the feasible region defined by the soft-thresholding operator.
        """
        constraints = []

        i = 0
        for lb, ub, lhs_lb, lhs_ub, lhs, rhs in zip(
            self.lb, self.ub, self.lhs.lb, self.lhs.ub, self.lhs, self.rhs
        ):
            if lb >= self.lambd:
                constraints.append(lhs == rhs - self.lambd)
            elif ub <= -self.lambd:
                constraints.append(lhs == rhs + self.lambd)
            elif lb >= -self.lambd and ub <= self.lambd:
                constraints.append(lhs == 0)
            else:
                if not self.relax_binary_vars:
                    w1 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    w2 = LinearExpression.from_int(rhs.ndim, is_binary=True)
                    self.idx_with_right_binary_vars.add(i)

                    if ub > self.lambd:
                        self.idx_with_left_binary_vars.add(i)
                        constraints.extend(
                            [
                                rhs >= self.lambd + (lb - self.lambd) * (1 - w1),
                                rhs <= self.lambd + (ub - self.lambd) * w1,
                                lhs <= rhs - self.lambd + (2 * self.lambd) * (1 - w1),
                            ]
                        )

                        if lb < -self.lambd:
                            constraints.extend(
                                [
                                    lhs <= lhs_ub * w1,
                                    lhs >= lhs_lb * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                        if -self.lambd <= lb <= self.lambd:
                            constraints.append(lhs <= lhs_ub * w1)

                    if lb < -self.lambd:
                        self.idx_with_left_binary_vars.add(i)
                        constraints.extend(
                            [
                                rhs <= -self.lambd + (ub + self.lambd) * (1 - w2),
                                rhs >= -self.lambd + (lb + self.lambd) * w2,
                                lhs >= rhs + self.lambd + (-2 * self.lambd) * (1 - w2),
                            ]
                        )

                        if -self.lambd <= lb <= self.lambd:
                            constraints.append(lhs >= lhs_lb * w2)

                if lb < -self.lambd and ub > self.lambd:
                    constraints.extend(
                        [
                            lhs >= rhs - self.lambd,
                            lhs <= rhs + self.lambd,
                            lhs <= lhs_ub / (ub + self.lambd) * (rhs + self.lambd),
                            lhs >= lhs_lb / (lb - self.lambd) * (rhs - self.lambd),
                        ]
                    )

                elif -self.lambd <= lb <= self.lambd < ub:
                    constraints.extend(
                        [
                            lhs >= 0,
                            lhs <= lhs_ub / (ub - lb) * (rhs - lb),
                            lhs >= rhs - self.lambd,
                        ]
                    )

                elif self.lambd >= ub >= -self.lambd >= lb:
                    constraints.extend(
                        [
                            lhs <= 0,
                            lhs >= lhs_lb / (lb - ub) * (rhs - ub),
                            lhs <= rhs + self.lambd,
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

            i += 1

        return constraints
