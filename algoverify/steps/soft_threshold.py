from dataclasses import dataclass, field

from algoverify.steps.step import Step
from algoverify.linearexpression import LinearExpression
import numpy as np

from algoverify.utils.steps_utils import linear_bound_propagation


@dataclass(frozen=False, kw_only=True)
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

    lambd: float
    idx_with_right_binary_vars: set[int] = field(default_factory=set)
    idx_with_left_binary_vars: set[int] = field(default_factory=set)

    def __hash__(self):
        return hash((self.rhs, self.lhs))

    def __post_init__(self):
        self.lb, self.ub = linear_bound_propagation(expression=self.rhs, bounds=self.bounds)

    def soft_threshold(self, x):
        return np.sign(x) * np.maximum(np.abs(x) - self.lambd, 0)

    def new_bound(self, x):
        return np.sign(x) * np.maximum(np.abs(x) - self.lambd, 0)

    def get_constraints(self):
        constraints = []

        lhs_lb = self.soft_threshold(self.lb)
        lhs_ub = self.soft_threshold(self.ub)
        i = 0
        for lb, ub, _lb, _ub, lhs, rhs in zip(self.lb, self.ub, lhs_lb, lhs_ub, self.lhs, self.rhs):
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
                                    lhs <= _ub * w1,
                                    lhs >= _lb * w2,
                                    w1 + w2 <= 1,
                                ]
                            )

                        if -self.lambd <= lb <= self.lambd:
                            constraints.append(lhs <= _ub * w1)

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
                            constraints.append(lhs >= _lb * w2)

                if lb < -self.lambd and ub > self.lambd:
                    constraints.extend(
                        [
                            lhs >= rhs - self.lambd,
                            lhs <= rhs + self.lambd,
                            lhs <= _ub / (ub + self.lambd) * (rhs + self.lambd),
                            lhs >= _lb / (lb - self.lambd) * (rhs - self.lambd),
                        ]
                    )

                elif -self.lambd <= lb <= self.lambd < ub:
                    constraints.extend(
                        [
                            lhs >= 0,
                            lhs <= _ub / (ub - lb) * (rhs - lb),
                            lhs >= rhs - self.lambd,
                        ]
                    )

                elif self.lambd >= ub >= -self.lambd >= lb:
                    constraints.extend(
                        [
                            lhs <= 0,
                            lhs >= _lb / (lb - ub) * (rhs - ub),
                            lhs <= rhs + self.lambd,
                        ]
                    )

                else:
                    raise RuntimeError("Unreachable code")

            i += 1

        return constraints
