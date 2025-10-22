from dataclasses import dataclass, field

from algoverify.steps.step import Step


@dataclass(frozen=True, kw_only=True)
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
