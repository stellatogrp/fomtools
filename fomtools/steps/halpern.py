import numpy as np


class HalpernStep:
    """
    Halpern iteration step allowing to chose dynamic or not.

    The Halpern iteration is defined as:
        z^{k+1} = γ_k * T(z^k) + (1-γ_k) * a^k
    where γ_k = (m+1)/(m+2) and m is the number of iterations since the last restart.

    Dynamic restart condition:
        ||z^k - T(z^k)||_∞ ≤ (1/ε) * ||a^k - T(a^k)||_∞  ⟹  reset m ← 0
        else m ← min(m+1, K)

    Attributes
    ----------
    lhs : LinearExpression
        Output iterate z^{k+1}.
    z_k_expr : LinearExpression
        Current iterate z^k.
    t_zk_expr : LinearExpression
        Operator applied to current iterate T(z^k).
    anchor_expr : LinearExpression
        Current anchor a^k.
    t_anchor_expr : LinearExpression
        Operator applied to current anchor T(a^k).
    K : int
        Maximum number of iterations for step-size schedule.
    eps : float
        Restart threshold parameter ε.
    sequence_id : str
        Identifier to chain Halpern y-state across iterations.
    dynamic : bool
        Whether to use dynamic restart logic.
    big_M : float, optional
        Big-M constant for MILP formulations.
    a_next_expr : LinearExpression, optional
        Next anchor a^{k+1}.
    relax_binary_vars : bool
        If True, binary variables are relaxed to continuous.
    alpha_array : np.ndarray
        Step size array: α_m = (m + 1) / (m + 2) for m = 0..K.
    """

    def __init__(
        self,
        lhs,
        z_k_expr,
        t_zk_expr,
        anchor_expr,
        t_anchor_expr,
        K,
        eps=np.e,
        sequence_id="_default",
        dynamic=True,
        big_M=None,
        a_next_expr=None,
        relax_binary_vars=False,
    ):
        self.lhs = lhs
        self.z_k_expr = z_k_expr
        self.t_zk_expr = t_zk_expr
        self.anchor_expr = anchor_expr
        self.t_anchor_expr = t_anchor_expr
        self.K = K
        self.eps = eps
        self.sequence_id = sequence_id
        self.dynamic = dynamic
        self.big_M = big_M
        self.a_next_expr = a_next_expr
        self.relax_binary_vars = relax_binary_vars
        self.alpha_array = np.array([(m + 1) / (m + 2) for m in range(K + 1)])

    def __hash__(self):
        return hash((id(self.lhs), id(self.z_k_expr), id(self.anchor_expr)))
