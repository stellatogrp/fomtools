import numpy as np
from dataclasses import dataclass
from algoverify.linearexpression import LinearExpression


class Step(object):
    def __init__(self, lhs_expr, rhs_expr, relax_binary_vars=False):
        self.lhs = lhs_expr
        self.rhs = rhs_expr
        self.lb = -np.inf
        self.ub = np.inf
        self.relax_binary_vars = relax_binary_vars

    def update_lb(self, new_rhs_lb):
        self.lb = np.maximum(self.lb, new_rhs_lb)

    def update_ub(self, new_rhs_ub):
        self.ub = np.minimum(self.ub, new_rhs_ub)
