import numpy as np
from dataclasses import dataclass
from algoverify.linearexpression import LinearExpression


class Step(object):
    def __init__(self, lhs_expr, rhs_expr, lb=-np.inf, ub=np.inf, relax_binary_vars=False):
        self.lhs = lhs_expr
        self.rhs = rhs_expr
        self.lb = lb
        self.ub = ub
        self.relax_binary_vars = relax_binary_vars
