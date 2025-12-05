import numpy as np

from algoverify.linearexpression import LinearExpression
from algoverify.proxable_functions import ProxableFunction


class Quadratic(ProxableFunction):
    def __init__(self, Q: np.ndarray, q: LinearExpression, iterative: bool = False):
        self.Q = Q
        self.q = q
        self.iterative = iterative

    def prox(self, x, gamma=1):
        pass
