import numpy as np

from algoverify.proxable_functions import ProxableFunction


class IndHalfSpace(ProxableFunction):
    def __init__(self, a: np.ndarray, b: int, norm_a: float):
        self.a = a
        self.b = b
        self.norm_a = norm_a

    def prox(self, x, gamma=1):
        pass
