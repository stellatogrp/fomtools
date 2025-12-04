import numpy as np

from algoverify.proxable_functions import ProxableFunction


class IndPolyhedral(ProxableFunction):
    def __init__(self, A: np.ndarray, l: float, u: float, xmin: float, xmax: float):
        self.A = A
        self.l = l
        self.u = u
        self.xmin = xmin
        self.xmax = xmax

    def prox(self, x, gamma):
        pass
