import numpy as np

from fomtools.proxable_functions import ProxableFunction


class IndPoint(ProxableFunction):
    def __init__(self, p: float | np.ndarray):
        self.p = p

    def prox(self, x, gamma=1):
        pass
