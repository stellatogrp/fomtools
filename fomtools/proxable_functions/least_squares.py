from fomtools.proxable_functions import ProxableFunction


class LeastSquares(ProxableFunction):
    def __init__(self, lambd: float = 1, iterative: bool = False):
        self.lambd = lambd
        self.iterative = iterative

    def prox(self, x, gamma=1):
        pass
