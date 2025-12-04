from algoverify.proxable_functions import ProxableFunction


class TotalVariation1D(ProxableFunction):
    def __init__(self, lambd: float = 1):
        self.lambd = lambd

    def prox(self, x, gamma):
        pass
