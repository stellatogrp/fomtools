from algoverify.proxable_functions import ProxableFunction


class IndSimplex(ProxableFunction):
    def __init__(self, a: float = 1):
        self.a = a

    def prox(self, x, gamma=1):
        pass
