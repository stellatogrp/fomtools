from fomtools.proxable_functions import ProxableFunction


class HingeLoss(ProxableFunction):
    def __init__(self, mu):
        self.mu = mu

    def prox(self, x, gamma=1):
        pass
