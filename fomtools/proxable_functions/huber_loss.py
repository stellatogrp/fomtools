from fomtools.proxable_functions import ProxableFunction


class HuberLoss(ProxableFunction):
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho

    def prox(self, x, gamma=1):
        pass
