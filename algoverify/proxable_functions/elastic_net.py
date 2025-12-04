from algoverify.proxable_functions import ProxableFunction
from algoverify.utils import norm


class ElasticNet(ProxableFunction):
    def __init__(self, mu, lambd):
        if mu < 0 or lambd < 0:
            raise ValueError("mu and lamb must be positive")

        self.mu = mu
        self.lambd = lambd

    def elastic_net(self, x):
        return self.mu * norm(x, 1) + (self.lambd / 2) * norm(x, 2) ** 2

    def prox(self, x, gamma):
        pass
