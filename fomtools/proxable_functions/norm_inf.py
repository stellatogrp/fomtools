from fomtools.proxable_functions import ProxableFunction


class NormInf(ProxableFunction):
    def __init__(self, float=1):
        self.float = float

    def prox(self, x, gamma=1):
        pass
