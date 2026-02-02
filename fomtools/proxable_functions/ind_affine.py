from fomtools.proxable_functions import ProxableFunction


class IndAffine(ProxableFunction):
    def __init__(self, iterative: bool = False):
        self.iterative = iterative

    def prox(self, x, gamma=1):
        pass

    def grad(self):
        raise NotImplementedError
