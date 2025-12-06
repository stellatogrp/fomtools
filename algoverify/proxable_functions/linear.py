from algoverify.proxable_functions import ProxableFunction


class Linear(ProxableFunction):
    def __init__(self, c: float):
        self.c = c

    def prox(self, x, gamma=1):
        pass
