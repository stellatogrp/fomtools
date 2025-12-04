from algoverify.proxable_functions import ProxableFunction


class IndBallL1(ProxableFunction):
    def __init__(self, r: float):
        self.r = r

    def prox(self, x, gamma):
        pass
