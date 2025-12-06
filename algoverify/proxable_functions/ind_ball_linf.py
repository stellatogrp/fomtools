from algoverify.proxable_functions import ProxableFunction


class IndBallLinf(ProxableFunction):
    def __init__(self, r: float):
        self.r = r

    def prox(self, x, gamma=1):
        pass
