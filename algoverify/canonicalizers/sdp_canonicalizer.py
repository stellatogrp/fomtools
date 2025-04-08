from algoverify.canonicalizers.canonicalizer import Canonicalizer


class SDPCanonicalizer(Canonicalizer):

    def __init__(self, solver_params={}):
        self.solver_params = solver_params
