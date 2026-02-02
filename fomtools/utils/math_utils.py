import numpy as np


def norm(x, norm_type):
    match norm_type:
        case 1:
            return np.sum(np.abs(x))

        case 2:
            return np.sqrt(np.sum(np.power(x, 2)))
