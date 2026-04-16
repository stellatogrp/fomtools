import numpy as np


def norm(x, norm_type):
    """
    Compute the vector norm of the given type.

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    norm_type : int
        The type of norm to compute:
        - 1: L1 norm (sum of absolute values)
        - 2: L2 norm (Euclidean norm)

    Returns
    -------
    float
        The computed norm value.

    Examples
    --------
    >>> norm(np.array([3, 4]), 2)
    5.0
    >>> norm(np.array([1, -2, 3]), 1)
    6.0
    """
    match norm_type:
        case 1:
            return np.sum(np.abs(x))

        case 2:
            return np.sqrt(np.sum(np.power(x, 2)))
