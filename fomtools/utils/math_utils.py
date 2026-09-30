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


def repair_crossed_bounds(lb, ub) -> tuple[np.ndarray, np.ndarray]:
    """
    Repair elementwise crossed bounds ``lb > ub`` by swapping them.

    LP solvers (e.g. in OBBT) can return a lower bound a hair above the
    upper bound (~1e-17) on a coordinate whose feasible range is a single
    point. Such a crossing makes ``LinearExpression.bound_propagation`` raise
    downstream. Returning ``(min(lb, ub), max(lb, ub))`` gives an interval
    that covers both values, so it remains a valid (if marginally looser)
    bound. Callers should reject large crossings before repairing.

    Parameters
    ----------
    lb, ub : np.ndarray or float
        Lower and upper bounds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The repaired ``(lb, ub)`` with ``lb <= ub`` elementwise.
    """
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return np.minimum(lb, ub), np.maximum(lb, ub)
