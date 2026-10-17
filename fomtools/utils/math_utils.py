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


def repair_crossed_bounds(
    lb, ub, tol: float = 1e-7, buffer: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Repair elementwise crossed bounds ``lb > ub`` by swapping them.

    LP solvers (e.g. in OBBT) can return a lower bound a hair above the
    upper bound (~1e-17) on a coordinate whose feasible range is a single
    point. Such a crossing makes ``LinearExpression.bound_propagation`` raise
    downstream. Returning ``(min(lb, ub), max(lb, ub))`` gives an interval
    that covers both values, so it remains a valid (if marginally looser)
    bound. Crossings larger than ``tol`` (relative to the bound magnitude)
    indicate a genuinely empty interval and raise instead of being hidden.

    Parameters
    ----------
    lb, ub : np.ndarray or float
        Lower and upper bounds.
    tol : float, default 1e-7
        Largest crossing ``lb - ub`` treated as solver noise, scaled by
        ``max(1, |lb|, |ub|)``.
    buffer : float, default 0.0
        Relative outward padding, scaled like ``tol``, applied to the repaired
        (crossed) entries only. A crossing within an LP solver's tolerance means
        each computed bound is only accurate to that tolerance, so widening the
        swapped interval by it keeps the bound valid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The repaired ``(lb, ub)`` with ``lb <= ub`` elementwise.

    Raises
    ------
    ValueError
        If any crossing exceeds the tolerance.
    """
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    with np.errstate(invalid="ignore"):
        scale = np.maximum(1.0, np.maximum(np.abs(lb), np.abs(ub)))
        too_crossed = lb - ub > tol * scale
    if np.any(too_crossed):
        raise ValueError(
            f"Bounds cross by more than tol={tol}: max(lb - ub) = {np.max(lb - ub):.3e}"
        )
    lo, hi = np.minimum(lb, ub), np.maximum(lb, ub)
    if buffer > 0:
        pad = np.where(lb > ub, buffer * scale, 0.0)
        lo, hi = lo - pad, hi + pad
    return lo, hi
