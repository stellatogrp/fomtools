import numpy as np


def process_2d_ranges(ranges1, ranges2):
    """
    Convert range tuples to numpy index arrays for 2D slicing.

    Takes two lists of (start, end) range tuples and returns numpy
    index arrays suitable for advanced indexing of 2D arrays.

    Parameters
    ----------
    ranges1 : list[tuple[int, int]]
        List of (start, end) pairs for row indices.
    ranges2 : list[tuple[int, int]]
        List of (start, end) pairs for column indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of index arrays from ``np.ix_`` for 2D fancy indexing.

    Examples
    --------
    >>> rows, cols = process_2d_ranges([(0, 2)], [(1, 3)])
    >>> A[rows, cols]  # Selects A[0:2, 1:3] as a submatrix
    """
    row_indices = []
    col_indices = []
    for r in ranges1:
        row_indices += list(range(r[0], r[1]))
    for r in ranges2:
        col_indices += list(range(r[0], r[1]))
    return np.ix_(row_indices, col_indices)


def process_1d_ranges(ranges, row=True):
    """
    Convert ranges to a row or column indexer for 2D arrays.

    Creates an indexer for selecting either a row vector (index 0) or
    column vector (index 0) combined with the specified ranges.

    Parameters
    ----------
    ranges : list[tuple[int, int]]
        List of (start, end) pairs for the non-singleton dimension.
    row : bool, default=True
        If True, returns indexer for row 0 with given column ranges.
        If False, returns indexer for column 0 with given row ranges.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Index arrays from ``np.ix_`` for 2D fancy indexing.

    Examples
    --------
    >>> idx = process_1d_ranges([(1, 4)], row=True)
    >>> A[idx]  # Selects A[0, 1:4] as a row vector
    """
    if row:
        return process_2d_ranges([(0, 1)], ranges)
    else:
        return process_2d_ranges(ranges, [(0, 1)])
