import numpy as np


def process_2d_ranges(ranges1, ranges2):
    row_indices = []
    col_indices = []
    for r in ranges1:
        row_indices += list(range(r[0], r[1]))
    for r in ranges2:
        col_indices += list(range(r[0], r[1]))
    return np.ix_(row_indices, col_indices)

def process_1d_ranges(ranges, row=True):
    if row:
        return process_2d_ranges([(0, 1)], ranges)
    else:
        return process_2d_ranges(ranges, [(0, 1)])
