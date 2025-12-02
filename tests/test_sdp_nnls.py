import numpy as np

from algoverify.utils.range_utils import process_1d_ranges, process_2d_ranges


def test_range_functions():

    test_mat = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ])

    assert np.allclose(
        test_mat[process_2d_ranges([(1, 2)], [(2, 4)])],
        np.array([
            [7, 8]
        ])
    )

    assert np.allclose(
        test_mat[process_2d_ranges([(1, 3)], [(0, 2)])],
        np.array([
            [5, 6],
            [9, 10],
        ])
    )

    assert np.allclose(
        test_mat[process_2d_ranges([(0, 1), (2, 3)], [(1, 4)])],
        np.array([
            [2, 3, 4],
            [10, 11, 12],
        ])
    )

    assert np.allclose(
        test_mat[process_1d_ranges([(2, 4)])],
        np.array([
            [3, 4]
        ])
    )

    assert np.allclose(
        test_mat[process_1d_ranges([(0, 4)])],
        np.array([
            [1, 2, 3, 4]
        ])
    )

def test_nnls_sdp():
    pass
