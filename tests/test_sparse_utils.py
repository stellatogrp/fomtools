import numpy as np
import scipy.sparse as spa

from algoverify.utils.sparse_utils import efficient_csc_pad


def test_sparse_manipulations():
    A_orig = spa.csc_matrix([[1, 0, 2],
                             [0, 0, 0],
                             [4, 5, 0]])

    new_shape = (5, 6)
    A_padded = efficient_csc_pad(A_orig, new_shape)

    assert np.allclose(A_padded.todense(),
        np.array([
            [1, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [4, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
    )
