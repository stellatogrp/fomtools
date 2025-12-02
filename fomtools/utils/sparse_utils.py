import numpy as np
import scipy.sparse as spa


def efficient_csc_pad(matrix, new_shape):
    """
    Efficiently increases the size of a CSC matrix without copying
    the main data.

    The new_shape must be larger than or equal to the current shape.
    """
    M_old, N_old = matrix.shape
    M_new, N_new = new_shape

    assert M_new >= M_old and N_new >= N_old, "New shape must be >= old shape"

    num_new_cols = N_new - N_old
    nnz = matrix.indptr[-1]

    new_indptr_tail = np.full(num_new_cols, nnz, dtype=matrix.indptr.dtype)
    new_indptr = np.concatenate((matrix.indptr, new_indptr_tail))

    return spa.csc_matrix((matrix.data, matrix.indices, new_indptr),
                           shape=new_shape)
