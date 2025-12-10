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

    assert M_new >= M_old and N_new >= N_old, \
        f"New shape {(M_new, N_new)} must be >= old shape {(M_old, N_old)}"

    if M_new == M_old and N_new == N_old:
        return matrix

    num_new_cols = N_new - N_old
    nnz = matrix.indptr[-1]

    new_indptr_tail = np.full(num_new_cols, nnz, dtype=matrix.indptr.dtype)
    new_indptr = np.concatenate((matrix.indptr, new_indptr_tail))

    return spa.csc_matrix((matrix.data, matrix.indices, new_indptr),
                           shape=new_shape)


def get_submatrix_selection_matrix(n, indices, scale=np.sqrt(2)):
    """
    Vectorized construction of a selection matrix using tril_indices swap trick.
    Produces Column-Major Upper Triangular format (Clarabel standard).
    """
    indices = np.array(indices)
    r = len(indices)

    # 1. Generate local coordinates using the swap trick.
    #    np.tril_indices returns (rows, cols) for the lower triangle.
    #    We assign them to (c_sub, r_sub) respectively.
    #    This implicitly transposes the traversal order to Column-Major Upper.
    c_sub_raw, r_sub_raw = np.tril_indices(r)

    # 2. Map to Global Coordinates
    r_glob = indices[r_sub_raw]
    c_glob = indices[c_sub_raw]

    # 3. Enforce Upper Triangularity in Global Space
    #    (Same logic as before to handle unsorted input indices)
    r_final = np.minimum(r_glob, c_glob)
    c_final = np.maximum(r_glob, c_glob)

    # 4. Calculate Linear Index for FULL Matrix (Clarabel/SCS Format)
    #    Formula: k = j*(j+1)/2 + i
    col_indices_in_A = (c_final * (c_final + 1)) // 2 + r_final

    # 5. Calculate Linear Index for SUB Matrix (Output Row Indices)
    #    Since tril_indices produces the exact order we need,
    #    the rows are just sequential integers.
    row_indices_in_A = np.arange(len(r_sub_raw))

    # 6. Apply Scaling
    #    Scale applies if the element is off-diagonal
    data = np.where(r_sub_raw == c_sub_raw, 1.0, scale)

    # 7. Construct Sparse Matrix
    dim_full = n * (n + 1) // 2
    dim_sub  = r * (r + 1) // 2

    A_sel = spa.coo_matrix((data, (row_indices_in_A, col_indices_in_A)),
                           shape=(dim_sub, dim_full))

    return A_sel.tocsc()
