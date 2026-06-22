import numpy as np
import pytest

from fomtools.utils.sparse_utils import get_submatrix_selection_matrix

# --- Helper Utilities for Testing ---


def get_triu_entries_col_major(n):
    """
    Helper to generate raw upper triangular indices in Column-Major order.
    Returns list of tuples [(0,0), (0,1), (1,1), (0,2)...]
    """
    entries = []
    for c in range(n):
        for r in range(c + 1):
            entries.append((r, c))
    return entries


def manual_vec(Matrix, scale=np.sqrt(2)):
    """
    Manually vectorizes a full numpy matrix into the Clarabel format
    to serve as Ground Truth.
    """
    n = Matrix.shape[0]
    entries = []
    for c in range(n):
        for r in range(c + 1):
            val = Matrix[r, c]
            if r != c:
                val *= scale
            entries.append(val)
    return np.array(entries)


@pytest.fixture
def unique_data_n5():
    """
    Creates a 5x5 Symmetric Matrix where every upper-triangular entry
    is a unique integer, making it easy to trace values.
    """
    n = 5
    # Create vector of unique values 100, 101, ...
    # Size = 5*6/2 = 15
    raw_vals = np.arange(100, 115, dtype=float)

    # Fill matrix to be Symmetric
    X = np.zeros((n, n))
    k = 0
    # Fill in Column-Major order to match our raw_vals vector layout
    for c in range(n):
        for r in range(c + 1):
            X[r, c] = raw_vals[k]
            X[c, r] = raw_vals[k]  # Symmetry
            k += 1

    # The 'x' input vector (unscaled decision variables)
    # This represents the raw entries of X before any PSD scaling
    x_input = raw_vals

    return n, X, x_input


# --- Unit Tests ---


def test_full_matrix_vectorization(unique_data_n5):
    """
    Test Case: Selecting all indices [0, 1, 2, 3, 4]
    Expected: The operator M should act as the standard 'vec' operator.
              It should scale off-diagonals by sqrt(2) and leave diagonals as is.
    """
    n, X_full, x_input = unique_data_n5
    indices = [0, 1, 2, 3, 4]

    # 1. Get the operator
    M = get_submatrix_selection_matrix(n, indices)

    # 2. Apply operator
    x_output = M @ x_input

    # 3. Calculate Ground Truth manually
    #    Apply standard vec(.) to the full matrix
    x_ground_truth = manual_vec(X_full, scale=np.sqrt(2))

    np.testing.assert_allclose(
        x_output, x_ground_truth, err_msg="Full matrix vectorization failed to match ground truth."
    )


def test_submatrix_selection(unique_data_n5):
    """
    Test Case: Selecting a subset of indices [1, 3]
    Expected: M extracts the 2x2 submatrix corresponding to rows/cols 1 and 3.
              X_sub = [[X11, X13],
                       [X31, X33]]
    """
    n, X_full, x_input = unique_data_n5
    indices = [1, 3]

    # 1. Get operator
    M = get_submatrix_selection_matrix(n, indices)

    # 2. Apply operator
    x_output = M @ x_input

    # 3. Calculate Ground Truth
    #    Manually slice the matrix first
    X_sub = X_full[np.ix_(indices, indices)]
    #    Then vectorize the small matrix
    x_ground_truth = manual_vec(X_sub, scale=np.sqrt(2))

    np.testing.assert_allclose(x_output, x_ground_truth, err_msg="Submatrix selection failed.")

    # Check dimensions
    # Output size should be r*(r+1)/2 = 2*3/2 = 3
    assert x_output.shape[0] == 3


def test_arbitrary_scaling_constant(unique_data_n5):
    """
    Test Case: Verify the 'scale' parameter works (e.g., using 2.0 or 10.0)
    """
    n, X_full, x_input = unique_data_n5
    indices = [1, 2, 4]
    custom_scale = 10.0

    M = get_submatrix_selection_matrix(n, indices, scale=custom_scale)
    x_output = M @ x_input

    X_sub = X_full[np.ix_(indices, indices)]
    x_ground_truth = manual_vec(X_sub, scale=custom_scale)

    np.testing.assert_allclose(
        x_output, x_ground_truth, err_msg="Custom scaling constant was not applied correctly."
    )


def test_unsorted_indices(unique_data_n5):
    """
    Test Case: Selecting indices out of order [3, 1]
    Expected: The output vector must represent the submatrix defined by order [3, 1].
              X_sub = [[X33, X31],
                       [X13, X11]]
              Note that the vectorization of X_sub must follow the order of the *submatrix*
              dimensions, i.e., col 0 (which is 3), then col 1 (which is 1).
    """
    n, X_full, x_input = unique_data_n5
    indices = [3, 1]

    M = get_submatrix_selection_matrix(n, indices)
    x_output = M @ x_input

    # Ground Truth: Construct the submatrix in the specific order [3, 1]
    # Row 0 -> global 3, Row 1 -> global 1
    # Col 0 -> global 3, Col 1 -> global 1
    X_sub = X_full[np.ix_(indices, indices)]

    # X_sub is now:
    # [[X33, X31],
    #  [X13, X11]]
    # Since X is symmetric, X31 == X13.
    # The output vector should correspond to vec(X_sub).
    x_ground_truth = manual_vec(X_sub)

    np.testing.assert_allclose(
        x_output, x_ground_truth, err_msg="Unsorted indices were not handled correctly."
    )


def test_unsorted_indices_logic_check():
    """
    Specific logic check for unsorted indices to verify the matrix entries
    are mapping correctly without relying on the helper.
    """
    n = 3
    # X = [[0, 1, 2],
    #      [1, 3, 4],
    #      [2, 4, 5]]
    #
    # x_in (col-major upper) = [0, 1, 3, 2, 4, 5]
    #                           00 01 11 02 12 22
    x_in = np.array([0.0, 1.0, 3.0, 2.0, 4.0, 5.0])

    # Submatrix indices [2, 0]
    # Expected Submatrix:
    # [[X22, X20],
    #  [X02, X00]]
    # = [[5, 2],
    #    [2, 0]]
    #
    # Expected Vector (Col-Major Upper of Submatrix):
    # 1. (0,0) -> X22 -> 5
    # 2. (0,1) -> X20 -> 2 (Scaled)
    # 3. (1,1) -> X00 -> 0
    #
    # Result = [5, 2*sqrt(2), 0]

    indices = [2, 0]
    M = get_submatrix_selection_matrix(n, indices, scale=np.sqrt(2))
    res = M @ x_in

    expected = np.array([5.0, 2.0 * np.sqrt(2), 0.0])

    np.testing.assert_allclose(res, expected)
