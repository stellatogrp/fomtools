import clarabel
import numpy as np
import scipy.sparse as spa

from fomtools.solvers.sdp_solver import SDPSolver
from fomtools.utils.sparse_utils import get_submatrix_selection_matrix


class ClarabelSolver(SDPSolver):
    """
    Solver wrapper for the Clarabel conic optimization solver.

    This class provides an interface to solve semidefinite programs (SDPs) using
    the Clarabel solver. It converts the SDP problem data (equality constraints,
    inequality constraints, and PSD cone constraints) into Clarabel's native format
    and invokes the solver.

    The SDP is formulated as:
        maximize    tr(C @ X)
        subject to  tr(Aeq[i] @ X) = beq[i]      for all i
                    tr(Aineq[j] @ X) <= bineq[j]  for all j
                    X[cone_indices] >> 0          for each PSD cone

    Attributes
    ----------
    q : np.ndarray
        Vectorized objective matrix.
    Aeq : scipy.sparse.csc_matrix
        Stacked equality constraint matrices (vectorized).
    beq : np.ndarray
        Equality constraint right-hand sides.
    Aineq : scipy.sparse.csc_matrix
        Stacked inequality constraint matrices (vectorized).
    bineq : np.ndarray
        Inequality constraint right-hand sides.
    M : scipy.sparse.csc_matrix
        PSD cone selection matrices.
    cone_dims : dict
        Cone dimensions with keys 'zero', 'nonneg', 'psd'.
    """

    def __init__(self, solver_params=None):
        """
        Initialize the Clarabel solver.

        Parameters
        ----------
        solver_params : dict, optional
            Dictionary of solver-specific parameters to configure.
        """
        if solver_params is not None:
            self.set_solver_params(solver_params)

    def setup_clarabel_data(
        self,
        mat_problem_dim,
        C,
        Aeq_mats,
        beq_consts,
        Aineq_mats,
        bineq_consts,
        psd_cone_indices,
    ):
        """
        Convert SDP problem data to Clarabel's native format.

        This method transforms the matrix-form SDP specification into the
        vectorized conic form expected by Clarabel. Symmetric matrices are
        vectorized using column-major upper triangular format with appropriate
        scaling for off-diagonal elements.

        Parameters
        ----------
        mat_problem_dim : int
            Dimension of the symmetric matrix variable X.
        C : scipy.sparse.csc_matrix
            Objective matrix for maximizing tr(C @ X).
        Aeq_mats : list[scipy.sparse.csc_matrix]
            List of equality constraint matrices A_i for tr(A_i @ X) = b_i.
        beq_consts : list[float]
            Right-hand sides for equality constraints.
        Aineq_mats : list[scipy.sparse.csc_matrix]
            List of inequality constraint matrices for tr(A_j @ X) <= b_j.
        bineq_consts : list[float]
            Right-hand sides for inequality constraints.
        psd_cone_indices : list[set]
            List of index sets defining PSD cone constraints on submatrices.
        """
        vec_problem_dim = mat_problem_dim * (mat_problem_dim + 1) // 2

        q = -sparse_symmetric_vectorize(C, scale_factor=2)
        # TODO: see if this can be replaced by a sparse 1 by n vector instead
        q = (
            q.todense()
            .getA()
            .reshape(
                -1,
            )
        )
        assert q.shape[0] == vec_problem_dim

        symmetrized_Aeq_mats = [sparse_symmetric_vectorize(A, scale_factor=2) for A in Aeq_mats]
        Aeq = spa.vstack(symmetrized_Aeq_mats)
        beq = np.array(beq_consts)

        assert Aeq.shape == (beq.shape[0], vec_problem_dim)

        symmetrized_Aineq_mats = [sparse_symmetric_vectorize(A, scale_factor=2) for A in Aineq_mats]
        Aineq = spa.vstack(symmetrized_Aineq_mats)
        bineq = np.array(bineq_consts)

        assert Aineq.shape == (bineq.shape[0], vec_problem_dim)

        psd_cone_sizes = []
        psd_cone_selectors = []

        for cone in psd_cone_indices:
            n = len(cone)
            psd_cone_sizes.append(n)
            M = -get_submatrix_selection_matrix(mat_problem_dim, list(cone))
            psd_cone_selectors.append(M)

        M = spa.vstack(psd_cone_selectors)

        cone_dims = {
            "zero": Aeq.shape[0],
            "nonneg": Aineq.shape[0],
            "psd": psd_cone_sizes,
        }

        self.q = q
        self.Aeq = Aeq
        self.beq = beq
        self.Aineq = Aineq
        self.bineq = bineq
        self.M = M
        self.cone_dims = cone_dims

    def solve(self):
        """
        Solve the SDP and return the objective value.

        Assembles the conic problem from the prepared data and invokes
        the Clarabel solver. The problem is formulated as a minimization
        in Clarabel's standard form, so the returned value is negated
        to give the maximization objective.

        Returns
        -------
        float
            The optimal objective value of the SDP maximization problem.
        """
        q = self.q
        b_psd = [n * (n + 1) // 2 for n in self.cone_dims["psd"]]
        b = np.concatenate([self.beq, self.bineq, np.zeros(sum(b_psd))])
        A = spa.vstack([self.Aeq, self.Aineq, self.M])
        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == q.shape[0]

        cones = [
            clarabel.ZeroConeT(self.beq.shape[0]),
            clarabel.NonnegativeConeT(self.bineq.shape[0]),
        ]
        cones += [clarabel.PSDTriangleConeT(n) for n in self.cone_dims["psd"]]

        xdim = q.shape[0]
        P = spa.csc_matrix((xdim, xdim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        sol = solver.solve()
        return -sol.obj_val


def sparse_symmetric_vectorize(P, scale_factor=np.sqrt(2)):
    """
    Vectorize a sparse symmetric matrix to Clarabel's conic format.

    Extracts the upper triangle of a symmetric matrix and maps it to a
    1D vector in column-major upper triangular format, which is the
    standard vectorization used by Clarabel and SCS for PSD cones.

    Off-diagonal elements are scaled to preserve the inner product:
    tr(A @ B) = vec(A)^T @ vec(B) when using scaled vectorization.

    Parameters
    ----------
    P : scipy.sparse matrix
        A sparse symmetric matrix to vectorize. Only the upper triangle
        is used; the lower triangle is ignored.
    scale_factor : float, default=sqrt(2)
        Scaling factor applied to off-diagonal elements. Use sqrt(2) for
        inner product preservation, or 2 for constraint coefficient matrices.

    Returns
    -------
    scipy.sparse.csc_matrix
        A 1 x (n*(n+1)/2) sparse CSC matrix containing the vectorized
        upper triangle in column-major order.
    """
    # 1. Extract only the upper triangle
    #    We guarantee rows <= cols because we take the upper triangle.
    P_upper = spa.triu(P, format="coo")

    # 2. Extract coordinates and data
    rows = P_upper.row
    cols = P_upper.col
    data = P_upper.data.copy()

    # 3. Apply scaling to off-diagonal elements
    mask_off_diag = rows != cols
    data[mask_off_diag] *= scale_factor

    # 4. Calculate 1D linear indices in Column-Major Order
    #    Formula: k = j*(j+1)/2 + i   (where j=col, i=row)
    #    Since we used spa.triu, we are guaranteed rows <= cols.
    k_indices = (cols * (cols + 1) // 2) + rows

    # 5. Construct the new 1-row matrix
    n = P.shape[0]
    total_elements = n * (n + 1) // 2

    # Create the vector.
    result = spa.coo_matrix(
        (data, (np.zeros_like(k_indices), k_indices)), shape=(1, total_elements)
    )

    return result.tocsc()
