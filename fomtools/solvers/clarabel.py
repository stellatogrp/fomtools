import clarabel
import numpy as np
import scipy.sparse as spa

from fomtools.solvers.sdp_solver import SDPSolver
from fomtools.utils.sparse_utils import get_submatrix_selection_matrix


class ClarabelSolver(SDPSolver):

    def __init__(self, solver_params=None):
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
        vec_problem_dim = mat_problem_dim * (mat_problem_dim + 1) // 2

        q = -sparse_symmetric_vectorize(C, scale_factor=2)
        # print(q.todense().getA().reshape(-1,))
        # TODO: see if this can be replaced by a sparse 1 by n vector instead
        q = q.todense().getA().reshape(-1,)
        assert q.shape[0] == vec_problem_dim

        symmetrized_Aeq_mats = [
            sparse_symmetric_vectorize(A, scale_factor=2) for A in Aeq_mats
        ]
        Aeq = spa.vstack(symmetrized_Aeq_mats)
        beq = np.array(beq_consts)

        assert Aeq.shape == (beq.shape[0], vec_problem_dim)

        symmetrized_Aineq_mats = [
            sparse_symmetric_vectorize(A, scale_factor=2) for A in Aineq_mats
        ]
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
            'zero': Aeq.shape[0],
            'nonneg': Aineq.shape[0],
            'psd': psd_cone_sizes,
        }

        self.q = q
        self.Aeq = Aeq
        self.beq = beq
        self.Aineq = Aineq
        self.bineq = bineq
        self.M = M
        self.cone_dims = cone_dims

    def solve(self):
        q = self.q
        b_psd = [n * (n + 1) // 2 for n in self.cone_dims['psd']]
        b = np.concatenate([self.beq, self.bineq, np.zeros(sum(b_psd))])
        A = spa.vstack([self.Aeq, self.Aineq, self.M])
        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == q.shape[0]

        cones = [clarabel.ZeroConeT(self.beq.shape[0]), clarabel.NonnegativeConeT(self.bineq.shape[0])]
        cones += [
            clarabel.PSDTriangleConeT(n) for n in self.cone_dims['psd']
        ]

        xdim = q.shape[0]
        P = spa.csc_matrix((xdim, xdim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        sol = solver.solve()
        print(-sol.obj_val)
        return -sol.obj_val


def sparse_symmetric_vectorize(P, scale_factor=np.sqrt(2)):
    """
    Vectorizes a sparse symmetric matrix P (upper triangle only)
    into Column-Major Upper Triangular format (Clarabel/SCS standard).

    Applies scale_factor (default sqrt(2)) to off-diagonal elements.

    Returns: 1 x (n*(n+1)/2) sparse CSC matrix.
    """
    # 1. Extract only the upper triangle
    #    We guarantee rows <= cols because we take the upper triangle.
    P_upper = spa.triu(P, format='coo')

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
        (data, (np.zeros_like(k_indices), k_indices)),
        shape=(1, total_elements)
    )

    return result.tocsc()
