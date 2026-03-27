import importlib.util

import numpy as np
import scipy.sparse as spa

from fomtools.solvers.clarabel import ClarabelSolver


def _check_cuda_available():
    """Check if CUDA dependencies are available."""
    if importlib.util.find_spec("cupy") is None:
        raise ValueError(
            "CuClarabelSolver requires CUDA support. "
            "Install cupy with: pip install cupy-cuda12x (adjust for your CUDA version)"
        )


class CuClarabelSolver(ClarabelSolver):
    """
    GPU-accelerated solver wrapper for Clarabel using CUDA.

    This class extends ClarabelSolver to leverage GPU acceleration via CuPy and
    Julia's CUDA-enabled Clarabel implementation. It transfers problem data to
    GPU memory and solves the SDP using cuDSS for sparse linear algebra.

    Requirements
    ------------
    - CuPy with CUDA support
    - Julia with Clarabel.jl, CUDA.jl packages
    - juliacall Python package

    Notes
    -----
    Problem setup uses the parent class's ``setup_clarabel_data`` method.
    Only the ``solve`` method differs, utilizing GPU kernels.
    """

    def __init__(self, solver_params=None):
        """
        Initialize the GPU-accelerated Clarabel solver.

        Parameters
        ----------
        solver_params : dict, optional
            Dictionary of solver-specific parameters to configure.

        Raises
        ------
        ValueError
            If CUDA dependencies (cupy) are not available.
        """
        _check_cuda_available()
        super(CuClarabelSolver, self).__init__(solver_params=solver_params)

    def solve(self):
        """
        Solve the SDP on GPU and return the objective value.

        Transfers the problem data to GPU memory using CuPy and invokes
        Julia's CUDA-enabled Clarabel solver via juliacall. Uses cuDSS
        for the direct linear system solves.

        Returns
        -------
        float
            The optimal objective value of the SDP maximization problem.

        Notes
        -----
        Requires CUDA-capable GPU and properly configured Julia environment
        with CUDA.jl and Clarabel.jl packages.
        """
        q = self.q
        b_psd = [n * (n + 1) // 2 for n in self.cone_dims["psd"]]
        b = np.concatenate([self.beq, self.bineq, np.zeros(sum(b_psd))])
        A = spa.vstack([self.Aeq, self.Aineq, self.M])

        ndim = q.shape[0]

        import cupy
        from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
        from juliacall import Main as jl

        jl.seval("using CUDA")
        jl.seval("using Clarabel")
        jl.seval("using LinearAlgebra, SparseArrays, CUDA.CUSPARSE")

        # 2. Move Data to GPU
        qgpu = cupy.array(q)
        Agpu = cucsr_matrix(A)
        bgpu = cupy.array(b)

        jl.seval("global q; global A; global b")

        # 5. Transfer Pointers to Julia
        # Note: We initialize P as an empty GPU matrix since it is Zero
        jl.seval(f"""
            P = CuSparseMatrixCSR(sparse(Float64[], Float64[], Float64[], {ndim}, {ndim}))
        """)

        jl.q = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(qgpu.data.ptr), qgpu.size)

        jl.A = jl.Clarabel.cupy_to_cucsrmat(
            jl.Float64,
            int(Agpu.data.data.ptr),
            int(Agpu.indices.data.ptr),
            int(Agpu.indptr.data.ptr),
            *Agpu.shape,
            Agpu.nnz,
        )
        jl.b = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bgpu.data.ptr), bgpu.size)

        # 6. Setup Cones
        jl.seval("cones = Clarabel.SupportedCone[]")
        jl.push_b(jl.cones, jl.Clarabel.ZeroConeT(self.beq.shape[0]))
        jl.push_b(jl.cones, jl.Clarabel.NonnegativeConeT(self.bineq.shape[0]))
        for n in self.cone_dims["psd"]:
            jl.push_b(jl.cones, jl.Clarabel.PSDTriangleConeT(n))

        # 7. Solve
        # The 'MethodError' will vanish because the Extension now provides the GPU signature.
        # settings = Clarabel.Settings(direct_solve_method = :cudss, verbose = true)
        jl.seval("""
            settings = Clarabel.Settings(direct_solve_method = :cudss, verbose = true)
            solver   = Clarabel.Solver(settings)
            Clarabel.setup!(solver, P, q, A, b, cones)
            global sol = Clarabel.solve!(solver)
        """)

        obj_val = float(jl.sol.obj_val)
        return -obj_val
