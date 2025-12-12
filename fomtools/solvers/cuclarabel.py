import numpy as np
import scipy.sparse as spa

from algoverify.solvers.clarabel import ClarabelSolver


class CuClarabelSolver(ClarabelSolver):

    def __init__(self, solver_params=None):
        super(CuClarabelSolver, self).__init__(solver_params=solver_params)

    def solve(self):
        # 1. Data Prep
        q = self.q
        b_psd = [n * (n + 1) // 2 for n in self.cone_dims['psd']]
        b = np.concatenate([self.beq, self.bineq, np.zeros(sum(b_psd))])
        A = spa.vstack([self.Aeq, self.Aineq, self.M])
        
        ndim = q.shape[0]
        
        import cupy
        from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
        from juliacall import Main as jl

        # jl.seval("import CUDA")
        # # 1. Set the version to match your cluster's local toolkit
        # jl.seval('CUDA.set_runtime_version!(v"12.9")')
        # # 2. Recompile packages to apply the change
        jl.seval("import Pkg; Pkg.instantiate()")

        jl.seval("using CUDA")
        jl.seval("using CUDSS")
        jl.seval("using Clarabel")
        jl.seval("using LinearAlgebra, SparseArrays, CUDA.CUSPARSE")

        jl.seval("Pkg.status()")

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
                jl.Float64, int(Agpu.data.data.ptr), int(Agpu.indices.data.ptr),
                int(Agpu.indptr.data.ptr), *Agpu.shape, Agpu.nnz)
        jl.b = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bgpu.data.ptr), bgpu.size)


        # 6. Setup Cones
        jl.seval("cones = Clarabel.SupportedCone[]")
        jl.push_b(jl.cones, jl.Clarabel.ZeroConeT(self.beq.shape[0]))
        jl.push_b(jl.cones, jl.Clarabel.NonnegativeConeT(self.bineq.shape[0]))
        for n in self.cone_dims['psd']:
            jl.push_b(jl.cones, jl.Clarabel.PSDTriangleConeT(n))

        print('here')
        return

        # 7. Solve
        # The 'MethodError' will vanish because the Extension now provides the GPU signature.
        # settings = Clarabel.Settings(direct_solve_method = :cudss, verbose = true)
        results = jl.seval("""
            settings = Clarabel.Settings(direct_solve_method = :cudss, verbose = true)
            solver   = Clarabel.Solver(settings)
            Clarabel.setup!(solver, P, q, A, b, cones)
            Clarabel.solve!(solver)
        """)

        print(results)
