"""Scalable (GPU-sized) variants of the example FOMs.

The built-in examples default to tiny verification-sized instances. The two
natural GPU showcases — CG-QP (pure CG) and PDHG-for-LP (pure matvec +
projection) — also accept a structure knob that produces *sparse* problems whose
generated code runs CSR matrix-vector products instead of dense ``n^2`` blocks,
so they scale. These tests check the sparse structure, the cheap (no-eigvals)
step sizing, and that a scaled instance still verifies against the oracle on the
CPU (the dtype/target machinery is shared, so no GPU is needed to validate it).
"""

from pathlib import Path

import numpy as np

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    SolveMethod,
    TerminationSpec,
    Version,
)
from fomtools.codegen.examples import (
    _banded_spd,
    _spectral_norm_bound,
    cg_qp_example,
    pdhg_lp_example,
)
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.factorize import is_spd

GOLDEN = Path(__file__).parent / "golden"
CG_OPENMP_GOLDEN = GOLDEN / "cgqp" / "large_scale_openmp" / "fom_solver.c"
PDHG_EMBEDDED_GOLDEN = GOLDEN / "pdhg_lp" / "embedded_c" / "fom_solver.c"


# -- structure / math helpers --------------------------------------------------
def test_banded_spd_is_sparse_and_well_conditioned():
    M = _banded_spd(400, bandwidth=4, seed=1)
    assert is_spd(M)
    nnz = int((np.abs(M) > 1e-12).sum())
    assert nnz < 0.1 * M.size  # structurally sparse
    eigs = np.linalg.eigvalsh(M)
    assert eigs.min() >= 2.0 - 1e-9  # contraction guarantee (no eigvals in builder)


def test_spectral_norm_bound_is_an_upper_bound():
    rng = np.random.default_rng(0)
    for shape in ((20, 30), (50, 50)):
        A = rng.normal(size=shape)
        assert _spectral_norm_bound(A) >= np.linalg.norm(A, 2) - 1e-9


def test_pdhg_sparse_step_size_is_stable():
    # tau*sigma*||A||^2 < 1 is the PDHG stability condition; the cheap upper-bound
    # norm (used for the sparse path) keeps it satisfied for the *true* spectral
    # norm, so the scaled iteration is stable.
    from fomtools.codegen.examples import _sparse_dense_matrix

    rng = np.random.default_rng(7)
    A = _sparse_dense_matrix(200, 300, density=0.05, rng=rng)
    tau = sigma = 0.9 / _spectral_norm_bound(A)
    assert tau * sigma * np.linalg.norm(A, 2) ** 2 < 1.0


# -- defaults preserved --------------------------------------------------------
def test_cg_qp_default_stays_dense():
    cfg = CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.OPENMP,
        solve_method=SolveMethod.CG,
        termination=TerminationSpec(max_iters=20),
    )
    ex = cg_qp_example()  # default: dense
    ir = extract_ir(ex["verifier"], ex["next_state"], config=cfg, name="cgqp")
    # dense 6x6 SPD -> CG matrix is fully populated
    assert ir.factorizations["t0"].nnz == 36


# -- end-to-end on CPU (scaled, sparse) ---------------------------------------
def test_cg_qp_scaled_emits_sparse_csr_and_verifies(tmp_path, requires_compiler):
    ex = cg_qp_example(n=200, bandwidth=4)
    cfg = CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.OPENMP,
        solve_method=SolveMethod.CG,
        termination=TerminationSpec(max_iters=40),
    )
    ir = extract_ir(ex["verifier"], ex["next_state"], config=cfg, name="cgqp")
    assert ir.factorizations["t0"].nnz < 0.1 * 200 * 200  # sparse CG matrix
    result = generate_code(
        ex["verifier"],
        ex["next_state"],
        tmp_path,
        config=cfg,
        driver=StubAgentDriver(CG_OPENMP_GOLDEN),
        name="cgqp",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    # the CG system matrix is emitted as CSR (sparse), not dense
    assert "FOM_FAC_T0_A_ROWPTR" in (tmp_path / "src" / "fom_linsolve.c").read_text()


def test_pdhg_scaled_emits_csr_and_verifies(tmp_path, requires_compiler):
    ex = pdhg_lp_example(n=200, m=150, density=0.05)
    cfg = CodegenConfig(termination=TerminationSpec(max_iters=60))  # embedded C
    result = generate_code(
        ex["verifier"],
        ex["next_state"],
        tmp_path,
        config=cfg,
        driver=StubAgentDriver(PDHG_EMBEDDED_GOLDEN),
        name="pdhg_lp",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    # at least one matvec is emitted as CSR (the sparse A-derived blocks)
    assert "ROWPTR" in (tmp_path / "src" / "fom_ops.c").read_text()
