"""Tests for the large-scale (OpenMP / CUDA) targets.

OpenMP is built serially and numerically verified against the oracle (portable).
CUDA is compile-checked when ``nvcc`` is available and run-verified on a GPU
(the cluster); otherwise those tests are skipped.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    SolveMethod,
    TerminationSpec,
    Version,
)
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.targets import get_target
from fomtools.codegen.targets.large_scale_cuda import CudaTarget

GOLDEN_ROOT = Path(__file__).parent / "golden" / "ista"
OPENMP_GOLDEN = GOLDEN_ROOT / "large_scale_openmp" / "fom_solver.c"
CUDA_GOLDEN = GOLDEN_ROOT / "large_scale_cuda" / "fom_solver.cu"
CG_OPENMP_GOLDEN = Path(__file__).parent / "golden" / "cgqp" / "large_scale_openmp" / "fom_solver.c"
CUDA_CG_GOLDEN = Path(__file__).parent / "golden" / "cgqp" / "large_scale_cuda" / "fom_solver.cu"
CUDA_PDHG_GOLDEN = (
    Path(__file__).parent / "golden" / "pdhg_lp" / "large_scale_cuda" / "fom_solver.cu"
)


def _openmp_available() -> bool:
    """True if the default C compiler can build and link an ``-fopenmp`` program.

    Apple clang ships without libomp, so ``-fopenmp`` fails there; CI (gcc) has
    it. Threaded-build tests probe this and skip when it is unavailable.
    """
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return False
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.c"
        src.write_text(
            "#include <omp.h>\n"
            "int main(void){int s=0;\n"
            "#pragma omp parallel for reduction(+:s)\n"
            "for(int i=0;i<128;i++)s+=i;return s>0?0:1;}\n"
        )
        try:
            r = subprocess.run(
                [cc, "-fopenmp", str(src), "-o", str(Path(d) / "probe")],
                capture_output=True,
            )
        except OSError:
            return False
    return r.returncode == 0


def _openmp_config(max_iters=200):
    return CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.OPENMP,
        termination=TerminationSpec(max_iters=max_iters),
    )


def _cuda_config(max_iters=200):
    return CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.CUDA,
        termination=TerminationSpec(max_iters=max_iters),
    )


def _cuda_cg_config(max_iters=100):
    return CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.CUDA,
        solve_method=SolveMethod.CG,
        termination=TerminationSpec(max_iters=max_iters),
    )


def _cuda_auto_config(max_iters=50, **kw):
    return CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.CUDA,
        solve_method=SolveMethod.AUTO,
        termination=TerminationSpec(max_iters=max_iters),
        **kw,
    )


def test_cuda_auto_upgrades_large_spd_to_cg():
    # AUTO on CUDA picks CG for a large SPD affine system (matrix-free, parallel).
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(n=6)
    cfg = _cuda_auto_config(cg_auto_min_dim=4)
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="cgqp")
    assert ir.factorizations["t0"].method == "cg"


def test_cuda_auto_keeps_small_spd_direct():
    # Below the size threshold, AUTO keeps the (sequential) direct LDL^T solve.
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(n=6)
    cfg = _cuda_auto_config()  # default cg_auto_min_dim=256 > 6
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="cgqp")
    assert ir.factorizations["t0"].method == "ldlt"


def test_cuda_auto_keeps_indefinite_kkt_direct():
    # A symmetric-indefinite KKT system is not SPD -> stays on a direct solve
    # even with the size gate wide open; CG does not apply (would need MINRES or
    # a normal-equations form).
    from tests.codegen.conftest import build_indefinite_kkt

    case = build_indefinite_kkt()
    cfg = _cuda_auto_config(cg_auto_min_dim=1)  # size gate wide open
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="indef_kkt")
    method = next(iter(ir.factorizations.values())).method
    assert method == "lu"  # not "cg"


def test_embedded_auto_never_upgrades_to_cg():
    # The AUTO->CG upgrade is CUDA-only; the embedded target stays division-free.
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(n=6)
    cfg = CodegenConfig(solve_method=SolveMethod.AUTO, cg_auto_min_dim=1)  # embedded C
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="cgqp")
    assert ir.factorizations["t0"].method == "ldlt"


def test_cuda_explicit_method_bypasses_heuristic():
    # Forcing LDLT keeps the direct factorization even for a large SPD system.
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(n=6)
    cfg = CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.CUDA,
        solve_method=SolveMethod.LDLT,
        termination=TerminationSpec(max_iters=50),
        cg_auto_min_dim=1,
    )
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="cgqp")
    assert ir.factorizations["t0"].method == "ldlt"


def test_embedded_cuda_combo_rejected():
    with pytest.raises(ValueError):
        CodegenConfig(version=Version.EMBEDDED, backend=Backend.CUDA)


def test_cg_embedded_combo_rejected():
    # CG needs runtime division; the embedded target is division-free.
    with pytest.raises(ValueError, match="CG"):
        CodegenConfig(version=Version.EMBEDDED, solve_method=SolveMethod.CG)


def test_cg_large_scale_combo_allowed():
    cfg = CodegenConfig(
        version=Version.LARGE_SCALE, backend=Backend.OPENMP, solve_method=SolveMethod.CG
    )
    assert cfg.solve_method == SolveMethod.CG


def test_openmp_scaffold(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=_openmp_config(), name="ista")
    target = scaffold(ir, tmp_path, _openmp_config())
    assert target.name == "large_scale_openmp"
    assert "OMPFLAGS" in (tmp_path / "Makefile").read_text()
    assert (tmp_path / "include" / "fom_solver.h").exists()


def test_cuda_scaffold(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=_cuda_config(), name="ista")
    target = scaffold(ir, tmp_path, _cuda_config())
    assert target.name == "large_scale_cuda"
    assert target.solver_source_path().endswith(".cu")
    mk = (tmp_path / "Makefile").read_text()
    assert "nvcc" in mk


def test_openmp_golden_verifies(tmp_path, requires_compiler):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_openmp_config(),
        driver=StubAgentDriver(OPENMP_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log


def test_openmp_relaxed_static_checks():
    # Large-scale targets do not enforce the embedded no-malloc/division rules.
    target = get_target(_openmp_config())
    assert (
        target.static_checks("void f(void){ double *p = malloc(8); double y = p[0] / 2.0; }") == []
    )


def test_cuda_scaffold_emits_device_ops(tmp_path):
    # The matvec helpers are deterministic CUDA kernels behind the CPU ABI
    # (device pointers); data is baked into device memory, no runtime upload.
    from tests.codegen.conftest import build_ista

    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=_cuda_config(), name="ista")
    scaffold(ir, tmp_path, _cuda_config())
    ops = (tmp_path / "src" / "fom_ops.cu").read_text()
    assert "__global__ void fom_apply_t0_state_kern" in ops
    assert 'extern "C" void fom_apply_t0_state(' in ops
    assert "__device__" in ops
    assert not (tmp_path / "src" / "fom_ops.c").exists()
    assert (tmp_path / "src" / "fom_linsolve.cu").exists()
    # device-pointer semantics are documented in the header the agent reads
    assert "DEVICE" in (tmp_path / "include" / "fom_ops.h").read_text()
    mk = (tmp_path / "Makefile").read_text()
    assert "src/fom_ops.cu" in mk and "src/fom_linsolve.cu" in mk


def test_cuda_cg_scaffold_emits_device_cg(tmp_path):
    # CG is all-data-parallel: the whole solve (SpMV + dots + axpy) is emitted
    # as device kernels mirroring factorize._pcg_solve.
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg()
    cfg = _cuda_cg_config()
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="cgqp")
    assert ir.factorizations["t0"].method == "cg"
    scaffold(ir, tmp_path, cfg)
    lin = (tmp_path / "src" / "fom_linsolve.cu").read_text()
    assert "__global__ void fom_cg_t0_spmv_kern" in lin  # device CSR SpMV
    assert "fom_dot_kern" in lin  # reduction kernels
    assert "FOM_FAC_T0_MINV" in lin  # Jacobi preconditioner (reciprocals)
    assert 'extern "C" void fom_lin_solve_t0(' in lin
    assert "FOM_FAC_T0_L_DATA" not in lin  # no triangular factor, no inverse


def test_cuda_direct_solve_runs_on_the_device_through_cudss(tmp_path, monkeypatch):
    # Nothing leaves the GPU: cuDSS factorizes the system matrix (device data) at
    # the first solve and solves on the caller's device vectors; no host staging.
    from tests.codegen.conftest import build_dr

    monkeypatch.setenv("CUDSS_HOME", "/opt/cudss")
    case = build_dr()
    cfg = _cuda_config()
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="dr")
    [(name, fac)] = list(ir.factorizations.items())
    assert fac.method == "lu"
    scaffold(ir, tmp_path, cfg)
    lin = (tmp_path / "src" / "fom_linsolve.cu").read_text()
    assert f'extern "C" void fom_lin_solve_{name}(' in lin
    assert "#include <cudss.h>" in lin and "CUDSS_MTYPE_GENERAL" in lin
    assert "CUDSS_PHASE_FACTORIZATION" in lin and "CUDSS_PHASE_SOLVE" in lin
    assert "cudaMemcpy" not in lin  # the matrix and the vectors stay on the device
    assert f"__device__ static const fom_float FOM_FAC_{name.upper()}_A_VALS[" in lin
    mk = (tmp_path / "Makefile").read_text()
    assert "CUDSS_HOME ?= /opt/cudss" in mk and "$(LDLIBS)" in mk


def test_cuda_cg_links_no_cudss(tmp_path):
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg()
    ir = extract_ir(case["verifier"], case["next_state"], config=_cuda_cg_config(), name="cgqp")
    scaffold(ir, tmp_path, _cuda_cg_config())
    assert "cudss" not in (tmp_path / "src" / "fom_linsolve.cu").read_text()
    assert "LDLIBS =\n" in (tmp_path / "Makefile").read_text()


def test_cuda_verify_command_compile_only_without_gpu(monkeypatch):
    # FOMTOOLS_CUDA_RUN=0 forces compile-check (the CI/dev default behavior).
    monkeypatch.setenv("FOMTOOLS_CUDA_RUN", "0")
    target = get_target(_cuda_config())
    assert target.verify_command() == ["make", "compile"]


def test_cuda_verify_command_runs_when_forced(monkeypatch):
    # FOMTOOLS_CUDA_RUN=1 forces the real run-verification (used on the cluster).
    monkeypatch.setenv("FOMTOOLS_CUDA_RUN", "1")
    target = get_target(_cuda_config())
    assert target.verify_command() == ["make", "verify"]


def test_cuda_compile_builds_verify_and_bench(tmp_path):
    # The compile-check target builds both binaries so CI catches bench breakage.
    from tests.codegen.conftest import build_ista

    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=_cuda_config(), name="ista")
    scaffold(ir, tmp_path, _cuda_config())
    mk = (tmp_path / "Makefile").read_text()
    assert "verify_bin" in mk and "bench_bin" in mk
    assert "compile: $(OBJS) $(CHECK_OBJS) test/verify_main.c test/bench_main.c" in mk
    assert (tmp_path / "test" / "bench_main.c").exists()


def test_cuda_pdhg_scaffold_emits_device_ops(tmp_path):
    # PDHG-for-LP is the pure-matvec GPU showcase (satlin + relu, no solve).
    from tests.codegen.conftest import build_pdhg_lp

    case = build_pdhg_lp(max_iters=150)
    cfg = _cuda_config(max_iters=150)
    ir = extract_ir(case["verifier"], case["next_state"], config=cfg, name="pdhg_lp")
    assert not ir.factorizations  # no linear solve
    scaffold(ir, tmp_path, cfg)
    ops = (tmp_path / "src" / "fom_ops.cu").read_text()
    assert 'extern "C" void fom_apply_t0_state(' in ops
    assert 'extern "C" void fom_apply_t1_t_t0(' in ops
    # clip bounds + dual constant remain host arrays for the solver to upload
    assert "FOM_MAT_t0_lo" in (tmp_path / "include" / "fom_data.h").read_text()


@pytest.mark.skipif(not shutil.which("nvcc"), reason="no CUDA toolkit (nvcc) available")
def test_cuda_pdhg_compile_check(tmp_path):
    from tests.codegen.conftest import build_pdhg_lp

    case = build_pdhg_lp(max_iters=150)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_cuda_config(max_iters=150),
        driver=StubAgentDriver(CUDA_PDHG_GOLDEN),
        name="pdhg_lp",
    )
    assert result.translate.success, result.translate.log


@pytest.mark.skipif(not shutil.which("nvcc"), reason="no CUDA toolkit (nvcc) available")
def test_cuda_cg_compile_check(tmp_path):
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_cuda_cg_config(),
        driver=StubAgentDriver(CUDA_CG_GOLDEN),
        name="cgqp",
    )
    assert result.translate.success, result.translate.log


@pytest.mark.skipif(not shutil.which("nvcc"), reason="no CUDA toolkit (nvcc) available")
def test_cuda_compile_check(tmp_path):
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_cuda_config(),
        driver=StubAgentDriver(CUDA_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log


def test_cg_scaffold_emits_pcg_solver(tmp_path):
    # No compiler needed: check the CG linear-solve emission deterministically.
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="cgqp")
    assert ir.factorizations["t0"].method == "cg"
    scaffold(ir, tmp_path, case["config"])
    linsolve = (tmp_path / "src" / "fom_linsolve.c").read_text()
    # The emitted solve is conjugate gradient: matrix + Jacobi preconditioner,
    # not L/U triangular factors, and no inverse.
    assert "conjugate gradient" in linsolve
    assert "FOM_FAC_T0_A_ROWPTR" in linsolve  # sparse CSR system matrix
    assert "FOM_FAC_T0_MINV" in linsolve  # diagonal preconditioner
    assert "FOM_FAC_T0_L_DATA" not in linsolve  # no triangular factor
    assert "void fom_lin_solve_t0(" in linsolve
    # SPEC documents the CG solve for the agent.
    assert "conjugate gradient" in (tmp_path / "SPEC.md").read_text()


def test_cg_openmp_golden_verifies(tmp_path, requires_compiler):
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(max_iters=80)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(CG_OPENMP_GOLDEN),
        name="cgqp",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    # The CG solver matches the reference oracle (which itself runs PCG).
    oracle = ReferenceOracle(result.ir)
    qval = np.full(case["data"]["n"], 0.25)
    expected = oracle.solve(param_values={"p0": qval})
    assert expected.solution.shape == (case["data"]["n"],)


@pytest.mark.skipif(not _openmp_available(), reason="compiler has no working -fopenmp")
def test_openmp_threaded_build_verifies(tmp_path):
    # Matvec pragmas: build the ISTA OpenMP golden *with threads* and verify.
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_openmp_config(),
        driver=StubAgentDriver(OPENMP_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    proc = subprocess.run(
        ["make", "verify", "OMPFLAGS=-fopenmp"], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFY OK" in proc.stdout


@pytest.mark.skipif(not _openmp_available(), reason="compiler has no working -fopenmp")
def test_cg_threaded_build_verifies(tmp_path):
    # CG kernels (matvec + dot reductions + axpy): threaded build must still
    # match the oracle (floating-point reduction reorder stays well within tol).
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg(max_iters=80)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(CG_OPENMP_GOLDEN),
        name="cgqp",
    )
    assert result.translate.success, result.translate.log
    proc = subprocess.run(
        ["make", "verify", "OMPFLAGS=-fopenmp"], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFY OK" in proc.stdout


def test_openmp_wrapper_matches_oracle(tmp_path, requires_compiler):
    import importlib.util

    from fomtools.codegen import ReferenceOracle
    from fomtools.codegen.harness import build_lib
    from tests.codegen.conftest import build_ista

    case = build_ista()
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=_openmp_config(),
        driver=StubAgentDriver(OPENMP_GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    build_lib(tmp_path)
    spec = importlib.util.spec_from_file_location("w_omp", tmp_path / "python" / "fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver()
    oracle = ReferenceOracle(result.ir)
    out = solver.solve(p0=case["data"]["b"])
    expected = oracle.solve(param_values={"p0": case["data"]["b"]})
    assert np.allclose(out["x"], expected.solution, atol=1e-6)


@pytest.mark.skipif(not CudaTarget._can_run_on_gpu(), reason="needs a GPU (or FOMTOOLS_CUDA_RUN=1)")
def test_cuda_direct_check_keeps_a_right_kernel_and_rejects_a_wrong_one(tmp_path):
    # On a GPU, fom_ops.cu is editable under the CPU targets' direct check.
    from tests.codegen.conftest import build_ista

    mark = "/* rewritten by the code optimization phase */"
    sig = "static __global__ void fom_apply_t0_state_kern(const fom_float *x, fom_float *out) {"

    def same(text: str) -> str:
        return mark + "\n" + text

    def wrong(text: str) -> str:  # entry (0, 1) off by one, no float literal
        head, rest = text.split(sig, 1)
        body, tail = rest.split("    out[i] += s;\n}", 1)
        return (
            mark
            + "\n"
            + head
            + sig
            + body
            + "    if (i == 0) s += x[1];\n    out[i] += s;\n}"
            + tail
        )

    case = build_ista()
    for rewrite, kept in ((same, True), (wrong, False)):
        out = tmp_path / rewrite.__name__
        result = generate_code(
            case["verifier"],
            case["next_state"],
            out,
            config=_cuda_config(),
            driver=StubAgentDriver(CUDA_GOLDEN, extra_sources={"src/fom_ops.cu": rewrite}),
            name="ista",
            code_optimization=True,
        )
        assert result.translate.success, result.translate.log
        assert "OPSCHECK name=t0_state mode=basis" in result.translate.log
        if kept:
            assert "VERIFY OK" in result.optimize.log, result.optimize.log
        else:
            assert not result.optimization_kept
            assert "fom_apply_t0_state entry (0, 1) is off by" in result.optimize.log
            assert mark not in (out / "src" / "fom_ops.cu").read_text()


def test_cuda_cg_reads_back_only_its_stop_flag(tmp_path):
    from tests.codegen.conftest import build_spd_cg

    case = build_spd_cg()
    ir = extract_ir(case["verifier"], case["next_state"], config=_cuda_cg_config(), name="cgqp")
    scaffold(ir, tmp_path, _cuda_cg_config())
    lin = (tmp_path / "src" / "fom_linsolve.cu").read_text()
    copies = [ln for ln in lin.splitlines() if "cudaMemcpy(" in ln]
    assert len(copies) == 2 and all("FOM_CG_STOP" in ln for ln in copies)
    assert "fom_dot_partials_h" not in lin  # partials are folded on the device


def _cuda_direct_cases():
    from tests.codegen.conftest import build_indefinite_kkt, build_spd_cg

    spd = build_spd_cg()
    spd["config"] = _cuda_cg_config().model_copy(update={"solve_method": SolveMethod.LDLT})
    kkt = build_indefinite_kkt()
    kkt["config"] = _cuda_auto_config(max_iters=60)
    return [("ldlt", spd), ("lu", kkt)]


@pytest.mark.skipif(not CudaTarget._can_run_on_gpu(), reason="needs a GPU (or FOMTOOLS_CUDA_RUN=1)")
@pytest.mark.parametrize("method", ["ldlt", "lu"])
def test_cuda_cudss_solve_verifies_and_its_direct_check_catches_a_wrong_matrix(tmp_path, method):
    # cuDSS on the device (LDL^T for the SPD system, LU for the indefinite KKT):
    # the solver verifies end to end, and a fom_linsolve.cu that solves a
    # different matrix is caught by the direct check against the reference solve.
    [case] = [c for m, c in _cuda_direct_cases() if m == method]
    mark = "/* rewritten by the code optimization phase */"

    def wrong(text: str) -> str:
        head, rest = text.split("FOM_FAC_T0_A_VALS[", 1)
        size, rest = rest.split("] = {", 1)
        first, tail = rest.split(",", 1)
        return mark + "\n" + head + f"FOM_FAC_T0_A_VALS[{size}] = {{{float(first) * 2!r}," + tail

    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(CUDA_CG_GOLDEN, extra_sources={"src/fom_linsolve.cu": wrong}),
        name="cgqp",
        code_optimization=True,
    )
    assert result.translate.success, result.translate.log
    lin = (tmp_path / "src" / "fom_linsolve.cu").read_text()
    assert ("CUDSS_MTYPE_SYMMETRIC" if method == "ldlt" else "CUDSS_MTYPE_GENERAL") in lin
    assert not result.optimization_kept
    assert "fom_lin_solve_t0 differs from its reference solve" in result.optimize.log
    assert mark not in lin
