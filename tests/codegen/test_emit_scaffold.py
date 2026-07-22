"""Tests for the deterministic scaffolding (no coding agent involved)."""

import subprocess

from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from tests.codegen.conftest import build_ista


def _scaffold_ista(tmp_path, max_iters=50):
    case = build_ista(max_iters=max_iters)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="ista")
    target = scaffold(ir, tmp_path, case["config"])
    return ir, target, case


def test_file_tree(tmp_path):
    _scaffold_ista(tmp_path)
    expected = [
        "include/fom_workspace.h",
        "include/fom_solver.h",
        "include/fom_data.h",
        "src/fom_data.c",
        "test/verify_main.c",
        "test/verify_data.h",
        "spec.json",
        "data.npz",
        "SPEC.md",
        "Makefile",
        "README.md",
        "python/fom_solver.py",
        "python/__init__.py",
    ]
    for rel in expected:
        assert (tmp_path / rel).exists(), f"missing {rel}"
    # The solver body is NOT scaffolded; the agent/golden writes it.
    assert not (tmp_path / "src" / "fom_solver.c").exists()


def test_workspace_defines_match_ir(tmp_path):
    ir, _, case = _scaffold_ista(tmp_path, max_iters=77)
    text = (tmp_path / "include" / "fom_workspace.h").read_text()
    assert f"#define FOM_STATE_DIM {ir.state_dim}" in text
    assert "#define FOM_MAX_ITER 77" in text
    assert f"#define FOM_PARAM_P0_DIM {case['data']['m']}" in text
    # ABI structs present
    for sym in ("FomSettings", "FomParams", "FomSolution", "FomWorkspace"):
        assert sym in text


def test_abi_prototypes_present(tmp_path):
    _scaffold_ista(tmp_path)
    text = (tmp_path / "include" / "fom_solver.h").read_text()
    for proto in (
        "fom_setup",
        "fom_update_params",
        "fom_solve",
        "fom_cleanup",
        "fom_set_default_settings",
    ):
        assert proto in text


def test_data_values_match_ir(tmp_path):
    ir, _, _ = _scaffold_ista(tmp_path)
    src = (tmp_path / "src" / "fom_data.c").read_text()
    header = (tmp_path / "include" / "fom_data.h").read_text()
    # initial state is 2.0 in both components
    assert "FOM_STATE_INIT" in src
    # every IR matrix is both declared (extern) and defined
    for name in ir.matrices:
        assert f"FOM_MAT_{name}" in header
        assert f"FOM_MAT_{name}" in src
        assert f"#define FOM_MAT_{name}_ROWS" in header


def test_data_source_compiles(tmp_path, requires_compiler):
    _scaffold_ista(tmp_path)
    obj = tmp_path / "fom_data.o"
    res = subprocess.run(
        [requires_compiler, "-O2", "-Iinclude", "-Itest", "-c", "src/fom_data.c", "-o", str(obj)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr


def test_verify_main_syntax_ok(tmp_path, requires_compiler):
    _scaffold_ista(tmp_path)
    res = subprocess.run(
        [requires_compiler, "-Iinclude", "-Itest", "-fsyntax-only", "test/verify_main.c"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
