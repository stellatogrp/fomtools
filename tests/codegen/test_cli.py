"""Tests for the fomtools-codegen CLI."""

from pathlib import Path

import pytest

from fomtools.codegen.cli import main

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"


def test_scaffold_only_ista(tmp_path):
    rc = main(["--example", "ista", "--out", str(tmp_path), "--scaffold-only"])
    assert rc == 0
    assert (tmp_path / "include" / "fom_solver.h").exists()
    assert (tmp_path / "src" / "fom_data.c").exists()


def test_scaffold_only_dr(tmp_path):
    rc = main(["--example", "dr", "--out", str(tmp_path), "--scaffold-only"])
    assert rc == 0
    assert (tmp_path / "test" / "SPEC.md").exists()


def test_stub_requires_golden(tmp_path):
    with pytest.raises(SystemExit):
        main(["--example", "ista", "--out", str(tmp_path), "--driver", "stub"])


def test_cli_stub_verify(tmp_path, requires_compiler):
    rc = main(
        [
            "--example",
            "ista",
            "--out",
            str(tmp_path),
            "--driver",
            "stub",
            "--golden",
            str(GOLDEN),
            "--max-iter",
            "100",
        ]
    )
    assert rc == 0
    assert (tmp_path / "verify_bin").exists()
