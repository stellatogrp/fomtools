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
    assert (tmp_path / "SPEC.md").exists()


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


# --------------------------------------------------------------------------
# A FOM given as a recorded run, supplied through --module: the mechanism for
# user-defined FOMs, so no built-in example is needed to cover the path.
# --------------------------------------------------------------------------

RECORDED_RUN_MODULE = """
import numpy as np
from fomtools.verifier import Verifier

STEPS = 12


def build_fom():
    rng = np.random.default_rng(0)
    m, n = 4, 6
    d = rng.normal(size=(m, n)) / np.sqrt(m)
    eta = 1.0 / np.max(np.linalg.eigvalsh(d.T @ d))
    at, bt = np.eye(n) - eta * d.T @ d, eta * d.T

    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(m, lb=-np.ones(m), ub=np.ones(m))
    z = v.iterate_trajectory(STEPS + 1)
    z[0] = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    for k in range(1, STEPS + 1):
        z[k] = v.step.soft_threshold(at @ z[k - 1] + bt @ x, lambd=0.05 * eta)
    return {"verifier": v, "iterates": z, "name": "recorded"}
"""


@pytest.fixture
def recorded_run_module(tmp_path):
    path = tmp_path / "recorded_fom.py"
    path.write_text(RECORDED_RUN_MODULE)
    return str(path)


def test_scaffold_only_from_a_recorded_run(tmp_path, recorded_run_module):
    """A stationary recording rolls back up, so one block is emitted."""
    out = tmp_path / "out"
    rc = main(["--module", recorded_run_module, "--out", str(out), "--scaffold-only"])
    assert rc == 0

    spec = (out / "SPEC.md").read_text()
    assert "to be implemented as a fixed loop" in spec
    assert "#define FOM_UNROLLED 0" in (out / "include" / "fom_workspace.h").read_text()


def test_no_roll_keeps_the_recorded_run_unrolled(tmp_path, recorded_run_module):
    out = tmp_path / "out"
    rc = main(["--module", recorded_run_module, "--out", str(out), "--scaffold-only", "--no-roll"])
    assert rc == 0

    spec = (out / "SPEC.md").read_text()
    assert "to be implemented as a single pass" in spec
    header = (out / "include" / "fom_workspace.h").read_text()
    assert "#define FOM_UNROLLED 1" in header
    # One block per recorded iteration.
    assert sum(1 for line in spec.splitlines() if line.startswith("### `t")) == 12
