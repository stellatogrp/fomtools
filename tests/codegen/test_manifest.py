"""The verification record (``fomtools_manifest.json``) and what protects it.

A run that finishes and verifies records the hashes of the sources a library
build compiles; a run that is interrupted or fails leaves the best verified
sources and no record; and every load path checks the directory against it.
"""

import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import pytest

import fomtools
from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen import generate as gen
from fomtools.codegen.agent.driver import AgentDriver
from fomtools.codegen.cli import main as cli_main
from fomtools.codegen.harness import build_lib
from fomtools.codegen.manifest import MANIFEST_NAME, check_manifest
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"
SOLVER = "src/fom_solver.c"
NO_RECORD = "no verification record"


def _generate(out_dir, driver=None, **kwargs):
    case = build_ista(max_iters=50)
    return generate_code(
        case["verifier"],
        case["next_state"],
        out_dir,
        config=case["config"],
        driver=driver or StubAgentDriver(GOLDEN),
        name="ista",
        **kwargs,
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(out_dir):
    """Every source and generated file a build reads, by content."""
    return {
        p.relative_to(out_dir).as_posix(): p.read_bytes()
        for sub in ("src", "include", "test")
        for p in sorted((out_dir / sub).iterdir())
        if p.is_file()
    }


def _wrapper(out_dir):
    spec = importlib.util.spec_from_file_location("w_manifest", out_dir / "python/fom_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- writing the record ------------------------------------------------------
@pytest.mark.requires_compiler
def test_a_verified_run_records_hashes_and_verify_numbers(tmp_path, requires_compiler):
    result = _generate(tmp_path)
    assert result.success
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["fomtools_version"] == fomtools.__version__
    assert manifest["target"] == "embedded_c" and manifest["dtype"] == "float64"
    # The numbers are the ones verify printed.
    line = next(ln for ln in result.translate.log.splitlines() if "worst max-abs error" in ln)
    verify = manifest["verify"]
    assert verify["command"] == "make verify"
    assert f"over {verify['samples']} samples" in line
    assert f"{verify['max_abs_error']:.3e}" in line and f"tol {verify['tolerance']:.3e}" in line
    assert verify["max_abs_error"] <= verify["tolerance"]
    assert "timing_us" not in manifest  # nothing was measured
    # Every source `make lib` and `make release` compile, with its hash on disk.
    sources = manifest["sources"]
    assert set(sources) >= {
        SOLVER,
        "src/fom_data.c",
        "src/fom_linsolve.c",
        "src/fom_ops.c",
        "src/fom_objective.c",
        "include/fom_workspace.h",
        "release/fom_solver.c",
    }
    assert all(_sha(tmp_path / rel) == digest for rel, digest in sources.items())
    assert not (tmp_path / (MANIFEST_NAME + ".tmp")).exists()


@pytest.mark.requires_compiler
def test_a_rerun_replaces_the_record(tmp_path, requires_compiler):
    _generate(tmp_path)
    first = json.loads((tmp_path / MANIFEST_NAME).read_text())
    _generate(tmp_path, driver=StubAgentDriver(GOLDEN.read_text() + "\n/* second run */\n"))
    second = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert second["sources"][SOLVER] != first["sources"][SOLVER]
    assert second["sources"][SOLVER] == _sha(tmp_path / SOLVER)


@pytest.mark.requires_compiler
def test_a_failed_translation_writes_no_record_and_removes_a_stale_one(tmp_path, requires_compiler):
    _generate(tmp_path)
    assert (tmp_path / MANIFEST_NAME).exists()
    result = _generate(tmp_path, driver=StubAgentDriver("this is not C;\n"))
    assert not result.success
    assert not (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.requires_compiler
def test_scaffold_only_removes_a_stale_record(tmp_path, requires_compiler):
    _generate(tmp_path)
    assert cli_main(["--example", "ista", "--out", str(tmp_path), "--scaffold-only"]) == 0
    assert not (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.requires_compiler
def test_timings_are_recorded_when_measured(tmp_path, monkeypatch, requires_compiler):
    monkeypatch.setattr(gen, "_bench_mean_us", lambda out_dir, **kwargs: 100.0)
    monkeypatch.setattr(gen, "run_profile", lambda out_dir, **kwargs: [])
    _generate(tmp_path, profile=True)
    timing = json.loads((tmp_path / MANIFEST_NAME).read_text())["timing_us"]
    assert timing["translated"] == 100.0 and timing["final"] == 100.0
    assert timing["python"] > 0


# -- an optimization round that raises or is interrupted ----------------------
class InterruptedDriver(AgentDriver):
    """Translation and round 1: the golden solver (round 1 tagged ``r1``). Round
    2 rewrites every editable source and one generated file, then raises."""

    def __init__(self, exc):
        self.exc = exc
        self.rounds = 0
        self.best: dict[str, bytes] = {}

    def run(self, task):
        if task.phase == "translation" or task.extra.get("summary_only"):
            return StubAgentDriver(GOLDEN).run(task)
        self.rounds += 1
        if self.rounds == 1:
            return StubAgentDriver(GOLDEN.read_text() + "\n/* r1 */\n").run(task)
        self.best = _snapshot(task.workdir)  # round 1 was kept: the best verified sources
        del self.best["test/kernel_check.json"]  # present only while an agent works
        for rel in task.extra["editable_paths"]:
            (task.workdir / rel).write_text("/* unverified candidate */\n")
        (task.workdir / "src" / "fom_data.c").write_text("/* not the data */\n")
        raise self.exc


@pytest.mark.requires_compiler
@pytest.mark.parametrize("exc", [RuntimeError("agent crashed"), KeyboardInterrupt()])
def test_an_interrupted_round_leaves_the_best_verified_sources(
    tmp_path, monkeypatch, requires_compiler, exc
):
    # Fake timings (r1 is faster than the translated solver), so round 1 is kept.
    def fake_head_to_head(out_dir, write_editable, candidate, best, **kwargs):
        write_editable(best)
        return 50.0, 100.0

    monkeypatch.setattr(gen, "_bench_mean_us", lambda out_dir, **kwargs: 100.0)
    monkeypatch.setattr(gen, "_head_to_head_us", fake_head_to_head)
    monkeypatch.setattr(gen, "run_profile", lambda out_dir, **kwargs: [])
    _generate(tmp_path)  # an earlier run: its record must not survive the interrupted one
    driver = InterruptedDriver(exc)
    with pytest.raises(type(exc)):
        _generate(tmp_path, driver=driver, code_optimization=True, profile=True)
    assert driver.rounds == 2
    assert "/* r1 */" in (tmp_path / SOLVER).read_text()
    assert _snapshot(tmp_path) == driver.best  # byte-identical, generated files included
    assert not (tmp_path / "test" / "kernel_check.json").exists()
    assert not (tmp_path / MANIFEST_NAME).exists()


# -- loading -----------------------------------------------------------------
@pytest.mark.requires_compiler
def test_an_untouched_directory_loads_silently(tmp_path, requires_compiler):
    _generate(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_lib(tmp_path)
        build_lib(tmp_path, make_target="release")
        _wrapper(tmp_path).Solver()


@pytest.mark.requires_compiler
def test_a_directory_without_a_record_warns_and_loads(tmp_path, requires_compiler):
    _generate(tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()  # as an older fomtools left it
    with pytest.warns(UserWarning, match=NO_RECORD):
        build_lib(tmp_path)
    with pytest.warns(UserWarning, match=NO_RECORD):
        _wrapper(tmp_path).Solver(allow_unverified=True)


@pytest.mark.requires_compiler
@pytest.mark.parametrize("rel", [SOLVER, "src/fom_data.c", "release/fom_solver.c"])
def test_a_modified_source_raises_and_is_named(tmp_path, requires_compiler, rel):
    _generate(tmp_path)
    build_lib(tmp_path)
    with (tmp_path / rel).open("a") as f:
        f.write("\n/* edited by hand */\n")
    for load in (build_lib, check_manifest, lambda out_dir, **kw: _wrapper(out_dir).Solver(**kw)):
        with pytest.raises(RuntimeError, match="modified after verification") as err:
            load(tmp_path)
        assert rel in str(err.value) and "allow_unverified=True" in str(err.value)
        # The opt-out turns the error into a warning that names the same file.
        with pytest.warns(UserWarning, match=f"modified after verification: {rel} "):
            load(tmp_path, allow_unverified=True)


def test_a_missing_source_counts_as_modified(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"sources": {SOLVER: "0" * 64}}))
    with pytest.raises(RuntimeError, match=r"src/fom_solver\.c \(missing\)"):
        check_manifest(tmp_path)
