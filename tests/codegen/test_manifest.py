"""The verification record (``fomtools_manifest.json``) and what protects it.

A run that finishes and verifies records the hashes of the sources a library
build compiles; a run that is interrupted or fails leaves the best verified
sources and no record; and every load path checks the directory against it.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import warnings
from pathlib import Path

import pytest

import fomtools
from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen import generate as gen
from fomtools.codegen.agent.driver import AgentDriver, AgentResult
from fomtools.codegen.cli import main as cli_main
from fomtools.codegen.harness import build_lib
from fomtools.codegen.manifest import MANIFEST_NAME, STAMP_NAME, check_manifest
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"
SOLVER = "src/fom_solver.c"
NO_RECORD = "no verification record"


def _generate(out_dir, driver=None, max_iters=50, **kwargs):
    case = build_ista(max_iters=max_iters)
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


class ForgingDriver(AgentDriver):
    """An agent that writes a record itself, and quotes verify numbers in its log."""

    def __init__(self, source, report_verify):
        self.stub = StubAgentDriver(source)
        self.report_verify = report_verify

    def run(self, task):
        result = self.stub.run(task)
        (task.workdir / MANIFEST_NAME).write_text(json.dumps({"sources": {}}))
        quoted = "\nworst max-abs error over 99 samples: 0.000e+00 (tol 1.000e-06)\n"
        return AgentResult(
            success=result.success,
            log=result.log + quoted,
            verify_output=result.verify_output if self.report_verify else None,
        )


@pytest.mark.requires_compiler
@pytest.mark.parametrize("source", [GOLDEN, "this is not C;\n"])
def test_a_record_written_by_the_agent_does_not_survive(tmp_path, requires_compiler, source):
    # A failed run, and a passing one whose driver reports no verify output (as
    # after a compile-only check): the agent's file and the numbers it quoted in
    # its log must not stand in for a record.
    result = _generate(tmp_path, driver=ForgingDriver(source, report_verify=False))
    assert result.success == (source is GOLDEN)
    assert not (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.requires_compiler
def test_the_record_takes_its_numbers_from_the_verify_output(tmp_path, requires_compiler):
    _generate(tmp_path, driver=ForgingDriver(GOLDEN, report_verify=True))
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["verify"]["samples"] != 99  # not what the agent quoted
    assert SOLVER in manifest["sources"]


def test_an_interrupted_head_to_head_leaves_no_bench_binaries(tmp_path, monkeypatch):
    def build(out_dir, name, timeout_s):
        if name == ".bench_best":
            raise KeyboardInterrupt
        (out_dir / name).write_text("binary")
        return out_dir / name

    monkeypatch.setattr(gen, "_build_bench", build)
    with pytest.raises(KeyboardInterrupt):
        gen._head_to_head_us(tmp_path, lambda sources: None, {}, {})
    assert list(tmp_path.iterdir()) == []


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
    (tmp_path / "src" / "fom_data.c").write_text("")
    sources = {SOLVER: "0" * 64, "src/fom_data.c": hashlib.sha256(b"").hexdigest()}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"sources": sources}))
    with pytest.raises(RuntimeError, match=r"verification: src/fom_solver\.c \(missing\) no"):
        check_manifest(tmp_path)


def test_a_record_without_any_of_its_sources_warns(tmp_path):
    # The wrapper and a prebuilt library, deployed with the record but no sources.
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"sources": {SOLVER: "0" * 64}}))
    with pytest.warns(UserWarning, match="none of the sources recorded"):
        check_manifest(tmp_path)


@pytest.mark.parametrize("text", ["{not json", "[]", "{}", '{"sources": 3}', '{"sources": {}}'])
def test_an_unreadable_record_raises_like_a_modified_source(tmp_path, text):
    (tmp_path / MANIFEST_NAME).write_text(text)
    with pytest.raises(RuntimeError, match="not a readable verification record"):
        check_manifest(tmp_path)
    with pytest.warns(UserWarning, match="not a readable verification record"):
        check_manifest(tmp_path, allow_unverified=True)


# -- the built library -------------------------------------------------------
def _load_silently(out_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _wrapper(out_dir).Solver()


@pytest.mark.requires_compiler
def test_a_library_built_before_a_regeneration_is_not_loaded(tmp_path, requires_compiler):
    _generate(tmp_path)
    lib = build_lib(tmp_path)
    stamp = json.loads((tmp_path / STAMP_NAME).read_text())
    assert stamp["sha256"] == _sha(lib) and stamp["library"] == lib.name
    assert stamp["sources"] == json.loads((tmp_path / MANIFEST_NAME).read_text())["sources"]

    _generate(tmp_path, max_iters=60)  # new verified sources, the old library still there
    os.utime(lib)  # a fresh mtime must not hide it: the stamp compares contents
    check_manifest(tmp_path)  # the sources themselves are fine
    with pytest.raises(RuntimeError, match="was built from other sources: ") as err:
        _wrapper(tmp_path).Solver()
    assert "build_lib" in str(err.value) and "allow_unverified=True" in str(err.value)
    with pytest.warns(UserWarning, match="was built from other sources: "):
        _wrapper(tmp_path).Solver(allow_unverified=True)

    build_lib(tmp_path)  # always rebuilds and stamps again
    _load_silently(tmp_path)


@pytest.mark.requires_compiler
def test_a_hand_built_library_without_a_stamp_is_checked_by_age(tmp_path, requires_compiler):
    _generate(tmp_path)
    subprocess.run(["make", "lib"], cwd=tmp_path, check=True, capture_output=True)
    assert not (tmp_path / STAMP_NAME).exists()
    _load_silently(tmp_path)  # newer than every source

    _generate(tmp_path, max_iters=60)
    with pytest.raises(RuntimeError, match="has no build stamp and is older than "):
        _wrapper(tmp_path).Solver()
    with pytest.warns(UserWarning, match="has no build stamp and is older than "):
        _wrapper(tmp_path).Solver(allow_unverified=True)

    subprocess.run(["make", "lib"], cwd=tmp_path, check=True, capture_output=True)
    _load_silently(tmp_path)


@pytest.mark.requires_compiler
def test_a_stamp_of_another_library_counts_as_no_stamp(tmp_path, requires_compiler):
    _generate(tmp_path)
    build_lib(tmp_path)
    _generate(tmp_path, max_iters=60)
    # Rebuilt by hand: the stamp describes the library that was replaced.
    subprocess.run(["make", "lib"], cwd=tmp_path, check=True, capture_output=True)
    _load_silently(tmp_path)


@pytest.mark.requires_compiler
@pytest.mark.filterwarnings(f"ignore:.*{NO_RECORD}")
def test_no_stamp_is_written_or_kept_without_a_record(tmp_path, requires_compiler):
    _generate(tmp_path)
    build_lib(tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()
    build_lib(tmp_path)  # no list of sources to stamp
    assert not (tmp_path / STAMP_NAME).exists()
    _wrapper(tmp_path).Solver()  # warned about the record; the library is not checked
