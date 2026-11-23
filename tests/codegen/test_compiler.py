"""Which C compiler the package builds with, and that ``make`` gets the same one."""

import os
import shutil
from types import SimpleNamespace

import pytest

from fomtools.codegen.agent import make_agent_driver
from fomtools.codegen.diagnostics import OK, doctor
from fomtools.codegen.harness import build_and_verify, find_compiler
from fomtools.codegen.proc import build_env, fallback_compiler, popen_in_session, run_check

MAKE = shutil.which("make")
#: ``make`` reports the compiler it would build with, as the generated Makefiles pick it.
# (`true &&` has make run the line with /bin/sh, whose `echo` needs no PATH.)
MAKEFILE = "CC ?= cc\nverify:\n\t@true && echo CC=$(CC)\n"


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    """Put the named stub executables on an otherwise empty PATH; return their directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CC", raising=False)

    def install(*names):
        for name in names:
            stub = bin_dir / name
            stub.write_text(f"#!/bin/sh\necho '{name} 1.2.3'\n")
            stub.chmod(0o755)
        return bin_dir

    return install


def test_cc_env_wins(fake_path, monkeypatch):
    bin_dir = fake_path("cc", "gcc", "clang", "gcc-15")
    monkeypatch.setenv("CC", "gcc-15 -m64")
    assert find_compiler() == str(bin_dir / "gcc-15")
    assert fallback_compiler() is None and build_env() is None


def test_cc_env_that_does_not_resolve_gets_no_fallback(fake_path, monkeypatch):
    fake_path("cc", "gcc", "clang")
    monkeypatch.setenv("CC", "no-such-cc")  # make would build with it, and fail
    assert find_compiler() is None
    assert fallback_compiler() is None and build_env() is None


def test_cc_on_path_is_used_before_gcc_and_clang(fake_path):
    bin_dir = fake_path("cc", "gcc", "clang")
    assert find_compiler() == str(bin_dir / "cc")
    assert fallback_compiler() is None and build_env() is None
    assert build_env({"A": "1"}) == {"A": "1"}


@pytest.mark.parametrize(
    "installed, expected", [(("gcc", "clang"), "gcc"), (("gcc",), "gcc"), (("clang",), "clang")]
)
def test_fallback_without_cc(fake_path, installed, expected):
    bin_dir = fake_path(*installed)
    path = str(bin_dir / expected)
    assert find_compiler() == fallback_compiler() == path
    assert build_env()["CC"] == path and build_env()["PATH"] == os.environ["PATH"]
    assert build_env({"A": "1"}) == {"A": "1", "CC": path}


def test_no_compiler(fake_path):
    fake_path("make")
    assert find_compiler() is None
    assert fallback_compiler() is None and build_env() is None


def test_doctor_reports_the_fallback(fake_path):
    bin_dir = fake_path("clang", "make")
    report = doctor()
    check = report.get("C compiler")
    assert report.ok and check.status == OK
    assert check.detail.startswith(f"clang 1.2.3  ({bin_dir}/clang)  (fallback: no `cc` on PATH")
    assert f"CC={bin_dir}/clang" in check.detail

    (bin_dir / "cc").symlink_to(bin_dir / "clang")
    assert "fallback" not in doctor().get("C compiler").detail


@pytest.fixture
def make_dir(fake_path, tmp_path):
    """A directory whose Makefile prints its compiler; the real ``make`` joins the stubs."""

    def install(*names):
        bin_dir = fake_path(*names)
        (bin_dir / "make").write_text(f'#!/bin/sh\nexec {MAKE} "$@"\n')
        (bin_dir / "make").chmod(0o755)
        (tmp_path / "Makefile").write_text(MAKEFILE)
        return bin_dir

    return install


def _make_target():
    """A target whose verify command is ``make verify``, as every bundled target's is."""
    config = SimpleNamespace(check_timeout_s=60)
    return SimpleNamespace(verify_command=lambda: ["make", "verify"], config=config)


@pytest.mark.skipif(MAKE is None, reason="needs make")
@pytest.mark.parametrize("installed", [("gcc",), ("clang",), ("gcc", "clang")])
def test_make_gets_the_fallback(make_dir, tmp_path, installed):
    bin_dir = make_dir(*installed)
    result = build_and_verify(tmp_path, _make_target())
    assert result.success, result.output
    assert result.stdout.strip() == f"CC={bin_dir / installed[0]}" == f"CC={find_compiler()}"


@pytest.mark.skipif(MAKE is None, reason="needs make")
def test_make_is_left_alone_without_a_fallback(make_dir, tmp_path, monkeypatch):
    make_dir("cc", "gcc", "gcc-15")
    assert run_check(["make", "verify"], tmp_path, 60).stdout.strip() == "CC=cc"
    monkeypatch.setenv("CC", "gcc-15 -m64")
    assert run_check(["make", "verify"], tmp_path, 60).stdout.strip() == "CC=gcc-15 -m64"


def test_agent_cli_gets_the_fallback(fake_path, tmp_path):
    # The agent runs `make verify` itself: it must build with the compiler the package uses.
    bin_dir = fake_path("gcc")
    agent = bin_dir / "agent"
    agent.write_text('#!/bin/sh\necho "CC=$CC DEADLINE=$FOMTOOLS_DEADLINE"\n')
    agent.chmod(0o755)
    env = make_agent_driver("claude")._child_env(123)
    proc = popen_in_session([str(agent)], cwd=tmp_path, env=env)
    assert proc.communicate(timeout=60)[0].strip() == f"CC={bin_dir}/gcc DEADLINE=123"
