"""The setup check: `fomtools.codegen.doctor` and ``fomtools-codegen --doctor``."""

import contextlib
import importlib.metadata
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fomtools.codegen import diagnostics, doctor
from fomtools.codegen.agent import cli_specs
from fomtools.codegen.cli import main
from fomtools.codegen.diagnostics import MISSING, OK, OPTIONAL, WARN, format_report, print_report
from fomtools.codegen.harness import find_compiler
from fomtools.codegen.proc import TIMEOUT_RETURNCODE, run_check
from fomtools.codegen.targets import large_scale_cuda

EVERYTHING = {"cc", "make", "nvcc", "nvidia-smi", "claude", "codex", "pi", "cursor-agent"}
EVERYTHING |= {"bwrap", "socat"}  # the sandbox of claude-sandboxed on Linux


class Machine:
    """A stand-in for the machine: which tools are on PATH and how they answer."""

    def __init__(self, monkeypatch):
        self.tools = set(EVERYTHING)
        self.openmp = True
        self.gpu = True
        self.cudss = "/opt/cudss"
        self.returncodes = {}  # tool -> exit status of its --version
        self.raises = set()  # tools whose --version cannot be started
        self.calls = []
        self.version = "{tool} 1.2.3\nmore\n"
        for var in ("CC", "FOMTOOLS_CODEGEN_BACKEND", "FOMTOOLS_CUDA_RUN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(shutil, "which", self.which)
        monkeypatch.setattr(diagnostics, "run_check", self.run_check)
        monkeypatch.setattr(
            large_scale_cuda, "_gpu_devices", lambda: ("GPU 0: Fake (UUID: GPU-0)",) * self.gpu
        )
        monkeypatch.setattr(large_scale_cuda, "_cudss_home", lambda: self.cudss)

    def which(self, name, *args, **kwargs):
        return f"/fake/bin/{name}" if name in self.tools else None

    def run_check(self, cmd, cwd, timeout_s=None, stdin=None):
        assert stdin == subprocess.DEVNULL  # a tool that asks a question gets no answer
        self.calls.append(cmd)
        tool = Path(cmd[0]).name
        if "openmp_probe.c" in cmd:
            assert Path(cwd, "openmp_probe.c").is_file()
            assert {"-O3", "-flto", "-fopenmp", "-lm"} <= set(cmd)  # the target's build line
            err = "" if self.openmp else "cc: error: unsupported option '-fopenmp'\n"
            return subprocess.CompletedProcess(cmd, 0 if self.openmp else 1, "", err)
        assert cmd[1:] == ["--version"]
        if tool in self.raises:
            raise PermissionError(13, "Permission denied")
        code = self.returncodes.get(tool, 0)
        return subprocess.CompletedProcess(
            cmd, code, self.version.format(tool=tool) if not code else "", ""
        )


@pytest.fixture
def machine(monkeypatch):
    return Machine(monkeypatch)


def test_everything_present(machine):
    report = doctor()
    assert report.ok
    assert all(c.status == OK for c in report.checks if c.group != "Python")
    assert report.targets == {
        "embedded C": True,
        "large-scale OpenMP": True,
        "large-scale CUDA": True,
    }
    assert report.get("C compiler").detail == "cc 1.2.3  (/fake/bin/cc)"
    assert report.get("make").detail == "make 1.2.3  (/fake/bin/make)"
    assert report.get("nvcc").detail == "nvcc 1.2.3  (/fake/bin/nvcc)"
    assert report.get("cuDSS").detail == "/opt/cudss"
    assert {"claude", "claude-sandboxed", "codex", "codex-sandboxed", "pi", "cursor"} <= set(
        report.agents
    )
    assert report.get("embedded C").detail == "ready"
    assert report.get("GPU").detail == "GPU 0: Fake (UUID: GPU-0)"
    # No model call, no login check: only `--version` and the one test build.
    assert all(c[1:] == ["--version"] or "openmp_probe.c" in c for c in machine.calls)
    # claude and claude-sandboxed share a CLI, asked for its version once.
    assert machine.calls.count(["/fake/bin/claude", "--version"]) == 1


@pytest.mark.parametrize("tool", ["cc", "make"])
def test_no_compiler_or_no_make_is_the_only_failure(machine, tool):
    machine.tools.discard(tool)
    report = doctor()
    name = "C compiler" if tool == "cc" else "make"
    assert not report.ok
    assert report.get(name).status == MISSING and "not found (required)" in report.get(name).detail
    assert report.targets == {
        "embedded C": False,
        "large-scale OpenMP": False,
        "large-scale CUDA": False,
    }
    assert report.get("embedded C").detail == f"cannot run: no {name}"
    text = format_report(report)
    assert f"✗ {name}" in text and "Code generation cannot run here" in text
    assert diagnostics.main(stream=io.StringIO()) == 1


def test_openmp_cuda_and_agents_are_optional(machine):
    machine.tools = {"cc", "make"}
    machine.openmp, machine.gpu, machine.cudss = False, False, ""
    report = doctor()
    assert report.ok and diagnostics.main(stream=io.StringIO()) == 0
    for name in ("OpenMP", "nvcc", "GPU", "cuDSS", "claude", "cursor"):
        assert report.get(name).status == OPTIONAL
    assert "unsupported option '-fopenmp'" in report.get("OpenMP").detail
    for name in ("nvcc", "GPU", "cuDSS"):
        assert (
            report.get(name).detail.startswith("not found") and "(needed" in report.get(name).detail
        )
        assert "CUDA target" in report.get(name).detail
    assert "Install Claude Code" in report.get("claude").detail
    assert report.agents == []
    assert report.targets == {
        "embedded C": True,
        "large-scale OpenMP": True,
        "large-scale CUDA": False,
    }
    assert "no agent CLI found" in report.get("embedded C").detail
    assert "single-threaded build only" in report.get("large-scale OpenMP").detail
    assert report.get("large-scale CUDA").detail == "cannot run: no nvcc"


def test_cuda_without_a_gpu_or_cudss_is_a_compile_check(machine):
    machine.gpu, machine.cudss = False, ""
    cuda = doctor().get("large-scale CUDA")
    assert cuda.status == OK
    assert "compile check only: no GPU" in cuda.detail and "no cuDSS" in cuda.detail


def test_gpu_is_reported_without_nvcc(machine):
    # The target's own probe wants nvcc too; the GPU line must not.
    machine.tools.discard("nvcc")
    report = doctor()
    assert report.get("GPU").status == OK and report.get("nvcc").status == OPTIONAL
    assert report.get("large-scale CUDA").detail == "cannot run: no nvcc"


@pytest.mark.parametrize(
    "value, gpu, note",
    [("1", False, "checks run on a GPU"), ("0", True, "compile check only")],
)
def test_cuda_summary_follows_the_run_override(machine, monkeypatch, value, gpu, note):
    monkeypatch.setenv("FOMTOOLS_CUDA_RUN", value)
    machine.gpu = gpu
    detail = doctor().get("large-scale CUDA").detail
    assert detail == f"ready ({note}: FOMTOOLS_CUDA_RUN={value})"
    assert large_scale_cuda.CudaTarget._can_run_on_gpu() is (value == "1")


def test_compiler_is_the_one_make_uses(machine, monkeypatch):
    # The generated Makefiles build with $CC, else cc; so do the doctor and find_compiler.
    machine.tools |= {"gcc-15", "gcc"}
    monkeypatch.setenv("CC", "gcc-15 -m64")
    report = doctor()
    assert find_compiler() == "/fake/bin/gcc-15"
    assert report.get("C compiler").detail == "gcc-15 1.2.3  (/fake/bin/gcc-15)  (CC=gcc-15 -m64)"
    probe = next(c for c in machine.calls if "openmp_probe.c" in c)
    assert probe[:2] == ["/fake/bin/gcc-15", "-m64"]

    monkeypatch.setenv("CC", "no-such-cc")  # cc is on PATH, but make would not use it
    report = doctor()
    assert find_compiler() is None and not report.ok
    assert report.get("C compiler").detail.startswith("CC=no-such-cc not found (required")

    monkeypatch.delenv("CC")
    machine.tools.discard("cc")  # gcc alone: a bare `make` would call cc, so fomtools sets CC
    report = doctor()
    detail = report.get("C compiler").detail
    assert find_compiler() == "/fake/bin/gcc" and report.ok
    assert detail.startswith("gcc 1.2.3  (/fake/bin/gcc)  (fallback: no `cc` on PATH")
    assert "CC=/fake/bin/gcc" in detail
    assert next(c for c in machine.calls[::-1] if "openmp_probe.c" in c)[0] == "/fake/bin/gcc"
    assert report.targets["embedded C"]

    machine.tools.discard("gcc")
    report = doctor()
    assert find_compiler() is None and not report.ok
    assert report.get("C compiler").detail == "not found (required): no `cc`, `gcc`, `clang`"


def test_openmp_probe_that_times_out_or_cannot_start(machine, monkeypatch):
    def slow(cmd, cwd, timeout_s=None, **kwargs):
        if "openmp_probe.c" in cmd:
            return subprocess.CompletedProcess(cmd, TIMEOUT_RETURNCODE, "", "note")
        return machine.run_check(cmd, cwd, timeout_s, **kwargs)

    monkeypatch.setattr(diagnostics, "run_check", slow)
    check = doctor().get("OpenMP")
    assert check.status == OPTIONAL and "did not finish" in check.detail

    def gone(cmd, cwd, timeout_s=None, **kwargs):
        if "openmp_probe.c" in cmd:
            raise FileNotFoundError(2, "No such file or directory")
        return machine.run_check(cmd, cwd, timeout_s, **kwargs)

    monkeypatch.setattr(diagnostics, "run_check", gone)
    check = doctor().get("OpenMP")
    assert check.status == OPTIONAL and "No such file" in check.detail


def test_a_version_that_times_out_or_fails_is_reported_not_fatal(machine):
    machine.returncodes = {"claude": TIMEOUT_RETURNCODE, "codex": 3, "make": 2}
    machine.raises = {"pi"}
    report = doctor()
    assert report.ok
    assert (
        report.get("claude").status == WARN and "did not return in" in report.get("claude").detail
    )
    assert "/fake/bin/claude" in report.get("claude").detail
    assert report.get("codex").status == WARN and "exit status 3" in report.get("codex").detail
    assert report.get("pi").status == WARN and "Permission denied" in report.get("pi").detail
    assert report.get("make").status == WARN and report.targets["embedded C"]
    # Still on PATH, so still usable agents.
    assert {"claude", "codex", "pi"} <= set(report.agents)
    assert "! claude" in format_report(report)


def test_login_is_not_checked_and_the_report_says_so(machine):
    report = doctor()
    assert report.get("codex").detail == "codex 1.2.3  (/fake/bin/codex)"
    assert "Agent logins are not checked" in format_report(report).splitlines()[-1]


def test_sandboxed_claude_needs_bubblewrap_and_socat_on_linux(machine, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    machine.tools.discard("socat")
    report = doctor()
    check = report.get("claude-sandboxed")
    assert check.status == OPTIONAL and "its sandbox cannot start: no socat" in check.detail
    assert "claude-sandboxed" not in report.agents and "claude" in report.agents
    assert report.get("codex-sandboxed").status == OK  # codex brings its own sandbox
    machine.tools.add("socat")
    assert doctor().get("claude-sandboxed").status == OK
    machine.tools -= {"bwrap", "socat"}
    monkeypatch.setattr(sys, "platform", "darwin")  # Seatbelt is part of macOS
    assert doctor().get("claude-sandboxed").status == OK


def test_targets_say_when_the_default_agent_cannot_run(machine):
    machine.tools.discard("claude")
    detail = doctor().get("embedded C").detail
    assert detail == "ready, but not with the default agent (claude): pass --agent"


def test_default_backend_is_marked_and_follows_the_env_var(machine, monkeypatch):
    report = doctor()
    assert report.default_backend == "claude"
    assert "claude (default) " in format_report(report)
    assert "codex (default)" not in format_report(report)

    monkeypatch.setenv("FOMTOOLS_CODEGEN_BACKEND", "codex-sandboxed")
    report = doctor()
    assert report.default_backend == "codex-sandboxed"
    assert "codex-sandboxed (default)" in format_report(report)
    assert "claude (default)" not in format_report(report)
    assert doctor(backend="pi").default_backend == "pi"  # an explicit backend wins

    monkeypatch.setenv("FOMTOOLS_CODEGEN_BACKEND", "nope")
    report = doctor()
    assert report.get("nope").status == OPTIONAL and "nope" not in report.agents
    assert "not a registered backend" in report.get("nope").detail


@pytest.fixture
def user_agents(tmp_path, monkeypatch):
    """Load the agents from a user's own agents.yaml, on top of the bundled ones."""

    def load(text):
        path = tmp_path / "agents.yaml"
        path.write_text(text)
        monkeypatch.setenv("FOMTOOLS_CODEGEN_AGENTS", str(path))
        monkeypatch.setattr(cli_specs, "_REGISTRY", {})
        monkeypatch.setattr(cli_specs, "_user_specs_loaded", False)

    return load


def test_user_defined_agents_are_listed(machine, user_agents):
    user_agents(
        "agents:\n"
        "  aider:\n    bin: aider\n    auth_hint: Set the provider key.\n"
        "  mine:\n    bin: my-agent\n"
    )
    machine.tools.add("aider")
    report = doctor()
    assert report.get("aider").status == OK
    assert report.get("aider").detail == "aider 1.2.3  (/fake/bin/aider)"
    assert report.get("mine").status == OPTIONAL
    assert report.get("mine").detail == "`my-agent` not found. put `my-agent` on PATH"
    assert "aider" in report.agents and "claude" in report.agents and "mine" not in report.agents


def test_a_malformed_agents_file_is_reported_not_raised(machine, user_agents):
    user_agents("agents:\n  broken:\n    flags: []\n")
    report = doctor()
    assert report.ok and report.agents == []
    assert "missing required key 'bin'" in report.get("agents.yaml").detail
    assert "missing required key 'bin'" in doctor().get("agents.yaml").detail  # every time
    # The command line still builds its parser, so --doctor can report the file.
    out = io.StringIO()
    with pytest.raises(SystemExit) as exit_, contextlib.redirect_stdout(out):
        main(["--doctor"])
    assert exit_.value.code == 0 and "agents.yaml" in out.getvalue()
    with pytest.raises(SystemExit) as exit_, contextlib.redirect_stdout(out):
        main(["--help"])
    assert exit_.value.code == 0


def test_runs_without_the_optional_packages(machine, monkeypatch):
    # The `package` job of CI installs the wheel with no extras.
    def version(dist):
        if dist != "fomtools":
            raise importlib.metadata.PackageNotFoundError(dist)
        return "1.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    report = doctor()
    assert report.ok and report.get("fomtools").detail == "1.0"
    assert report.get("extra: codegen").status == OK
    for extra in ("cvxpygen", "gpu", "cuda"):
        check = report.get(f"extra: {extra}")
        assert check.status == OPTIONAL and f"fomtools[{extra}]" in check.detail


def test_an_extra_with_a_package_missing_is_not_installed(machine, monkeypatch):
    def version(dist):
        if dist not in ("fomtools", "osqp", "cupy"):
            raise importlib.metadata.PackageNotFoundError(dist)
        return "1.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    report = doctor()
    check = report.get("extra: cvxpygen")  # needs cvxpygen and osqp
    assert check.status == OPTIONAL and check.detail.startswith("osqp 1.0, but not cvxpygen (")
    assert report.get("extra: gpu").status == OK  # any one CuPy distribution


def test_optional_packages_are_never_imported(tmp_path):
    # In a fresh interpreter where importing an optional package fails, as it
    # does when the package is not installed.
    code = (
        "import sys\n"
        "optional = ('cvxpygen', 'osqp', 'cupy', 'gurobipy', 'torch', 'jax')\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.partition('.')[0] in optional:\n"
        "            raise ImportError(name + ' is blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "from fomtools.codegen.cli import main\n"
        "main(['--doctor'])\n"
    )
    env = {"PATH": str(tmp_path), "NO_COLOR": "1", "HOME": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    # An empty PATH: no compiler, so the exit status is 1, with the full report.
    assert proc.returncode == 1, proc.stderr
    assert "C compiler" in proc.stdout and "not found (required)" in proc.stdout
    assert "Traceback" not in proc.stderr


# --- the formatter -----------------------------------------------------------


class Tty(io.StringIO):
    def isatty(self):
        return True


class Latin1(io.StringIO):
    encoding = "latin-1"


def test_colors_only_on_a_terminal_and_never_with_no_color(machine, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    report = doctor()
    plain = io.StringIO()
    print_report(report, plain)
    assert "\033[" not in plain.getvalue()
    assert plain.getvalue() == format_report(report) == str(report)

    tty = Tty()
    print_report(report, tty)
    assert "\033[32m✓\033[0m" in tty.getvalue()

    monkeypatch.setenv("NO_COLOR", "1")
    tty = Tty()
    print_report(report, tty)
    assert "\033[" not in tty.getvalue() and tty.getvalue() == plain.getvalue()


def test_ascii_marks_when_the_encoding_lacks_the_glyphs(machine):
    machine.tools.discard("nvcc")
    out = Latin1()
    print_report(doctor(), out)
    text = out.getvalue()
    text.encode("latin-1")
    assert "[ok] make" in text and "[--] nvcc" in text and "✓" not in text


def test_home_directory_is_written_as_a_tilde(machine, monkeypatch):
    home = str(Path.home())
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"{home}/bin/{name}")
    machine.version = "{tool} 1.2.3 in " + home + "/lib\n"  # a tool that prints its path
    machine.cudss = home + "/cudss"
    report = doctor()
    assert report.get("make").detail == "make 1.2.3 in ~/lib  (~/bin/make)"
    assert home not in str(report)


def test_odd_version_output_is_cleaned_and_always_printable(machine):
    machine.version = "\n\x1b[1mtöol\x1b[0m " + "9" * 200 + "\n"
    report = doctor()
    assert report.get("make").detail == "[1mtöol[0m " + "9" * 69 + "  (/fake/bin/make)"
    out = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    print_report(report, out)  # "ö" is not ASCII: replaced, not raised
    assert b"t?ol" in out.buffer.getvalue()


def test_tool_output_that_is_not_text_does_not_raise(tmp_path):
    proc = run_check(
        [sys.executable, "-c", "import os; os.write(1, b'\\xff\\xfe v1')"], tmp_path, 60
    )
    assert proc.returncode == 0 and proc.stdout.endswith(" v1")


# --- the command line --------------------------------------------------------


def test_cli_doctor_needs_no_other_argument(machine, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["--doctor"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert "✓ C compiler" in out and "Targets" in out and "claude (default)" in out


@pytest.mark.parametrize("argv", [["--agent", "codex", "--doctor"], ["--doctor", "--agent=codex"]])
def test_cli_doctor_exit_status_and_agent(machine, capsys, argv):
    machine.tools.discard("cc")
    with pytest.raises(SystemExit) as exit_:
        main(argv)
    assert exit_.value.code == 1
    out = capsys.readouterr().out
    assert "✗ C compiler" in out and "codex (default)" in out


def test_cli_without_doctor_is_unchanged(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_:
        main([])  # the source and --out are still required
    assert exit_.value.code == 2
    assert "required" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_:
        main(["--help"])
    assert exit_.value.code == 0
    assert "--doctor" in capsys.readouterr().out
    assert main(["--example", "ista", "--out", str(tmp_path / "ista"), "--scaffold-only"]) == 0
    assert (tmp_path / "ista" / "SPEC.md").exists()


def test_missing_cli_messages_point_to_the_doctor(monkeypatch):
    from fomtools.codegen.agent import AgentBackendError, get_default_driver, make_agent_driver

    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    with pytest.raises(AgentBackendError, match="fomtools-codegen --doctor"):
        make_agent_driver("claude").run(None)
    monkeypatch.delenv("FOMTOOLS_CODEGEN_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="fomtools-codegen --doctor"):
        get_default_driver()


# --- the real machine --------------------------------------------------------


def test_smoke_on_this_machine():
    compiler = find_compiler()
    if compiler is None:
        pytest.skip("no C compiler available")
    report = doctor()
    assert report.get("C compiler").found
    assert set(report.targets) == {"embedded C", "large-scale OpenMP", "large-scale CUDA"}
    assert str(report)
