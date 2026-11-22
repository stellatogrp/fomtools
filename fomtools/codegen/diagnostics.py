"""Check what a code generation run needs on this machine (``fomtools-codegen --doctor``).

`doctor` looks for the C compiler, ``make``, OpenMP, the CUDA toolchain and the
agent CLIs, and returns a `DoctorReport`: one `Check` per item and one per
target. `print_report` writes it with the marks of the progress output. Nothing
here calls a model or the network, so an agent's login is not checked.

Standard library only. The optional packages are looked up by their installed
metadata and never imported.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fomtools.codegen.proc import TIMEOUT_RETURNCODE, run_check

#: Statuses of a `Check`.
OK = "ok"  # found
WARN = "warn"  # found, but something about it could not be checked
OPTIONAL = "optional"  # not found; only some targets or features need it
MISSING = "missing"  # not found, and nothing can run without it

#: Limit, in seconds, of one ``--version`` call and of the OpenMP test build.
VERSION_TIMEOUT_S = 10.0
BUILD_TIMEOUT_S = 60.0

# A loop like the ones the OpenMP target annotates. It needs no header, and
# fails to compile when the flags do not turn OpenMP on.
_OPENMP_PROBE = """\
#ifndef _OPENMP
#error "compiled without OpenMP"
#endif
int main(void) {
    double s = 0.0;
#pragma omp parallel for reduction(+:s)
    for (int i = 0; i < 8; i++) s += i;
    return s == 28.0 ? 0 : 1;
}
"""

_MARKS = {OK: ("✓", "32"), WARN: ("!", "33"), OPTIONAL: ("○", "2"), MISSING: ("✗", "31")}
_ASCII_MARKS = {OK: "[ok]", WARN: "[!!]", OPTIONAL: "[--]", MISSING: "[XX]"}


@dataclass(frozen=True)
class Check:
    """One line of the report: what was looked for, and what was found."""

    group: str
    name: str
    status: str
    detail: str = ""

    @property
    def found(self) -> bool:
        return self.status in (OK, WARN)


@dataclass(frozen=True)
class DoctorReport:
    """The result of `doctor`; ``str(report)`` is the plain-text report."""

    checks: tuple[Check, ...]
    default_backend: str

    @property
    def ok(self) -> bool:
        """Whether anything can run: ``False`` without a C compiler or ``make``."""
        return not any(c.status == MISSING for c in self.checks if c.group != "Targets")

    @property
    def targets(self) -> dict[str, bool]:
        """Which targets can run here."""
        return {c.name: c.found for c in self.checks if c.group == "Targets"}

    @property
    def agents(self) -> list[str]:
        """The agent backends whose CLI is on PATH."""
        return [c.name for c in self.checks if c.group == "Agents" and c.found]

    def get(self, name: str) -> Check:
        """The check called ``name``."""
        return next(c for c in self.checks if c.name == name)

    def __str__(self) -> str:
        return format_report(self)


def _short(path: str) -> str:
    """``path`` with the home directory written as ``~``."""
    home = str(Path.home())
    return "~" + path[len(home) :] if path == home or path.startswith(home + os.sep) else path


def _version(path: str, cwd: str, flag: str = "--version") -> tuple[str, str]:
    """``(version line, problem)`` from running ``path --version``; one of them is empty."""
    try:
        proc = run_check([path, flag], cwd, VERSION_TIMEOUT_S)
    except OSError as exc:
        return "", f"`{flag}` could not run ({exc.strerror or exc})"
    if proc.returncode == TIMEOUT_RETURNCODE:
        return "", f"`{flag}` did not return in {VERSION_TIMEOUT_S:g} s"
    if proc.returncode != 0:
        return "", f"`{flag}` failed (exit status {proc.returncode})"
    lines = [line.strip() for line in (proc.stdout or proc.stderr).splitlines() if line.strip()]
    return (lines[0], "") if lines else ("", f"`{flag}` printed nothing")


def _tool(group: str, name: str, path: str | None, cwd: str, missing: str, status: str) -> Check:
    """The check of an executable: its version and path, or ``missing``."""
    if path is None:
        return Check(group, name, status, missing)
    version, problem = _version(path, cwd)
    if problem:
        return Check(group, name, WARN, f"{_short(path)}  ({problem})")
    return Check(group, name, OK, f"{version}  ({_short(path)})")


def _package(name: str, dists: tuple[str, ...], missing: str) -> Check:
    """The check of an extra: the versions of its distributions, without importing them."""
    found = []
    for dist in dists:
        try:
            found.append(f"{dist} {importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    if found:
        return Check("Python", name, OK, ", ".join(found))
    return Check("Python", name, OPTIONAL, f"not installed ({missing})")


def _python_checks() -> list[Check]:
    try:
        fomtools_version = importlib.metadata.version("fomtools")
    except importlib.metadata.PackageNotFoundError:
        fomtools_version = "unknown version"
    return [
        Check("Python", "python", OK, f"{platform.python_version()}  ({_short(sys.executable)})"),
        Check("Python", "fomtools", OK, fomtools_version),
        Check("Python", "extra: codegen", OK, "nothing to install"),
        _package(
            "extra: cvxpygen",
            ("cvxpygen", "osqp"),
            'needed for parametric CVXPY problems: pip install "fomtools[cvxpygen]"',
        ),
        _package(
            "extra: gpu",
            ("cupy-cuda12x", "cupy"),
            'needed to simulate on a GPU: pip install "fomtools[gpu]"',
        ),
        _package(
            "extra: cuda",
            ("nvidia-cudss-cu13",),
            'cuDSS for the CUDA target, Linux only: pip install "fomtools[cuda]"',
        ),
    ]


def _openmp_check(compiler: str | None, cwd: str) -> Check:
    """Whether ``compiler`` builds and links an OpenMP loop with the target's flag."""
    from fomtools.codegen.targets.large_scale_openmp import OPENMP_FLAG

    needed = "needed for a multithreaded build of the OpenMP target; the serial build works"
    if compiler is None:
        return Check("Build tools", "OpenMP", OPTIONAL, f"not checked: no C compiler ({needed})")
    Path(cwd, "openmp_probe.c").write_text(_OPENMP_PROBE)
    # The compile and link line of the target's Makefile, with OMPFLAGS set.
    cmd = [compiler, "-O3", "-Wall", OPENMP_FLAG, "openmp_probe.c", "-o", "openmp_probe", "-lm"]
    try:
        proc = run_check(cmd, cwd, BUILD_TIMEOUT_S)
    except OSError as exc:
        proc = None
        reason = exc.strerror or str(exc)
    if proc is not None and proc.returncode == 0:
        return Check("Build tools", "OpenMP", OK, f"{_short(compiler)} builds with {OPENMP_FLAG}")
    if proc is not None:
        errors = [line.strip() for line in proc.stderr.splitlines() if "error" in line.lower()]
        reason = errors[0][:120] if errors else f"exit status {proc.returncode}"
        if proc.returncode == TIMEOUT_RETURNCODE:
            reason = f"the test build did not finish in {BUILD_TIMEOUT_S:g} s"
    detail = f"{_short(compiler)} does not build with {OPENMP_FLAG}: {reason} ({needed}"
    if sys.platform == "darwin":
        detail += "; Apple clang has no OpenMP, set CC to a GCC, e.g. CC=gcc-15"
    return Check("Build tools", "OpenMP", OPTIONAL, detail + ")")


def _cuda_checks(cwd: str) -> list[Check]:
    from fomtools.codegen.targets import large_scale_cuda as cuda

    needed = "needed for the CUDA target"
    nvcc = _tool("CUDA", "nvcc", shutil.which("nvcc"), cwd, f"not found ({needed})", OPTIONAL)
    if cuda._gpu_visible():
        gpu = Check("CUDA", "GPU", OK, "nvidia-smi reports a device")
    else:
        gpu = Check(
            "CUDA",
            "GPU",
            OPTIONAL,
            "not found (needed to run the CUDA target's checks; without one they only compile)",
        )
    home = cuda._cudss_home()
    if home:
        cudss = Check("CUDA", "cuDSS", OK, _short(home))
    else:
        cudss = Check(
            "CUDA",
            "cuDSS",
            OPTIONAL,
            "not found (needed for the CUDA target's direct linear solves: "
            "pip install nvidia-cudss-cu13, or set CUDSS_HOME)",
        )
    return [nvcc, gpu, cudss]


def _default_backend(backend: str | None) -> str:
    from fomtools.codegen.agent import _DEFAULT_BACKEND

    return backend or os.environ.get("FOMTOOLS_CODEGEN_BACKEND") or _DEFAULT_BACKEND


def _agent_checks(default: str, cwd: str) -> list[Check]:
    """One check per registered agent backend: its CLI on PATH and its version."""
    from fomtools.codegen.agent import available_backends
    from fomtools.codegen.agent.cli_specs import get_spec

    try:
        names = available_backends()
    except Exception as exc:  # noqa: BLE001 - a malformed agents.yaml is reported, not raised
        return [Check("Agents", "agents.yaml", OPTIONAL, f"could not be loaded: {exc}")]
    checks, versions = [], {}
    for name in names:
        spec = get_spec(name)
        path = shutil.which(spec.bin)
        if path is None:
            hint = spec.install_hint or f"put `{spec.bin}` on PATH"
            checks.append(Check("Agents", name, OPTIONAL, f"`{spec.bin}` not found. {hint}"))
            continue
        if path not in versions:  # several backends share one CLI
            versions[path] = _version(path, cwd)
        version, problem = versions[path]
        login = "login not checked" + (f": {spec.auth_hint}" if spec.auth_hint else "")
        detail = f"{version or _short(path)}  ({problem or _short(path)}); {login}"
        checks.append(Check("Agents", name, WARN if problem else OK, detail))
    if default not in names:
        detail = f"not a registered backend; choose from {', '.join(names)}"
        checks.append(Check("Agents", default, OPTIONAL, detail))
    return checks


def _target_checks(checks: list[Check]) -> list[Check]:
    """The summary: which targets can run here, and with which agents."""
    found = {c.name: c.found for c in checks}
    agents = [c.name for c in checks if c.group == "Agents" and c.found]
    if agents:
        with_agents = "agents: " + ", ".join(agents)
    else:
        with_agents = "no agent CLI found, so only `--scaffold-only` and a stub driver work"
    lacking = [name for name in ("C compiler", "make") if not found[name]]

    def target(name: str, needs: list[str], note: str = "") -> Check:
        if needs:
            status = MISSING if name == "embedded C" else OPTIONAL
            return Check("Targets", name, status, "cannot run: no " + ", no ".join(needs))
        return Check("Targets", name, OK, f"ready{note}; {with_agents}")

    openmp = "" if found["OpenMP"] else " (serial build only)"
    cuda = "" if found["GPU"] else " (compile check only: no GPU)"
    cuda += "" if found["cuDSS"] else " (CG solves only: no cuDSS)"
    return [
        target("embedded C", lacking),
        target("large-scale OpenMP", lacking, openmp),
        target("large-scale CUDA", lacking + ([] if found["nvcc"] else ["nvcc"]), cuda),
    ]


def doctor(backend: str | None = None) -> DoctorReport:
    """Check this machine for what a code generation run needs.

    Parameters
    ----------
    backend:
        The agent backend to mark as the default. ``None`` reads
        ``FOMTOOLS_CODEGEN_BACKEND``, else ``"claude"``, as a run does.

    Returns
    -------
    DoctorReport
        One `Check` per item. ``report.ok`` is ``False`` only when nothing can
        run, that is without a C compiler or ``make``. Print it with
        `print_report`, or ``print(report)``.
    """
    from fomtools.codegen.harness import find_compiler

    default = _default_backend(backend)
    with tempfile.TemporaryDirectory(prefix="fomtools-doctor-") as cwd:
        # The generated Makefiles build with $CC when it is set, else with `cc`.
        compiler = shutil.which(os.environ.get("CC") or "") or find_compiler()
        checks = _python_checks()
        checks += [
            _tool("Build tools", "C compiler", compiler, cwd, "not found (required)", MISSING),
            _tool(
                "Build tools", "make", shutil.which("make"), cwd, "not found (required)", MISSING
            ),
            _openmp_check(compiler, cwd),
        ]
        checks += _cuda_checks(cwd)
        checks += _agent_checks(default, cwd)
    checks += _target_checks(checks)
    return DoctorReport(tuple(checks), default)


def format_report(report: DoctorReport, color: bool = False, ascii_only: bool = False) -> str:
    """The report as text: a mark, the name and the detail of each check, by group."""
    names = [
        c.name + " (default)" * (c.group == "Agents" and c.name == report.default_backend)
        for c in report.checks
    ]
    width = max(map(len, names))
    lines, group = [], None
    for check, name in zip(report.checks, names):
        if check.group != group:
            group = check.group
            title = f"\033[1m{group}\033[0m" if color else group
            lines += ["", title] if lines else [title]
        mark, code = _MARKS[check.status]
        if ascii_only:
            mark = _ASCII_MARKS[check.status]
        if color:
            mark = f"\033[{code}m{mark}\033[0m"
        lines.append(f"  {mark} {name:<{width}}  {check.detail}".rstrip())
    if report.ok:
        lines += ["", "Login of the agents is not checked: that would call a model."]
    else:
        lines += ["", "Code generation cannot run here: install a C compiler and make."]
    return "\n".join(lines) + "\n"


def print_report(report: DoctorReport, stream=None) -> None:
    """Write the report to ``stream`` (default: standard output).

    Colors only on a terminal and never with ``NO_COLOR`` set, as the progress
    output. ASCII marks when the stream's encoding lacks the glyphs.
    """
    from fomtools.codegen.progress import Progress

    stream = stream or sys.stdout
    try:
        "".join(mark for mark, _ in _MARKS.values()).encode(
            getattr(stream, "encoding", None) or "utf-8"
        )
        ascii_only = False
    except (UnicodeEncodeError, LookupError):
        ascii_only = True
    stream.write(format_report(report, Progress(True, stream).color, ascii_only))
    stream.flush()


def main(backend: str | None = None, stream=None) -> int:
    """Run `doctor`, print the report, and return the exit status (1 if nothing can run)."""
    report = doctor(backend)
    print_report(report, stream)
    return 0 if report.ok else 1
