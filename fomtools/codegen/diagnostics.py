"""Check what a code generation run needs on this machine (``fomtools-codegen --doctor``).

`doctor` looks for the C compiler, ``make``, OpenMP, the CUDA toolchain and the
agent CLIs, and returns a `DoctorReport`: one `Check` per item and one per
target. `print_report` writes it with the marks of the progress output.

What is run: each program with ``--version``, one small OpenMP build in a
temporary directory, and ``nvidia-smi -L``. Nothing calls a model, so an agent's
login is not checked. The optional packages are looked up by their installed
metadata and never imported.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from fomtools.codegen.proc import (
    FALLBACK_COMPILERS,
    TIMEOUT_RETURNCODE,
    fallback_compiler,
    find_compiler,
    run_check,
)

#: Statuses of a `Check`.
OK = "ok"  # found
WARN = "warn"  # found, but its version could not be read
OPTIONAL = "optional"  # not found or not usable; only some targets or features need it
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


def _short(text: str) -> str:
    """``text`` with the home directory written as ``~``, wherever it occurs."""
    home = os.path.expanduser("~")
    return text.replace(home, "~") if len(home) > 1 else text


def _run(cmd: list[str], cwd: str, timeout_s: float) -> subprocess.CompletedProcess:
    """`run_check` with no standard input, so a tool that asks a question gets no answer."""
    return run_check(cmd, cwd, timeout_s, stdin=subprocess.DEVNULL)


def _version(path: str, cwd: str) -> tuple[str, str]:
    """``(version line, problem)`` from running ``path --version``; one of them is empty."""
    try:
        proc = _run([path, "--version"], cwd, VERSION_TIMEOUT_S)
    except OSError as exc:
        return "", f"`--version` could not run ({exc.strerror or exc})"
    if proc.returncode == TIMEOUT_RETURNCODE:
        return "", f"`--version` did not return in {VERSION_TIMEOUT_S:g} s"
    if proc.returncode != 0:
        return "", f"`--version` failed (exit status {proc.returncode})"
    for line in (proc.stdout or proc.stderr).splitlines():
        line = "".join(ch for ch in line if ch.isprintable()).strip()
        if line:
            return line[:80], ""
    return "", "`--version` printed nothing"


def _tool(group: str, name: str, path: str | None, cwd: str, missing: str, status: str) -> Check:
    """The check of an executable: its version and path, or ``missing``."""
    if path is None:
        return Check(group, name, status, missing)
    version, problem = _version(path, cwd)
    if problem:
        return Check(group, name, WARN, f"{path}  ({problem})")
    return Check(group, name, OK, f"{version}  ({path})")


def _extra(name: str, dists: tuple[str, ...], missing: str, need=all) -> Check:
    """The check of an extra, from the installed metadata; nothing is imported.

    ``need`` is `all` when every distribution is needed, `any` when one is enough.
    """
    found = {}
    for dist in dists:
        try:
            found[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    have = ", ".join(f"{dist} {version}" for dist, version in found.items())
    if found and need(dist in found for dist in dists):
        return Check("Python", name, OK, have)
    lacking = ", ".join(dist for dist in dists if dist not in found)
    detail = f"{have}, but not {lacking}" if found else "not installed"
    return Check("Python", name, OPTIONAL, f"{detail} ({missing})")


def _python_checks() -> list[Check]:
    try:
        fomtools_version = importlib.metadata.version("fomtools")
    except importlib.metadata.PackageNotFoundError:
        fomtools_version = "unknown version"
    return [
        Check("Python", "python", OK, f"{platform.python_version()}  ({sys.executable})"),
        Check("Python", "fomtools", OK, fomtools_version),
        Check("Python", "extra: codegen", OK, "no Python packages needed"),
        _extra(
            "extra: cvxpygen",
            ("cvxpygen", "osqp"),
            'needed for parametric CVXPY problems: pip install "fomtools[cvxpygen]"',
        ),
        _extra(
            "extra: gpu",
            ("cupy-cuda12x", "cupy"),
            'needed to simulate on a GPU: pip install "fomtools[gpu]"',
            need=any,
        ),
        _extra(
            "extra: cuda",
            ("nvidia-cudss-cu13",),
            'cuDSS for the CUDA target, Linux only: pip install "fomtools[cuda]"',
        ),
    ]


def _compiler_check(compiler: str | None, cwd: str) -> Check:
    """The compiler `find_compiler` returns: ``$CC``, else ``cc``, else a fallback."""
    cc = " ".join(os.environ.get("CC", "").split())
    if compiler is not None:
        check = _tool("Build tools", "C compiler", compiler, cwd, "", MISSING)
        note = f"  (CC={cc})" if cc else ""
        if fallback_compiler():
            note = (
                f"  (fallback: no `cc` on PATH; fomtools runs make with CC={compiler}, "
                "and so must a `make` you run yourself)"
            )
        return replace(check, detail=check.detail + note)
    if cc:
        detail = f"CC={cc} not found (required: the generated Makefiles build with $CC)"
        return Check("Build tools", "C compiler", MISSING, detail)
    names = ", ".join(f"`{name}`" for name in ("cc", *FALLBACK_COMPILERS))
    return Check("Build tools", "C compiler", MISSING, f"not found (required): no {names}")


def _openmp_check(compiler: str | None, cwd: str) -> Check:
    """Whether ``compiler`` builds and links an OpenMP loop with the target's flags."""
    from fomtools.codegen.targets.large_scale_openmp import OPENMP_FLAG

    serial = "the OpenMP target then builds single-threaded"
    if compiler is None:
        return Check("Build tools", "OpenMP", OPTIONAL, "not checked: no C compiler")
    Path(cwd, "openmp_probe.c").write_text(_OPENMP_PROBE)
    # The compile and link line of the target's Makefile, with OMPFLAGS set.
    flags = [*os.environ.get("CC", "").split()[1:], "-O3", "-Wall", "-flto", OPENMP_FLAG]
    cmd = [compiler, *flags, "openmp_probe.c", "-o", "openmp_probe", "-lm"]
    try:
        proc = _run(cmd, cwd, BUILD_TIMEOUT_S)
    except OSError as exc:
        reason = exc.strerror or str(exc)
    else:
        if proc.returncode == 0:
            return Check("Build tools", "OpenMP", OK, f"{compiler} builds with {OPENMP_FLAG}")
        errors = [line.strip() for line in proc.stderr.splitlines() if "error" in line.lower()]
        reason = errors[0][:120] if errors else f"exit status {proc.returncode}"
        if proc.returncode == TIMEOUT_RETURNCODE:
            reason = f"the test build did not finish in {BUILD_TIMEOUT_S:g} s"
    detail = f"{compiler} does not build with {OPENMP_FLAG} ({reason}); {serial}"
    if sys.platform == "darwin":
        detail += ". Apple clang has no OpenMP: for threads set CC to a GCC, e.g. CC=gcc-15"
    return Check("Build tools", "OpenMP", OPTIONAL, detail)


def _cuda_checks(cwd: str) -> list[Check]:
    from fomtools.codegen.targets import large_scale_cuda as cuda

    needed = "needed for the CUDA target"
    nvcc = _tool("CUDA", "nvcc", shutil.which("nvcc"), cwd, f"not found ({needed})", OPTIONAL)
    devices = cuda._gpu_devices()  # not `_gpu_visible`, which also wants nvcc
    if devices:
        more = f" (and {len(devices) - 1} more)" if len(devices) > 1 else ""
        gpu = Check("CUDA", "GPU", OK, f"{devices[0][:80]}{more}")
    else:
        gpu = Check(
            "CUDA",
            "GPU",
            OPTIONAL,
            "not found by nvidia-smi (needed to run the CUDA target's checks; "
            "without one they only compile)",
        )
    home = cuda._cudss_home()
    if home:
        cudss = Check("CUDA", "cuDSS", OK, home)
    else:
        cudss = Check(
            "CUDA",
            "cuDSS",
            OPTIONAL,
            "not found (needed for the CUDA target's direct linear solves: "
            "pip install nvidia-cudss-cu13, or set CUDSS_HOME)",
        )
    return [nvcc, gpu, cudss]


def _sandbox_needs(spec) -> list[str]:
    """The programs the sandbox of a Claude Code profile needs and that are not on PATH.

    Only on Linux, where Claude Code's Bash sandbox runs on bubblewrap and socat,
    and only for a spec whose ``--settings`` turn the sandbox on.
    """
    if not sys.platform.startswith("linux"):
        return []
    for flag in spec.flags:
        try:
            sandbox = json.loads(flag).get("sandbox") or {}
        except (ValueError, AttributeError):
            continue
        if isinstance(sandbox, dict) and sandbox.get("enabled"):
            return [tool for tool in ("bwrap", "socat") if shutil.which(tool) is None]
    return []


def _agent_checks(default: str, cwd: str) -> list[Check]:
    """One check per registered agent backend: its CLI on PATH and its version."""
    from fomtools.codegen.agent import available_backends
    from fomtools.codegen.agent.cli_specs import get_spec

    try:
        names = available_backends()
    except Exception as exc:  # noqa: BLE001 - a malformed agents.yaml is reported, not raised
        problem = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
        detail = (
            f"could not be loaded, so no agent can run ({problem}). Check "
            "~/.config/fomtools/agents.yaml and $FOMTOOLS_CODEGEN_AGENTS"
        )
        return [Check("Agents", "agents.yaml", OPTIONAL, detail)]
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
        status = WARN if problem else OK
        detail = f"{version or path}  ({problem or path})"
        lacking = _sandbox_needs(spec)
        if lacking:
            status = OPTIONAL
            detail += (
                f"; its sandbox cannot start: no {', no '.join(lacking)} "
                "(install bubblewrap and socat)"
            )
        checks.append(Check("Agents", name, status, detail))
    if default not in names:
        detail = f"not a registered backend; choose from {', '.join(names)}"
        checks.append(Check("Agents", default, OPTIONAL, detail))
    return checks


def _target_checks(checks: list[Check], default: str) -> list[Check]:
    """The summary: which targets can run here, and whether an agent can write them."""
    from fomtools.codegen.targets.large_scale_cuda import CudaTarget

    found = {c.name: c.found for c in checks}
    agents = [c.name for c in checks if c.group == "Agents" and c.found]
    if default in agents:
        with_agents = ""
    elif agents:
        with_agents = f", but not with the default agent ({default}): pass --agent"
    else:
        with_agents = "; no agent CLI found, so only `--scaffold-only` and a stub driver work"
    lacking = [name for name in ("C compiler", "make") if not found[name]]

    def target(name: str, needs: list[str], note: str = "") -> Check:
        if needs:
            status = MISSING if name == "embedded C" else OPTIONAL
            return Check("Targets", name, status, "cannot run: no " + ", no ".join(needs))
        return Check("Targets", name, OK, f"ready{note}{with_agents}")

    openmp = "" if found["OpenMP"] else " (single-threaded build only)"
    # What `CudaTarget.verify_command` decides: FOMTOOLS_CUDA_RUN, else a GPU.
    override = os.environ.get("FOMTOOLS_CUDA_RUN")
    if override is not None:
        on_gpu = "checks run on a GPU" if CudaTarget._can_run_on_gpu() else "compile check only"
        cuda = f" ({on_gpu}: FOMTOOLS_CUDA_RUN={override})"
    else:
        cuda = "" if found["GPU"] else " (compile check only: no GPU)"
    cuda += "" if found["cuDSS"] else " (no cuDSS: a direct linear solve does not build)"
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
    from fomtools.codegen.agent import _DEFAULT_BACKEND

    default = backend or os.environ.get("FOMTOOLS_CODEGEN_BACKEND") or _DEFAULT_BACKEND
    with tempfile.TemporaryDirectory(prefix="fomtools-doctor-") as cwd:
        compiler = find_compiler()
        checks = _python_checks()
        checks += [
            _compiler_check(compiler, cwd),
            _tool(
                "Build tools", "make", shutil.which("make"), cwd, "not found (required)", MISSING
            ),
            _openmp_check(compiler, cwd),
        ]
        checks += _cuda_checks(cwd)
        checks += _agent_checks(default, cwd)
    checks += _target_checks(checks, default)
    # No home directory in the report, whichever tool or error message wrote it.
    return DoctorReport(tuple(replace(c, detail=_short(c.detail)) for c in checks), default)


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
        lines += [
            "",
            "Agent logins are not checked: that would call a model. A run that cannot "
            "authenticate stops and says how to log in.",
        ]
    else:
        lines += [
            "",
            "Code generation cannot run here: it needs a C compiler and make (see above).",
        ]
    return "\n".join(lines) + "\n"


def print_report(report: DoctorReport, stream=None) -> None:
    """Write the report to ``stream`` (default: standard output).

    Colors only on a terminal and never with ``NO_COLOR`` set, as the progress
    output. ASCII marks when the stream's encoding lacks the glyphs.
    """
    from fomtools.codegen.progress import Progress

    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "".join(mark for mark, _ in _MARKS.values()).encode(encoding)
        ascii_only = False
    except (UnicodeEncodeError, LookupError):
        ascii_only = True
    text = format_report(report, Progress(True, stream).color, ascii_only)
    try:  # a version string or a path may still hold a character the stream lacks
        text = text.encode(encoding, "replace").decode(encoding)
    except LookupError:
        pass
    stream.write(text)
    stream.flush()


def main(backend: str | None = None, stream=None) -> int:
    """Run `doctor`, print the report, and return the exit status (1 if nothing can run)."""
    report = doctor(backend)
    print_report(report, stream)
    return 0 if report.ok else 1
