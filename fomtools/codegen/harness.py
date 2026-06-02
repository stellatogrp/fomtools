"""Build / verify harness for emitted solvers.

Thin wrappers around the target's build commands plus a compiler probe, so the
rest of the pipeline (and tests) need not shell out directly.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fomtools.codegen.targets import TargetDescriptor


def find_compiler() -> str | None:
    """Return an available C compiler executable, or ``None``."""
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@dataclass
class VerifyResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def build_and_verify(workdir: str | Path, target: TargetDescriptor) -> VerifyResult:
    """Run the target's verify command in ``workdir``."""
    proc = subprocess.run(
        target.verify_command(),
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    return VerifyResult(
        success=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def build_lib(workdir: str | Path, make_target: str = "lib") -> Path:
    """Build the shared library (``make lib``) and return its path."""
    workdir = Path(workdir)
    proc = subprocess.run(
        ["make", make_target],
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"library build failed:\n{proc.stdout}\n{proc.stderr}")
    for name in ("libfomsolver.dylib", "libfomsolver.so"):
        cand = workdir / name
        if cand.exists():
            return cand
    raise RuntimeError("library build reported success but no library file was found")
