"""Recursively export marimo tutorial notebooks for the documentation site."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS_DIRECTORY = REPOSITORY_ROOT / "docs" / "tutorials"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class TutorialExportError(RuntimeError):
    """Raised when a marimo tutorial cannot be exported."""


def is_marimo_notebook(path: Path) -> bool:
    """Return whether ``path`` is a marimo notebook source file."""
    return path.is_file() and b"app = marimo.App(" in path.read_bytes()


def discover_notebooks(tutorials_directory: Path = TUTORIALS_DIRECTORY) -> list[Path]:
    """Discover marimo notebooks recursively in a deterministic order."""
    return sorted(
        path for path in tutorials_directory.rglob("*.py") if is_marimo_notebook(path)
    )


def needs_export(
    source: Path,
    output: Path,
    *,
    dependencies: Sequence[Path] = (),
    force: bool = False,
) -> bool:
    """Return whether an exported notebook is missing or stale."""
    if force or not output.is_file():
        return True

    output_mtime = output.stat().st_mtime_ns
    inputs = (source, *(path for path in dependencies if path.is_file()))
    return any(path.stat().st_mtime_ns > output_mtime for path in inputs)


def export_notebook(
    source: Path,
    output: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    runner: Runner = subprocess.run,
) -> None:
    """Execute and export one marimo notebook as HTML."""
    command = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        str(source),
        "-o",
        str(output),
        "-f",
    ]
    completed = runner(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and output.is_file():
        return

    details = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if details:
        details = f"\n\nmarimo output:\n{details}"
    reason = (
        "marimo exited successfully but did not create the requested HTML file"
        if completed.returncode == 0
        else f"marimo exited with status {completed.returncode}"
    )
    raise TutorialExportError(
        f"Failed to export marimo tutorial '{source}': {reason}. Ensure the docs extra is "
        "installed and a valid Gurobi license is available."
        f"{details}"
    )


def export_tutorials(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> tuple[list[Path], list[Path]]:
    """Export missing or stale tutorial notebooks and return exported/skipped sources."""
    tutorials_directory = repository_root / "docs" / "tutorials"
    recorded_timings = tutorials_directory / "_recorded_timings.json"
    dependencies = (recorded_timings,)
    exported: list[Path] = []
    skipped: list[Path] = []

    for source in discover_notebooks(tutorials_directory):
        output = source.with_suffix(".html")
        if not needs_export(source, output, dependencies=dependencies, force=force):
            skipped.append(source)
            continue
        export_notebook(source, output, repository_root=repository_root, runner=runner)
        exported.append(source)

    return exported, skipped


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recursive tutorial exporter from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate every notebook even when its HTML is current",
    )
    args = parser.parse_args(argv)
    exported, skipped = export_tutorials(force=args.force)
    print(f"Marimo tutorials: exported {len(exported)}; already current {len(skipped)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
