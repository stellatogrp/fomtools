import os
import subprocess
from pathlib import Path

import pytest

from scripts import export_marimo


def _write_notebook(path: Path) -> None:
    path.write_text("import marimo\napp = marimo.App()\n")


def test_discover_notebooks_selects_marimo_files_recursively(tmp_path):
    tutorials = tmp_path / "tutorials"
    tutorials.mkdir()
    first = tutorials / "a_notebook.py"
    second = tutorials / "b_notebook.py"
    _write_notebook(first)
    _write_notebook(second)
    (tutorials / "helper.py").write_text("def helper():\n    pass\n")
    nested = tutorials / "nested"
    nested.mkdir()
    nested_notebook = nested / "nested_notebook.py"
    _write_notebook(nested_notebook)

    assert export_marimo.discover_notebooks(tutorials) == [
        first,
        second,
        nested_notebook,
    ]


def test_needs_export_detects_missing_stale_and_forced_outputs(tmp_path):
    source = tmp_path / "tutorial.py"
    timing_data = tmp_path / "_recorded_timings.json"
    output = tmp_path / "tutorial.html"
    source.write_text("source")
    timing_data.write_text("{}")

    assert export_marimo.needs_export(source, output, dependencies=(timing_data,))

    output.write_text("html")
    os.utime(source, ns=(1, 1))
    os.utime(timing_data, ns=(2, 2))
    os.utime(output, ns=(3, 3))
    assert not export_marimo.needs_export(source, output, dependencies=(timing_data,))

    os.utime(timing_data, ns=(4, 4))
    assert export_marimo.needs_export(source, output, dependencies=(timing_data,))
    assert export_marimo.needs_export(source, output, force=True)


def test_export_tutorials_generates_only_missing_or_stale_files(tmp_path):
    repository_root = tmp_path
    tutorials = repository_root / "docs" / "tutorials"
    tutorials.mkdir(parents=True)
    stale = tutorials / "stale.py"
    current = tutorials / "current.py"
    _write_notebook(stale)
    _write_notebook(current)
    timing_data = tutorials / "_recorded_timings.json"
    timing_data.write_text("{}")
    current_output = current.with_suffix(".html")
    current_output.write_text("current")

    os.utime(stale, ns=(1, 1))
    os.utime(current, ns=(1, 1))
    os.utime(timing_data, ns=(1, 1))
    os.utime(current_output, ns=(2, 2))

    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("-o") + 1])
        output.write_text("generated")
        return subprocess.CompletedProcess(command, 0, stdout="solver output", stderr="")

    exported, skipped = export_marimo.export_tutorials(
        repository_root=repository_root,
        runner=runner,
    )

    assert exported == [stale]
    assert skipped == [current]
    assert stale.with_suffix(".html").read_text() == "generated"
    command, kwargs = calls[0]
    assert command[:5] == [
        export_marimo.sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
    ]
    assert command[-1] == "-f"
    assert kwargs == {
        "cwd": repository_root,
        "capture_output": True,
        "text": True,
        "check": False,
    }

    forced, skipped = export_marimo.export_tutorials(
        repository_root=repository_root,
        force=True,
        runner=runner,
    )
    assert forced == [current, stale]
    assert skipped == []


def test_export_notebook_reports_command_output_and_license_guidance(tmp_path):
    source = tmp_path / "tutorial.py"
    output = tmp_path / "tutorial.html"
    _write_notebook(source)

    def failing_runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="license checkout failed",
            stderr="export aborted",
        )

    with pytest.raises(export_marimo.TutorialExportError) as exc_info:
        export_marimo.export_notebook(
            source,
            output,
            repository_root=tmp_path,
            runner=failing_runner,
        )

    message = str(exc_info.value)
    assert str(source) in message
    assert "valid Gurobi license" in message
    assert "license checkout failed" in message
    assert "export aborted" in message


def test_export_notebook_requires_the_requested_output(tmp_path):
    source = tmp_path / "tutorial.py"
    output = tmp_path / "tutorial.html"
    _write_notebook(source)

    def runner_without_output(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(
        export_marimo.TutorialExportError,
        match="did not create the requested HTML file",
    ):
        export_marimo.export_notebook(
            source,
            output,
            repository_root=tmp_path,
            runner=runner_without_output,
        )


def test_main_forwards_force_and_prints_summary(monkeypatch, capsys):
    calls = []

    def fake_export_tutorials(*, force=False):
        calls.append(force)
        return [Path("first.py"), Path("second.py")], [Path("current.py")]

    monkeypatch.setattr(export_marimo, "export_tutorials", fake_export_tutorials)

    assert export_marimo.main(["--force"]) == 0
    assert calls == [True]
    assert capsys.readouterr().out == ("Marimo tutorials: exported 2; already current 1.\n")


def test_export_tutorials_treats_helper_modules_as_dependencies(tmp_path):
    tutorials = tmp_path / "docs" / "tutorials"
    helpers = tutorials / "helper_functions"
    helpers.mkdir(parents=True)
    notebook = tutorials / "notebook.py"
    _write_notebook(notebook)
    helper = helpers / "helper.py"
    helper.write_text("def helper():\n    pass\n")
    output = notebook.with_suffix(".html")
    output.write_text("html")
    os.utime(notebook, ns=(1, 1))
    os.utime(output, ns=(2, 2))
    os.utime(helper, ns=(3, 3))

    def runner(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text("generated")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    exported, skipped = export_marimo.export_tutorials(repository_root=tmp_path, runner=runner)

    assert exported == [notebook]
    assert skipped == []


def test_main_reports_export_errors_without_traceback(monkeypatch, capsys):
    def failing_export_tutorials(*, force=False):
        raise export_marimo.TutorialExportError("Failed to export marimo tutorial 'x.py'")

    monkeypatch.setattr(export_marimo, "export_tutorials", failing_export_tutorials)

    assert export_marimo.main([]) == 1
    assert "Failed to export marimo tutorial 'x.py'" in capsys.readouterr().err
