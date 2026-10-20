# Contributing to FOMTools

## Development setup

Follow the environment instructions in [docs/get_started.md](docs/get_started.md) to create a
virtual environment and install `fomtools` in editable mode with its `dev` and `docs` extras.

Install the pre-commit hooks once per clone:

    pre-commit install

## Tests and linting

    pytest                      # Run the test suite
    ruff check                  # Check code style
    ruff format                 # Auto-format
    pre-commit run --all-files  # Run all hooks

## Previewing the documentation

Install the documentation dependencies and start the local MkDocs server:

    uv sync --extra docs
    make docs-serve

The `docs-serve` target recursively discovers marimo notebooks under `docs/tutorials`, exports
missing or stale notebooks to ignored HTML files, and then starts MkDocs. Restart the server after
editing a notebook so that its HTML is refreshed.

Use these targets to export or validate the documentation without starting the server:

    make docs-export        # Export only missing or stale notebooks
    make docs-export-force  # Regenerate every notebook
    make docs-build         # Force export, then run a strict MkDocs build
