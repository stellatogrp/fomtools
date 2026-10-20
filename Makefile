.PHONY: docs-export docs-export-force docs-serve docs-build

docs-export:
	uv run --extra docs python scripts/export_marimo.py

docs-export-force:
	uv run --extra docs python scripts/export_marimo.py --force

docs-serve: docs-export
	uv run --extra docs mkdocs serve

docs-build: docs-export-force
	uv run --extra docs mkdocs build --strict
