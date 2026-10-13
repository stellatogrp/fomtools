# FOMTools

FOMTools develops, analyzes, and generates implementations of first-order
optimization methods from a shared mathematical specification. It offers MIP
verification, experimental SDP relaxations, batched simulation, and checked
code generation for C, OpenMP, and CUDA targets.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

HiGHS is the default MIP solver. Gurobi is optional:
`uv pip install -e ".[gurobi]"`; using it requires a suitable license.
For development use `uv pip install -e ".[dev,docs]"`.
See [installation](docs/installation.md) for other extras.

## Start here

- [Executable Quick Start and solver contract](docs/index.md)
- [API reference](docs/api.md)
- [Tutorials](docs/tutorials.md)
- [Code generation](docs/api/codegen.md)

MIP solves return best upper bounds, not incumbent objectives. SDP solves reject
inaccurate or early termination. Numerical solver tolerances and sampled code
checks are distinct from exact-arithmetic certificates.

Run `uv run --no-sync pytest` and `uv run --no-sync ruff check` for local checks.
