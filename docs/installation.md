# Installation

FOMTools requires Python 3.12 or newer.

## From PyPI

Install the latest release from [PyPI](https://pypi.org/project/fomtools/):

```bash
pip install fomtools
```

Optional dependencies are installed as extras (see the table below), for example:

```bash
pip install "fomtools[gurobi,codegen]"
```

To upgrade to a newer release and check the installed version:

```bash
pip install --upgrade fomtools
python -c "import fomtools; print(fomtools.__version__)"
```

## From source

To work from source, use `uv` to create an environment:

```bash
git clone https://github.com/stellatogrp/fomtools.git
cd fomtools
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

To reproduce the exact dependency versions we develop against, use the checked-in
`uv.lock` instead (ask for Python 3.12 explicitly, otherwise `uv` picks the newest
interpreter it finds):

```bash
uv sync --python 3.12 --extra dev --extra docs
```

The base package includes HiGHS, the default MIP solver, and CVXPY with Clarabel
for CPU SDP solves. It does not require a Gurobi license.

## Optional dependencies

Quote extras so shells such as zsh do not interpret the brackets. From PyPI use
`pip install "fomtools[<extra>]"`; from a source checkout use the commands below.

| Extra | Installation | Purpose |
|---|---|---|
| Gurobi | `uv pip install -e ".[gurobi]"` | Optional MIP/MIQP/MIQCP backend; a suitable Gurobi license is required |
| Code generation | `uv pip install -e ".[codegen]"` | Coding-agent SDKs; authentication is configured separately |
| CVXPYgen | `uv pip install -e ".[cvxpygen]"` | High-level problem-modeling integration |
| GPU | `uv pip install -e ".[gpu]"` | CuPy for CUDA simulation; GPU SDP additionally requires Julia, Clarabel.jl with CUDA, and juliacall |
| Neural networks | `uv pip install -e ".[nn]"` | PyTorch/JAX/Flax integrations |
| Development and docs | `uv pip install -e ".[dev,docs]"` | Tests, lint, and documentation tools; these extras also install Gurobi |

For Gurobi use `Verifier(solver="gurobi")`. For an SDP use
`Verifier(method="sdp", solver="clarabel", obbt_solver=None)` or
`solver="cvxpy"`. The `cuclarabel` backend needs a CUDA-capable environment.

Run the tests with `uv run --no-sync pytest` and build the documentation with
`uv run --no-sync mkdocs build`. Some tests require optional dependencies,
a compiler, or a Gurobi license. Real coding-agent tests are opt-in.
