Before you can use the `fomtools` package, ensure that your system meets the following prerequisites:

* Python 3.12 or later
* Pip package manager
* A [Gurobi](https://www.gurobi.com) license. `gurobipy` is a required dependency, and the
  size-limited license bundled with the PyPI package is too small for most verification
  problems. Academic licenses are [free](https://www.gurobi.com/academia/academic-program-and-licenses/).

Clone the repository and enter it:

    git clone git@github.com:stellatogrp/fomtools.git
    cd fomtools

## Creating a fomtools environment
All of `fomtools`'s dependencies are on [PyPi](https://pypi.org). We have developed and tested
`fomtools` on Python 3.12, but it should work on later Python versions as well. You can create a new
virtual environment using `venv`, and install dependencies using `pip`.

1. Verify that Python 3.12 or newer is installed on your system.

        python --version

2. Create a new environment and activate it.

        python -m venv .venv
        source .venv/bin/activate

3. In the activated environment, install the `fomtools` package in editable mode, along with
its dev and docs extras:

        pip install -e ".[dev,docs]"

    Quote the argument and leave no space after the comma, otherwise your shell will either
    split the extras across two arguments or expand the brackets as a glob.

    Installing the `dev` extra is not optional in practice: `fomtools` imports `cvxpy`, which is
    currently declared only in the extras, so a bare `pip install -e .` leaves the package
    unimportable.

## Optional extras
Beyond `dev` and `docs`, the following extras are available and can be combined in the same
way, for example `pip install -e ".[dev,docs,nn]"`:

| Extra      | Purpose                                                                      |
| ---------- | ---------------------------------------------------------------------------- |
| `dev`      | Test and lint tooling (`pytest`, `ruff`, `pre-commit`), plus `cvxpy`.        |
| `docs`     | Building this documentation (`mkdocs`, `mkdocstrings`, `marimo`).            |
| `nn`       | Neural network layers in `fomtools.nn` (`torch`, `jax`, `flax`).             |
| `gpu`      | CuPy backend for GPU simulation (`cupy-cuda12x`); needs CUDA 12.             |
| `codegen`  | Running the LLM coding agent behind `fomtools-codegen`.                      |
| `cvxpygen` | The `fomtools.cvxpygen` integration (`cvxpy`, `cvxpygen`, `osqp`).           |

## Using a different Python version than the system default
If your Python version is not 3.12 or later, or if  you're getting errors when using a non-tested
Python version, we recommend using the [uv](https://github.com/astral-sh/uv) tool to create a virtual
environment with the correct Python version. `uv` is quick to [install](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
and easy to use, both locally as well as on research clusters.

Once `uv` is installed, the repository's checked-in `uv.lock` gives you the exact dependency
versions we develop against. Ask for Python 3.12 explicitly, since the project only requires
`>=3.12` and `uv` otherwise picks the newest interpreter it can find:

    uv sync --python 3.12 --extra dev --extra docs

This downloads Python 3.12 if needed, creates `.venv`, and installs `fomtools` in editable mode
together with the locked dependency versions. Activate it with

    source .venv/bin/activate

or prefix commands with `uv run` (for example `uv run pytest`) to skip activation entirely.

If you would rather resolve dependencies freshly instead of using the lockfile, create the
environment and install with `pip` as above:

1. Create a new environment with Python 3.12 and activate it.

        uv venv --python 3.12
        source .venv/bin/activate

2. In the activated environment, install the `fomtools` package in editable mode, along with
   its dev and docs extras:

        pip install -e ".[dev,docs]"
