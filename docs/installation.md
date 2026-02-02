## Installation
Before you can use the `fomtools` package, ensure that your system meets the following prerequisites:

* Python 3.12 or later
* Pip package manager

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

        pip install -e .[dev, docs]

## Using a different Python version than the system default
If your Python version is not 3.12 or later, or if  you're getting errors when using a non-tested
Python version, we recommend using the [uv](https://github.com/astral-sh/uv) tool to create a virtual
environment with the correct Python version. `uv` is quick to [install](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
and easy to use, both locally as well as on research clusters.

Once `uv` is installed.

1. Create a new environment with Python 3.12 and activate it.

        uv venv --python 3.12
        source .venv/bin/activate

2. In the activated environment, install the `fomtools` package in editable mode, along with
   its dev and docs extras:

        pip install -e .[dev, docs]


