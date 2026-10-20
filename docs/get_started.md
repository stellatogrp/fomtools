# Get started

Getting started takes two steps:

1. [Install the Python package](#install-the-package), which is all you need to
   verify, simulate, and generate code for first-order methods.
2. [Install the agent skill](#install-the-agent-skill), so that coding agents such
   as Claude Code, Codex, or Cursor know the FOMTools API when they write or
   review FOMTools code for you.

Then follow the [Quick start](quick_start.md).

## Install the package

FOMTools requires Python 3.12 or newer.

### From PyPI

Install the latest release from [PyPI](https://pypi.org/project/fomtools/):

```bash
pip install fomtools
```

This is all you need for verification and simulation with the default HiGHS
solver. Optional dependencies are installed as extras (see the table below),
for example:

```bash
pip install "fomtools[gurobi,codegen]"
```

To install every optional feature at once:

```bash
pip install "fomtools[all]"
```

`all` skips the GPU packages on macOS. Gurobi still needs a license, and code
generation still needs a C compiler and a signed-in coding agent.

To upgrade to a newer release:

```bash
pip install --upgrade fomtools
```

### From source

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

### Optional dependencies

Quote extras so shells such as zsh do not interpret the brackets. From PyPI use
`pip install "fomtools[<extra>]"`; from a source checkout use the commands below.

| Extra | Installation | Purpose |
|---|---|---|
| Gurobi | `uv pip install -e ".[gurobi]"` | Optional MIP/MIQP/MIQCP backend; a suitable Gurobi license is required |
| Code generation | `uv pip install -e ".[codegen]"` | Coding-agent SDKs; authentication is configured separately |
| CVXPYgen | `uv pip install -e ".[cvxpygen]"` | High-level problem-modeling integration |
| CUDA code generation | `uv pip install -e ".[cuda]"` | cuDSS for direct linear solves in the generated CUDA solver (Linux) |
| GPU | `uv pip install -e ".[gpu]"` | CuPy for CUDA simulation; GPU SDP additionally requires Julia, Clarabel.jl with CUDA, and juliacall |
| Neural networks | `uv pip install -e ".[nn]"` | PyTorch/JAX/Flax integrations |
| Everything | `uv pip install -e ".[all]"` | All of the above; the GPU packages are skipped on macOS |
| Development and docs | `uv pip install -e ".[dev,docs]"` | Tests, lint, and documentation tools; these extras also install Gurobi |

For Gurobi use `Verifier(solver="gurobi")`. For the
[experimental SDP relaxation](user_guide/sdp.md) use
`Verifier(method="sdp", solver="clarabel", obbt_solver=None)` or
`solver="cvxpy"`. The `cuclarabel` backend needs a CUDA-capable environment.


## Install the agent skill

FOMTools ships an [Agent Skill](https://agentskills.io/) in
[`skills/fomtools`](https://github.com/stellatogrp/fomtools/tree/main/skills/fomtools).
It gives AI coding agents a reference for the `Verifier` API, the verification
backends, code generation, and the CVXPYgen integration, so they write FOMTools
code that matches the current API. The skill uses the open `SKILL.md`
format and contains only instructions and references, no scripts or tools.

The [`skills` CLI](https://github.com/vercel-labs/skills) installs it into the
agents it finds on your machine. It needs Node.js; if `npx` is missing, install the
[LTS release of Node.js](https://nodejs.org/en/download) and open a new terminal.

```bash
npx skills add stellatogrp/fomtools --list   # show the skills in the repository
npx skills add stellatogrp/fomtools -g       # install globally, for every project
```

The installer asks which agents to install the skill for. Drop `-g` to install it
only in the current project. To install it by hand instead, copy the
`skills/fomtools` folder into your agent's skills directory, for example
`~/.claude/skills/fomtools` for Claude Code.

The skill follows the documentation on `main`. To update it, run

```bash
npx skills@latest update fomtools -g
```

and start a new agent session so it loads the new instructions. As with any
skill, read its [`SKILL.md`](https://github.com/stellatogrp/fomtools/blob/main/skills/fomtools/SKILL.md)
before installing it.

## Check the installation

```bash
python -c "import fomtools; print(fomtools.__version__)"
```

Then run the [Quick start](quick_start.md) example, which uses the default HiGHS
solver and needs no license. To check the skill, ask your agent a FOMTools
question in a new session, for example "How do I verify ISTA with FOMTools?".

## Development

Run the tests with `uv run --no-sync pytest` and build the documentation with
`uv run --no-sync mkdocs build`. Some tests require optional dependencies,
a compiler, or a Gurobi license. Real coding-agent tests are opt-in.
