# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, pi, and others) that
*develop* this repository. To learn how to *use* FOMTools (the Verifier API,
verification, simulation, code generation, CVXPYgen), read the installable skill
in [`skills/fomtools/`](skills/fomtools/SKILL.md) and the docs in `docs/`. Do
not duplicate that material here.

## Project

FOMTools is a Python package for developing, analyzing, and deploying
first-order methods for parametric linear and convex quadratic programs. The
user records an iteration with the `Verifier`; the package represents it as a
graph of affine and piecewise affine steps and uses that one description to

- verify worst-case performance with an exact mixed-integer formulation
  (`method="mip"`, HiGHS by default, Gurobi optional);
- simulate on sampled inputs with KL-inverse probabilistic bounds
  (`method="simulator"`, NumPy or CuPy);
- generate C, OpenMP, or CUDA solvers, written by an LLM coding agent and
  checked against a NumPy reference execution (`fomtools/codegen/`);
- generate solvers for parametric CVXPY problems through CVXPYgen
  (`fomtools/cvxpygen/`).

An experimental SDP relaxation (`method="sdp"`) also exists.

## Development commands

```bash
uv venv && uv pip install -e ".[dev]"   # dev install (adds Gurobi, pytest, ruff, pre-commit)
uv pip install -e ".[codegen,cvxpygen]" # extras for code generation and CVXPYgen
pytest                                  # all tests
pytest tests/codegen                    # code generation and CVXPYgen tests
ruff check && ruff format               # lint and format (line length 100, py312)
pre-commit run --all-files              # hooks: whitespace, YAML, ruff, uv-lock
mkdocs serve                            # docs at http://127.0.0.1:8000
```

Other extras: `gurobi`, `gpu` (CuPy), `cuda` (cuDSS for the CUDA target),
`nn`, `docs`. CI (`.github/workflows/ci.yml`) builds the wheel, installs it
without extras, and checks the import, the `fomtools-codegen` CLI, and the
HiGHS default.

## Layout

| Path | Contents |
|---|---|
| `fomtools/verifier.py` | `Verifier`: variables, bounds, steps, objectives, `solve`, `codegen` |
| `fomtools/linearexpression.py` | affine expressions `Ay + b` and interval bounds |
| `fomtools/steps/`, `fomtools/proxable_functions/`, `fomtools/sets/` | PWA steps, `prox`/`project` functions, sets |
| `fomtools/stepdecomposers/` | per-backend decomposition of steps (MIP, SDP, simulator) |
| `fomtools/canonicalizers/` | MIP, SDP, and simulator backends; OBBT in `obbt_handler.py` |
| `fomtools/solvers/` | thin wrappers for HiGHS, Gurobi, Clarabel, CuClarabel |
| `fomtools/codegen/` | extraction (`extract*.py`, `ir.py`), factorizations, NumPy oracle, targets (`targets/`), agent drivers (`agent/`), CLI |
| `fomtools/cvxpygen/` | CVXPYgen front end: canonical QP data, builders (`admm`, `pdhg`), scaling |
| `fomtools/nn/` | neural-network layers for verification experiments |
| `tests/`, `tests/codegen/` | unit and end-to-end tests; `tests/codegen/golden/` holds fixed solver loops that tests use in place of an agent |
| `docs/` | MkDocs site and tutorials |
| `skills/fomtools/` | installable agent skill for *users*; kept in sync by `.github/workflows/update-skill.yml` |

## Code generation rules

These invariants are checked by tests; keep them when editing `fomtools/codegen/`.

- Never form or use a matrix inverse; solve with factorizations or CG.
- The embedded C target is allocation-free and division-free in the iteration.
- Every generated solver must pass the numerical acceptance check against the
  NumPy oracle (`oracle.py`); the agent may only edit the files a target marks
  editable, and the package restores the rest before each check.
- Agent backends are pluggable: SDK drivers in `agent/`, any headless CLI agent
  through `agent/agents.yaml`. `fomtools/codegen/agent/stub.py` lets the
  pipeline and tests run without a model.

Useful environment variables: `FOMTOOLS_CUDA_RUN=1|0` forces or disables GPU
execution of the CUDA checks; `FOMTOOLS_CODEGEN_BACKEND` selects the agent.

## Conventions

- Python 3.12+, `ruff` for lint and format, NumPy-style docstrings.
- Prefer small diffs that fit the existing architecture; add a test with each
  behavior change.
- When the public API or docs change, the `update-skill` workflow refreshes
  `skills/fomtools/`; check its output rather than editing the skill by hand
  in the same PR.
