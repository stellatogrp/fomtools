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

## Tutorial math notation

All `mo.md` cells in `docs/tutorials/` (verification and codegen notebooks) use one notation per concept, following Boyd-style conventions. Keep new notebooks in sync with this table.

| Concept | Write | Never |
|---|---|---|
| Reals, ones | `\mathbf{R}`, `\mathbf{R}^n_+`, `\mathbf{1}` | `{\bf R}`, `\R`, `\mathbb{R}` |
| Transpose | `A^T` | `A^\top` |
| Calligraphic | `\mathcal{X}` | `{\cal X}` |
| Problem keywords | `\text{minimize}`, `\text{maximize}`, `\text{subject to}` | `\textnormal{...}` |
| Delimiters | inline `$...$`, display `$$...$$`, problems and updates as bare `\begin{array}{ll}`/`{rcl}` | `\(...\)`, `\[...\]` |
| Soft-thresholding | `\mathbf{soft}_{\lambda\eta}` (`soft_threshold` step) | `{\cal T}_\lambda`, `S_\lambda` |
| Projection | `\Pi_C`: `\Pi_{\mathbf{R}^n_+}` (`relu` step), `\Pi_{[l,u]}` (`satlin` step / `IndBox`) | `\mathbf{ReLU}`, `\mathbf{prox}_{\mathbb{I}_C}` |
| Proximal operator | `\mathbf{prox}_f` | `\operatorname{prox}` |
| Fractions | inline slash: `(1/2) z^T P z`, `(t_k - 1)/t_{k+1}` | `\frac` |

Norms always carry a subscript: `\|v\|_2`, `\|v\|_\infty` (never `\lVert v \rVert` or a bare `\|v\|`).
Define soft-thresholding where it is first used: `\mathbf{soft}_\kappa(v)_i = \mathbf{sign}(v_i)\max\{|v_i| - \kappa, 0\}`, applied element-wise.
Matplotlib labels (`label=`, `xlabel`, `ylabel`, `title`) are rendered by mathtext, not KaTeX: write `\geq`/`\leq` there (mathtext rejects `\ge`/`\le`), and keep `\|v\|_2` norms.

Symbol roles: `x` problem parameter in `\mathcal{X}`; `z` decision variable and primal iterate, `z^0 \in \mathcal{Z}`, `z^\star` solution; `v, w` dual or auxiliary iterates; `y` (lasso) or `w` (NNLS) for the un-extrapolated proximal output of a momentum method, matching the helper code; `k` iteration, `K` horizon, `\eta` step size, `L` smoothness constant, `r^k` residual, `R` PEP initial radius.
Search-box bounds are `x_\mathrm{lb}, x_\mathrm{ub}` and `z^0_\mathrm{lb}, z^0_\mathrm{ub}`; problem-data bounds keep `_{\min}, _{\max}`. Code names such as `x_lb` or `s0_samples` appear only in backticks in prose. When a math letter differs from the code name, say so once.

Iterations are written `k \to k+1` for `k = 0, 1, \dots`. A momentum coefficient `\beta_k` uses the same index as the helper code (slot `k` uses `beta_k`, and `k` starts at 1), so the update reads `z^{k+1} = w^{k+1} + \beta_{k+1}(w^{k+1} - w^k)` with `\beta_1 = 0`. Use "Nesterov-type momentum" for `\beta_k = (k-1)/(k+2)` and reserve "FISTA" for the `t_k` schedule `\beta_k = (t_{k-1}-1)/t_k`.

Known clashes kept on purpose because both uses are standard: control input `u_t` vs. OSQP upper bound `u(x)`, LQR weight `R` vs. PEP radius `R`, and `\mu` as network cost, portfolio return, or strong-convexity constant.
