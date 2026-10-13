# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FOMTools is a Python toolbox for verification of first-order optimization methods. It provides a framework for formally verifying convergence properties of algorithms like ISTA, Douglas-Rachford, and other first-order methods using convex optimization techniques.

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_verifier.py  # Run specific test file
pytest -v                 # Verbose output
```

### Linting and Code Quality
```bash
ruff check               # Check code style and formatting
ruff format              # Auto-format code
pre-commit run --all-files  # Run all pre-commit hooks
```

### Documentation
```bash
mkdocs serve            # Serve documentation locally
mkdocs build            # Build documentation
```

### Installation
```bash
pip install -e .        # Install in development mode
pip install -e .[dev]   # Install with development dependencies
pip install -e .[docs]  # Install with documentation dependencies
```

## Architecture

### Core Components

**Verifier** (`fomtools/verifier.py`): The main orchestrator class that:
- Manages optimization variables and their bounds
- Coordinates between canonicalizers and OBBT handlers
- Implements step operations (ReLU, soft threshold, saturated linear)
- Performs interval bound propagation
- Interfaces with solvers through canonicalizers

**LinearExpression** (`fomtools/linearexpression.py`): Represents linear combinations of function values and supports:
- Decomposition into coefficient matrices
- Arithmetic operations
- Vector-like behavior with @ operator support

**Steps** (`fomtools/steps/`): Implement specific algorithm operations:
- `AffineStep`: Linear transformations
- `ReluStep`: ReLU activation functions
- `SaturatedLinearStep`: Saturated linear operations
- `SoftThresholdStep`: Soft thresholding operations

**Canonicalizers** (`fomtools/canonicalizers/`): Convert verification problems to solver formats:
- `GurobiCanonicalizer`: Interfaces with Gurobi optimizer
- `SDPCanonicalizer`: For semidefinite programming formulations
- Support for adding constraints and objectives

**OBBT Handlers** (`fomtools/canonicalizers/obbt_handlers/`): Optimization-Based Bound Tightening:
- `GurobiOBBTHandler`: Tightens variable bounds using optimization

### Key Design Patterns

- **Vector abstraction**: The `Vector` class represents optimization variables
- **Expression trees**: `LinearExpression` builds computational graphs
- **Bound propagation**: Interval arithmetic for bound tightening
- **Solver abstraction**: Canonicalizers provide solver-agnostic interface

### Dependencies

- Core: `cvxpy`, `jax`, `numpy`
- Optimization: `gurobipy` (optional, for commercial solver)
- Development: `pytest`, `ruff`, `pre-commit`
- Documentation: `mkdocs`, `mkdocs-material`

## Configuration

- **Ruff**: Configured in `pyproject.toml` with line length 100, targets Python 3.12
- **Pre-commit**: Available for code quality enforcement
- **MkDocs**: Material theme with automatic dark/light mode switching

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

Norms always carry a subscript: `\|v\|_2`, `\|v\|_\infty` (never `\lVert v \rVert` or a bare `\|v\|`).
Define soft-thresholding where it is first used: `\mathbf{soft}_\kappa(v)_i = \mathbf{sign}(v_i)\max\{|v_i| - \kappa, 0\}`, applied element-wise.

Symbol roles: `x` problem parameter in `\mathcal{X}`; `z` decision variable and primal iterate, `z^0 \in \mathcal{Z}`, `z^\star` solution; `v, w` dual or auxiliary iterates; `y` (lasso) or `w` (NNLS) for the un-extrapolated proximal output of a momentum method, matching the helper code; `k` iteration, `K` horizon, `\eta` step size, `L` smoothness constant, `r^k` residual, `R` PEP initial radius.
Search-box bounds are `x_\mathrm{lb}, x_\mathrm{ub}` and `z^0_\mathrm{lb}, z^0_\mathrm{ub}`; problem-data bounds keep `_{\min}, _{\max}`. Code names such as `x_lb` or `s0_samples` appear only in backticks in prose. When a math letter differs from the code name, say so once.

Iterations are written `k \to k+1` for `k = 0, 1, \dots`. A momentum coefficient `\beta_k` uses the same index as the helper code (slot `k` uses `beta_k`, and `k` starts at 1), so the update reads `z^{k+1} = w^{k+1} + \beta_{k+1}(w^{k+1} - w^k)` with `\beta_1 = 0`. Use "Nesterov-type momentum" for `\beta_k = (k-1)/(k+2)` and reserve "FISTA" for the `t_k` schedule `\beta_k = (t_{k-1}-1)/t_k`.

Known clashes kept on purpose because both uses are standard: control input `u_t` vs. OSQP upper bound `u(x)`, LQR weight `R` vs. PEP radius `R`, and `\mu` as network cost, portfolio return, or strong-convexity constant.
