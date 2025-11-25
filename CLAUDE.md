# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlgoVerify is a Python toolbox for verification of first-order optimization methods. It provides a framework for formally verifying convergence properties of algorithms like ISTA, Douglas-Rachford, and other first-order methods using convex optimization techniques.

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

**Verifier** (`algoverify/verifier.py`): The main orchestrator class that:
- Manages optimization variables and their bounds
- Coordinates between canonicalizers and OBBT handlers
- Implements step operations (ReLU, soft threshold, saturated linear)
- Performs interval bound propagation
- Interfaces with solvers through canonicalizers

**LinearExpression** (`algoverify/linearexpression.py`): Represents linear combinations of function values and supports:
- Decomposition into coefficient matrices
- Arithmetic operations
- Vector-like behavior with @ operator support

**Steps** (`algoverify/steps/`): Implement specific algorithm operations:
- `AffineStep`: Linear transformations
- `ReluStep`: ReLU activation functions
- `SaturatedLinearStep`: Saturated linear operations
- `SoftThresholdStep`: Soft thresholding operations

**Canonicalizers** (`algoverify/canonicalizers/`): Convert verification problems to solver formats:
- `GurobiCanonicalizer`: Interfaces with Gurobi optimizer
- `SDPCanonicalizer`: For semidefinite programming formulations
- Support for adding constraints and objectives

**OBBT Handlers** (`algoverify/canonicalizers/obbt_handlers/`): Optimization-Based Bound Tightening:
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
