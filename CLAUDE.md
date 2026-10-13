# Repository guide

FOMTools represents first-order iterations as affine expressions and steps.
It analyzes finite-iteration performance and generates implementations. Numerical
solver tolerances and sampled implementation checks do not constitute formal
floating-point proofs.

## Development

Use Python 3.12+ and `uv`:

```bash
uv venv
uv pip install -e ".[dev,docs]"
uv run --no-sync pytest
uv run --no-sync ruff check
uv run --no-sync mkdocs build
```

HiGHS is the default MIP backend. Gurobi is optional and needs a suitable license.
Some tests require a compiler or optional neural-network/codegen dependencies.
Real coding-agent tests are opt-in via `FOMTOOLS_CODEGEN_AGENT=1`.

## Architecture

- `Verifier` coordinates variables, objectives, steps, proxes, and projections.
- `LinearExpression` stores affine combinations of leaf variables plus a constant.
  Multiplication is scalar; use `M @ expr` for linear maps.
- `StepDecomposer` converts composite steps/proxes into backend primitives.
- `MIP`, `SDP`, and `Simulator` are canonicalizers. Solver-specific status,
  parameters, and bound access belong in `fomtools/solvers/`.
- `OBBTHandler` maintains a separate LP relaxation for bound tightening.
- `fomtools/codegen/` extracts an iteration, generates deterministic scaffolding,
  and checks translated/optimized code on sampled inputs.

MIP `solve()` returns the best upper bound, including after limits when usable.
Never substitute an incumbent for it. SDP rejects inaccurate or early termination.
Keep Python, main-solver, and OBBT bounds synchronized when tightening a leaf.
Simulator default residuals follow public outputs, excluding auxiliary variables.

The canonical executable Quick Start is `docs/index.md` and its regression is in
`tests/test_docs_examples.py`. Keep the public API, docs, and package skill aligned.
