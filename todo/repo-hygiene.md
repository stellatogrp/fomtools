# Repository hygiene TODOs

Scope: API-page coverage, mkdocs nav and configuration, the CI workflows and ignore rules. Prose fixes to existing pages are in [markdown-docs.md](markdown-docs.md).

Baseline: commit `d235580`. Claim an item by appending `(@yourname)`; tick it in the PR that
fixes it. Priorities: **P1** affects what readers or CI see, **P2** tidiness.

## API pages: public symbols with no entry

All existing `::: fomtools.…` directives resolve. These public symbols have no API page, and
several of them are used by the user guide, the cvxpygen tutorial or the codegen guide.

- [ ] **P1** [docs/api/canonicalizer.md](../docs/api/canonicalizer.md): add `fomtools.canonicalizers.sdp` (`SDP`) next to the MIP section.
- [ ] **P1** [docs/api/solvers.md](../docs/api/solvers.md): add `ClarabelSolver`, `CuClarabelSolver`, `MIPSolver`, `SDPSolver`, `make_solver`.
- [ ] **P1** [docs/api/steps.md](../docs/api/steps.md): add `QuadraticProgram`, `LinearProgram`, `StandardFormLinearProgram`.
- [ ] **P1** new page for `fomtools.sets` (`Set`, `Box`, `NonNegative`, `NonPositive`, `Point`, `L1Ball`, `LinfBall`, `Polyhedron`, `HalfSpace`, `Simplex`, `AffineSubspace`); the user guide imports `Box` from it.
- [ ] **P2** new page or section for `fomtools.stepdecomposers` and `fomtools.iteratetrajectory.IterateTrajectory` (the latter is exported at top level and used by the cvxpygen tutorial).
- [ ] **P1** new page for `fomtools.cvxpygen` (`generate_solver`, `register`, `builders.FomSpec` / `ScheduledFomSpec` / `fom_builder`, `qpdata.QPData`), referenced throughout `docs/tutorials/cvxpygen.md`.
- [ ] **P2** new page for `fomtools.nn` (`LinearLayer`, `ReLULayer`, and `Verifier.neural_network`), once the docstrings exist (see [docstrings.md](docstrings.md)).
- [ ] **P2** [docs/api/linearexpression.md](../docs/api/linearexpression.md) / [constraint.md](../docs/api/constraint.md): add top-level `stack`, `solve_linexpr`, `sum_squares` and `ComplementarityConstraint`, `QuadraticConstraint`, `SumSquares`.
- [ ] **P1** [docs/api/codegen.md](../docs/api/codegen.md): add `TerminationSpec`, `TerminationKind`, `Version`, `Backend`, `SolveMethod`, `CodegenResult`, `TimingReport`, `Recording`, `Schedule`, `as_schedule`, `check_stationary`, `scaffold` and the error classes (`UnrolledBlowupError`, `ScheduleFitError`), all of which the codegen guide uses.

## mkdocs.yml and nav

- [ ] **P2** [mkdocs.yml](../mkdocs.yml): `docs/tutorials/codegen/{lasso,control,network,nnls,portfolio}.md` and `speedup_results.md` are not in the nav (reachable by links only). Add them under Tutorials / Code generation or state that this is intended.
- [ ] **P2** [mkdocs.yml:1](../mkdocs.yml#L1): `site_name: FOM Tools` vs "FOMTools" everywhere else; no `site_url` or `repo_url` (the Material theme uses `repo_url` for the GitHub link in the header).
- [ ] **P2** [mkdocs.yml:48](../mkdocs.yml#L48) and [pyproject.toml:79](../pyproject.toml#L79): the `marimo` mkdocs plugin is enabled, but no page uses marimo markdown features, and on `mkdocs serve` it starts a marimo kernel per page (the source of the `ResourceTracker` warnings). Decide whether to drop the plugin line and the `mkdocs-marimo` dependency.

## CI workflows

- [ ] **P1** [docs.yml:37](../.github/workflows/docs.yml#L37): installs with `pip install ."[docs, cvxpygen]"` rather than the lockfile that `ci.yml` uses (`uv sync --frozen`), so the docs build can resolve different versions than the tests. Consider `uv sync --extra docs --extra cvxpygen` + `uv run`.
- [ ] **P1** [docs.yml](../.github/workflows/docs.yml): the Cloudflare deploy step has no `if: github.event_name == 'push'` guard, so it also runs on `pull_request`.
- [ ] **P0** [docs.yml:37](../.github/workflows/docs.yml#L37) + [pyproject.toml](../pyproject.toml): the notebook export needs PEPit and networkx, which the `docs` extra does not provide. See the P0 items in [notebooks.md](notebooks.md).
- `ci.yml` is clean: it checks the HiGHS default and the `fomtools[gurobi]` ImportError hint.

## Working tree and ignore rules

- [ ] **P2** [.gitignore:226](../.gitignore#L226) and [:395](../.gitignore#L395): `docs/tutorials/**/*.html` is listed twice.
- [ ] **P2** `build/` and `site/` are gitignored but present locally and contain stale copies of `docs/`, `fomtools/` and `tests/`; greps and IDE searches hit them. Nothing to commit, but `rm -rf build site` before auditing, and consider adding both to `ruff`'s `exclude` (`build` already is; `site` is not).
