# Markdown documentation TODOs

Scope: hand-written pages under `docs/` (except the tutorial gallery pages, which are in
[notebooks.md](notebooks.md)), plus `README.md`, `CONTRIBUTING.md` and `CLAUDE.md`.
API-page coverage gaps, nav and CI are in [repo-hygiene.md](repo-hygiene.md); the `skills/`
folder is in [skills.md](skills.md).

Baseline: line numbers are from commit `d235580` and drift; search by the quoted text if a
link misses. Claim an item by appending `(@yourname)`; tick it in the PR that fixes it.
Priorities: **P0** wrong install or usage claim that breaks a reader's first attempt,
**P1** a wrong statement a reader would act on, **P2** consistency and polish.

Facts the fixes rely on (from `pyproject.toml` and `fomtools/`):

- Core dependencies: `cvxpy`, `highspy`, `numpy`, `pydantic`, `pyyaml`, `scipy`. Extras:
  `gurobi`, `gpu`, `codegen`, `cvxpygen`, `dev`, `nn`, `docs`. `dev` and `docs` both pull in
  `fomtools[gurobi]`. `cvxpygen` is `cvxpygen>=1.0,<2`, `osqp`, `pip` (no cvxpy).
- `Verifier(method="mip", solver="highs", obbt_solver="auto", ...)`. MIP solvers: `highs`,
  `gurobi`/`grb`. SDP solvers: `clarabel`, `cvxpy`, `cuclarabel`. The simulator ignores
  `solver`. cuClarabel also needs Julia and `juliacall`, which no extra installs.
- `vp.step.*` accepts `relu`, `satlin`, `soft_threshold`, `implicit_linear` only.
- Codegen never forms an inverse; each solve is factorized (`codegen/factorize.py`).
  `SPEC.md`, `spec.json`, `data.npz` are written at the output root, not under `test/`.

## README.md

- [ ] **P1** [README.md:1](../README.md#L1): a single heading, and `pyproject.toml:8` uses it as the package readme, so the wheel's long description is just "# FOMTools". Fix: one-paragraph description, install one-liner (`pip install -e ".[dev,docs]"` for a clone), link to the docs site and the paper, pointer to `CONTRIBUTING.md`.

## CONTRIBUTING.md

- [ ] **P1** [CONTRIBUTING.md:23](../CONTRIBUTING.md#L23): `uv sync --extra docs` is an exact sync and uninstalls the `dev` extra (pytest, ruff) that the installation page just installed. Fix: `uv sync --extra dev --extra docs`, or drop the line since the `make` targets already use `uv run --extra docs`.
- [ ] **P2** [CONTRIBUTING.md:26-28](../CONTRIBUTING.md#L26-L28): "exports missing or stale notebooks" is only partly true; staleness looks at the notebook source and `_recorded_timings.json` (`scripts/export_marimo.py:101-111`), not at `helper_functions/*.py` or `fomtools/`. Say when `docs-export-force` is needed (or fix the script, see [notebooks.md](notebooks.md)).
- [ ] **P2** does not say that exporting the notebooks needs a Gurobi license (`scripts/export_marimo.py:88-90`, `docs.yml`).
- [ ] **P2** does not say that the pre-commit `uv-lock` hook needs `uv` on `PATH` (`.pre-commit-config.yaml:19-24`).

## CLAUDE.md

- [ ] **P1** [CLAUDE.md:7](../CLAUDE.md#L7): "using convex optimization techniques" is wrong (MIP is nonconvex); the overview omits code generation and CVXPYgen.
- [ ] **P1** [CLAUDE.md:27-28](../CLAUDE.md#L27-L28): plain `mkdocs serve` / `mkdocs build` skip the notebook export. Use `make docs-serve` / `make docs-build` (and mention `make docs-export`).
- [ ] **P2** [CLAUDE.md:33-35](../CLAUDE.md#L33-L35): pip commands are unquoted (`.[dev]` globs in zsh) and the `uv sync` / `uv.lock` workflow is not mentioned.
- [ ] **P1** [CLAUDE.md:46](../CLAUDE.md#L46): the Verifier bullet lists only ReLU, soft threshold and satlin; add `implicit_linear`, `prox`/`project`, `quadratic_program`/`linear_program`, `fixed_point`, `codegen`.
- [ ] **P2** [CLAUDE.md:49](../CLAUDE.md#L49): "linear combinations of function values" is PEP wording; they are affine maps of leaf variables.
- [ ] **P1** [CLAUDE.md:55-58](../CLAUDE.md#L55-L58): `AffineStep`, `ReluStep`, `SaturatedLinearStep`, `SoftThresholdStep` do not exist. Real classes: `Affine`, `Relu`, `SaturatedLinear`, `SoftThreshold`, `QuadraticProgram`, `LinearProgram`, `StandardFormLinearProgram` (`fomtools/steps/__init__.py`).
- [ ] **P1** [CLAUDE.md:61-62](../CLAUDE.md#L61-L62): `GurobiCanonicalizer` and `SDPCanonicalizer` do not exist. Real classes: `Canonicalizer`, `MIP`, `SDP`, `Simulator`, `OBBTHandler` (`fomtools/canonicalizers/__init__.py`).
- [ ] **P1** [CLAUDE.md:65-66](../CLAUDE.md#L65-L66): there is no `canonicalizers/obbt_handlers/` package and no `GurobiOBBTHandler`; the handler is `canonicalizers/obbt_handler.py`, class `OBBTHandler(solver=...)`, Gurobi or HiGHS.
- [ ] **P2** [CLAUDE.md:70](../CLAUDE.md#L70): no `Vector` class exists anywhere under `fomtools/`.
- [ ] **P1** [CLAUDE.md:60-66](../CLAUDE.md#L60-L66): omits `solvers/`, `stepdecomposers/`, `sets/`, `proxable_functions/`, `codegen/`, `cvxpygen/`, `nn/`, `iteratetrajectory.py`, `utils/`.
- [ ] **P1** [CLAUDE.md:77](../CLAUDE.md#L77): "Core: cvxpy, jax, numpy" is wrong (see the facts above; jax is in the `nn` extra, gurobipy in `gurobi`).
- [ ] **P2** [CLAUDE.md:90](../CLAUDE.md#L90): the notation-table scope names the verification and codegen notebooks only; add `docs/tutorials/cvxpygen_mpc.py`.

## docs/index.md

- [ ] **P2** [index.md:1](../docs/index.md#L1): "FOM Tools's" is ungrammatical, and `site_name: FOM Tools` (`mkdocs.yml:1`) clashes with "FOMTools" everywhere else.
- [ ] **P1** [index.md:3-4](../docs/index.md#L3-L4): "parametric quadratic optimization" is too narrow (lasso, LP network flow, NNLS, NN verification), and the page never mentions code generation, CVXPYgen, the SDP backend or the simulator.
- [ ] **P1** [index.md:66](../docs/index.md#L66): the print label `"vp_gurobi residuals"` is stale; the example runs on HiGHS.
- [ ] **P1** [index.md:71-72](../docs/index.md#L71-L72): "using mixed-integer optimization" is true for this example only; the package has three backends.
- [ ] **P2** [index.md:79](../docs/index.md#L79): "Highs" should be "HiGHS"; the sentence omits Clarabel.
- [ ] **P2** [index.md:84-87](../docs/index.md#L84-L87): "contains the Python source code to reproduce experiments in our paper" is old framing for a toolbox; the author list includes the paper's co-authors but not all of `pyproject.toml:13-18`. Decide what this paragraph is for (citation box vs. authors).
- [ ] **P2** [index.md:12](../docs/index.md#L12), :17: `\mathbb{R}` and `z^\top`, `q^\top`; the parameter is called `q` while the tutorials use `x` for the parameter. See the notation table below.

## docs/installation.md

- [ ] **P2** [installation.md:1](../docs/installation.md#L1): the page has no H1.
- [ ] **P0** [installation.md:5-7](../docs/installation.md#L5-L7): "`gurobipy` is a required dependency" is wrong. It is the optional `gurobi` extra and HiGHS is the default; a license is only needed for `solver="gurobi"` and for exporting the notebooks.
- [ ] **P2** [installation.md:15](../docs/installation.md#L15): "PyPi" should be "PyPI".
- [ ] **P2** [installation.md:16](../docs/installation.md#L16): "developed and tested on 3.12" is out of date; CI tests 3.12 and 3.13 and the docs build on 3.13.
- [ ] **P0** [installation.md:36-38](../docs/installation.md#L36-L38): "cvxpy is declared only in the extras, so a bare `pip install -e .` leaves the package unimportable" is wrong; cvxpy is core and CI smoke-tests a no-extras install. Delete.
- [ ] **P0** [installation.md:40](../docs/installation.md#L40) and [:53](../docs/installation.md#L53): two `## Optional extras` sections with conflicting tables. Merge into one table with the seven extras.
- [ ] **P1** [installation.md:46](../docs/installation.md#L46): "`dev`: test and lint tooling, plus `cvxpy`" is wrong; `dev` is `fomtools[gurobi]`, pytest, pre-commit, ruff, ruff-lsp, ipdb, hydra-core, pandas, networkx, pip.
- [ ] **P2** [installation.md:47](../docs/installation.md#L47): does not say `docs` pulls in `fomtools[gurobi]` (needed to export the notebooks).
- [ ] **P1** [installation.md:49](../docs/installation.md#L49) vs [:62](../docs/installation.md#L62): "GPU simulation" vs "GPU-accelerated SDP solving". CuPy serves both the simulator and cuClarabel; say so, and note that cuClarabel also needs Julia + `juliacall`.
- [ ] **P1** [installation.md:51](../docs/installation.md#L51): the `cvxpygen` extra does not contain cvxpy.
- [ ] **P1** the first table has no `gurobi` row.
- [ ] **P2** [installation.md:60](../docs/installation.md#L60), :70: `pip install fomtools[gurobi]` is unquoted, contradicting the page's own advice at :33-34; for the clone workflow it should be `pip install -e ".[gurobi]"`.

## docs/quick_start.md

- [ ] **P2** [quick_start.md:3-4](../docs/quick_start.md#L3-L4): claims to cover proxable functions, but no `prox` call appears on the page. Either add a `prox(NormL1(...), ...)` variant or drop the claim.
- [ ] **P2** [quick_start.md:67](../docs/quick_start.md#L67), :115: uses `solver="gurobi"` (needs the extra and a license) while `index.md` uses the HiGHS default. Pick one for the entry pages.
- [ ] **P1** [quick_start.md:120-121](../docs/quick_start.md#L120-L121): "change `method` to `"sdp"` ... the subsequent code is unchanged" fails with `solver="gurobi"` (`Unknown SDP solver`); the solver must also become `"clarabel"`.
- [ ] **P2** [quick_start.md:40](../docs/quick_start.md#L40): `\|s^{k+1} - s^k\|` has no subscript.
- [ ] **P2** notation: `\[ \]` display blocks x4 (:11, :23, :27, :47), `\lVert` x6, `\operatorname` x3 (:30 prox, :49 maximize, :53 FISTA), `\min_s` (:12), `\underset`, `aligned` x2 instead of `array`, `^\top` (:31); iterate is `s` where the tutorials use `z`; `\beta` is used for FISTA's `t_k` while the tutorials reserve `\beta_k` for the momentum coefficient. See the table below.

## docs/user_guide/index.md

- [ ] **P1** [user_guide/index.md:27](../docs/user_guide/index.md#L27): "reuse the imports and dimensions from the workflow above" points at nothing on this page; the snippets need `ft`, `np`, `m`, `n`, `D`, `eta`, `lambd`, `K`, `VP`, `s` from the quick start. Add a setup block or an explicit pointer.
- [ ] **P1** [user_guide/index.md:67-93](../docs/user_guide/index.md#L67-L93): `demo` is rebound to new Verifiers, but the later snippets (:106-187) still use `z0` and `x_demo` from the first one, so the page cannot be run top to bottom (on SDP: `Expression not found`). Use distinct names for the backend examples.
- [ ] **P1** [user_guide/index.md:77](../docs/user_guide/index.md#L77): `solver="numpy"` is not a solver name; it only works because the simulator ignores `solver`. Use `Verifier(method="simulator", obbt_solver=None)` as `user_guide/codegen.md:44` does.
- [ ] **P2** [user_guide/index.md:80-93](../docs/user_guide/index.md#L80-L93): lists 3 of the 10 SDP cut flags (`ineq_product_cuts`, `ineq_box_cuts`, `ineq_square_cuts`, `relu_envelope_cuts`, `rlt_all_pairs`, `cross_cut`, `subgrad_mono_cut` are missing). List them or point at the API page.
- [ ] **P1** [user_guide/index.md:99-101](../docs/user_guide/index.md#L99-L101): "linear solves ... introduce no new leaves" is wrong for `step.implicit_linear`, which creates a leaf (an `Affine` step). Only `fomtools.solve_linexpr` stays inline. Name both.
- [ ] **P1** [user_guide/index.md:61](../docs/user_guide/index.md#L61), :146-148: "all three backends support the same objective norms" and the "zero objective" are inaccurate; SDP `set_zero_objective` raises `NotImplementedError`, and the zero objective is a method, not an `ord` value.
- [ ] **P1** `solve()` options are never mentioned: `convex_hull_cuts`, `cut_rounds`, `tighten_objective_bounds`, `warm_start`, and native solver kwargs. Add a short "Solve options" subsection (the quick start on `index.md` already uses `convex_hull_cuts=True`).
- [ ] **P2** [user_guide/index.md:144](../docs/user_guide/index.md#L144): `\lVert` with no subscript.
- [ ] **P2** unhyphenated compound modifiers at :8, :52, :61, :66, :81, :82, :169, :170, :206, :207, :222, :237 ("first order method", "cone mate", "self contained", "user level", ...).

## docs/user_guide/codegen.md

- [ ] **P1** [user_guide/codegen.md:191-195](../docs/user_guide/codegen.md#L191-L195): the project tree puts `SPEC.md`, `spec.json`, `data.npz` under `test/`; they are written to the root (`codegen/targets/embedded_c.py:134-138`). It also omits `fom_ops.{c,h}`, `fom_linsolve.{c,h}`, `fom_profile.{c,h}` and `bench_main.c`.
- [ ] **P1** [user_guide/codegen.md:299-301](../docs/user_guide/codegen.md#L299-L301): "`CG` ... reserved for a future backend" is stale; CG is implemented for the large-scale version and `AUTO` upgrades to it on CUDA (`codegen/config.py:42-66`, `factorize.py:331`). The enum values are lowercase `"ldlt"`, `"lu"`, `"cg"`.
- [ ] **P1** [user_guide/codegen.md:441-443](../docs/user_guide/codegen.md#L441-L443), [:459-460](../docs/user_guide/codegen.md#L459-L460): "matrix inverses are folded into the coefficients" contradicts :292-302 on the same page and `factorize.py:3`.
- [ ] **P1** [user_guide/codegen.md:448-485](../docs/user_guide/codegen.md#L448-L485) CLI section: omits `--agent`, `--driver stub` / `--golden`, `--version`, `--backend`, `--solve-method`, `--max-iter`, `--tol`, `--dtype`, `--phase-b` / `--no-phase-b`, `--no-profile`; the examples `cgqp`, `pdhg_lp`, `indef_kkt`; and the `param_values` key (`codegen/cli.py:74-146`, `examples.py:468-477`).
- [ ] **P1** [user_guide/codegen.md:183-186](../docs/user_guide/codegen.md#L183-L186): only the Claude driver is described; add `FOMTOOLS_CODEGEN_BACKEND`, the codex and pi backends, `agents.yaml` and `$FOMTOOLS_CODEGEN_AGENTS` (`codegen/agent/__init__.py:57-108`, `cli_specs.py:165-185`).
- [ ] **P2** [user_guide/codegen.md:294](../docs/user_guide/codegen.md#L294), :317, :398-401: unicode math (`γ_k`, `β_{k-1}`, `M₀`, `LDLᵀ`) in prose; `γ` is used for momentum and `β` for `t_k`, the reverse of the tutorial convention.

## docs/api/*.md (page prose; coverage gaps are in repo-hygiene.md)

- [ ] **P2** [api/index.md:5](../docs/api/index.md#L5), :35-37 and [api/proxablefunctions.md:3-5](../docs/api/proxablefunctions.md#L3-L5): "the function class an algorithm is verified over" is PEP framing; verification ranges over parameter sets, and proxable functions are the prox operators applied inside the method.
- [ ] **P2** [api/index.md:13-14](../docs/api/index.md#L13-L14) and [api/canonicalizer.md:3-4](../docs/api/canonicalizer.md#L3-L4): "mixed-integer or semidefinite" omits the simulator, which the same page then documents.
- [ ] **P2** [api/index.md:33-34](../docs/api/index.md#L33-L34), [api/solvers.md:3-4](../docs/api/solvers.md#L3-L4): "Gurobi and HiGHS" only; add Clarabel / cuClarabel.
- [ ] **P2** [api/index.md:22](../docs/api/index.md#L22), [api/linearexpression.md:3](../docs/api/linearexpression.md#L3): "linear combinations of iterates and function values" mirrors the stale docstring.
- [ ] **P2** [api/index.md:19-21](../docs/api/index.md#L19-L21), [api/verifier.md:3-5](../docs/api/verifier.md#L3-L5): "drives the canonicalizers and OBBT handlers" (plural); there is one `OBBTHandler`.

## Notation drift outside the notebooks

The `CLAUDE.md` table is stated for the tutorial notebooks only, so these are consistency
opportunities, not violations. Counts are occurrences.

| File | Drift |
| --- | --- |
| `docs/index.md` | `\mathbb{R}` x3 (:17), `^\top` x2 (:12); parameter `q` instead of `x` |
| `docs/quick_start.md` | `\[ \]` x4; `\lVert` x6; `\operatorname` x3; `\min_s`; `\underset`; `aligned` x2; `^\top`; iterate `s`; `\beta` for `t_k` |
| `docs/user_guide/index.md` | `\lVert` x1 without subscript (:144) |
| `docs/user_guide/codegen.md` | unicode math at :294, :317, :398-401; `γ`/`β` roles swapped |
| `docs/tutorials/codegen/*.md` | see [notebooks.md](notebooks.md) |

- [ ] **P2** decide whether the table applies to all of `docs/`; if yes, extend its scope line in `CLAUDE.md` and fix the files above.
