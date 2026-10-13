# Skill folder TODOs

Scope: `skills/README.md`, `skills/fomtools/SKILL.md`, `skills/fomtools/references/*.md`,
`skills/update-prompt.md` and `.github/workflows/update-skill.yml`.

Baseline: line numbers are from commit `d235580` and drift; search by the quoted text if a
link misses. Claim an item by appending `(@yourname)`; tick it in the PR that fixes it.
Priorities: **P0** an example that raises when copied, **P1** a wrong statement an agent
would act on, **P2** polish.

The skill is meant to be kept in sync automatically, and it is not: this is the single most
stale body of text in the repo. Fix the sync mechanism first, then the content, or the
content will drift again.

## Sync workflow

- [ ] **P0** [update-skill.yml](../.github/workflows/update-skill.yml): no sync commit has ever landed. `git log -- skills/` shows only the two creation commits from 2026-09-22 (`e8a9291`, `fd719d4`) against 32 qualifying commits on `main` since. Causes to rule out, in order:
  - the `allowed_tools` list has `git add` / `git commit` but no `git push` and no PR creation, so even a successful run cannot land anything;
  - `prompt_file` and `allowed_tools` may not be inputs of `anthropics/claude-code-action@v1` (v1 documents `prompt` and `claude_args`); check the action's `action.yml` for the pinned version;
  - the trigger path `.claude/CLAUDE.md` and `update-prompt.md:10` reference a file that does not exist.
- [ ] **P1** [update-prompt.md](../skills/update-prompt.md): the prompt points the agent at docs, `verifier.py`, `codegen/`, `cvxpygen/` and the two CLAUDE files, but not at `pyproject.toml` (extras, deps), `codegen/cli.py` and `examples.py` (CLI flags, example names), or `solvers/` (solver names), which is exactly where the content below drifted.
- [ ] **P2** [skills/README.md:53-55](../skills/README.md#L53-L55): claims the auto-sync works. Correct once it does.

## skills/README.md

- [ ] **P1** [README.md:20](../skills/README.md#L20), :26: repository `stellatogrp/algo_verify` should be `stellatogrp/fomtools`. Same at [SKILL.md:248](../skills/fomtools/SKILL.md#L248).

## skills/fomtools/SKILL.md

- [ ] **P0** [SKILL.md:36-37](../skills/fomtools/SKILL.md#L36-L37): `add_iterate(dim=n, ...)` / `add_param(dim=n, ...)` raise `TypeError`; the argument is `ndim`.
- [ ] **P0** [SKILL.md:42](../skills/fomtools/SKILL.md#L42): `vp.solve(huchette_cuts=True)` is forwarded verbatim to the backend and raises `ValueError`. The option is `convex_hull_cuts=True`.
- [ ] **P1** [SKILL.md:43](../skills/fomtools/SKILL.md#L43): `vp.post_process(z[k], z[0], [res])` is real but a no-op today (see [code-bugs.md](code-bugs.md)) and nothing in the docs or tutorials uses it. Drop it from the canonical example.
- [ ] **P0** [SKILL.md:66](../skills/fomtools/SKILL.md#L66), :243: `vp.step.saturated_linear(x, lb, ub)` does not exist; it is `vp.step.satlin(x, l=, u=)`.
- [ ] **P0** [SKILL.md:67](../skills/fomtools/SKILL.md#L67): `vp.step.affine(x, output_mat=M)` does not exist; the step is `vp.step.implicit_linear(x, output_mat=M)`.
- [ ] **P1** [SKILL.md:141-146](../skills/fomtools/SKILL.md#L141-L146): the generated-project tree puts `SPEC.md`, `spec.json`, `data.npz` under `test/`; they are at the output root (`codegen/targets/embedded_c.py:134-138`).
- [ ] **P2** [SKILL.md:185](../skills/fomtools/SKILL.md#L185): the cvxpygen `fom=` list omits `"relaxed_admm"`.
- [ ] **P1** [SKILL.md:197-204](../skills/fomtools/SKILL.md#L197-L204): "Core deps: numpy, scipy, gurobipy/highspy; Optional: cvxpy" is wrong. Core: `cvxpy`, `highspy`, `numpy`, `pydantic`, `pyyaml`, `scipy`; extras `gurobi`, `gpu`, `codegen`, `cvxpygen`, `dev`, `nn`, `docs`. The `gurobi`, `nn` and `gpu` extras are not mentioned.

## skills/fomtools/references/api.md

- [ ] **P1** [api.md:14-15](../skills/fomtools/references/api.md#L14-L15): defaults `solver="gurobi"` and `obbt_solver="gurobi"` are wrong (`"highs"` and `"auto"`); `obbt_params` is missing; the `"cvxpy"` and `"grb"` names are missing.
- [ ] **P0** [api.md:33](../skills/fomtools/references/api.md#L33), :37: `dim` should be `ndim`.
- [ ] **P1** [api.md:52-53](../skills/fomtools/references/api.md#L52-L53): `complementarity` returns a constraint that you pass to `add_constraint` (`verifier.py:609-629`); the text implies it is added for you.
- [ ] **P0** [api.md:64-65](../skills/fomtools/references/api.md#L64-L65): same wrong step names as `SKILL.md` (`saturated_linear`, `affine`).
- [ ] **P0** [api.md:75](../skills/fomtools/references/api.md#L75): `prox(..., step_size=1.0)`; the keyword is `gamma` (`verifier.py:718`).
- [ ] **P1** [api.md:93-95](../skills/fomtools/references/api.md#L93-L95): the QP / LP signatures are invented. Real: `quadratic_program(P, q, A, b, C, d, dual_ub, primal_bounds, nu_bounds, arg, relax_binary_vars)`, `linear_program(q, ...)`, `standard_form_linear_program(c, A, b, dual_ub, ...)`; there are no `lb` / `ub` / `n` arguments (`verifier.py:806-939`).
- [ ] **P0** [api.md:104](../skills/fomtools/references/api.md#L104): `set_objective_norm(expr, norm="inf")` with `"two"` / `"one"` raises; the argument is `ord` in `{1, 2, np.inf, "inf"}`.
- [ ] **P1** [api.md:111-114](../skills/fomtools/references/api.md#L111-L114): `solve` options omit `convex_hull_cuts`, `cut_rounds`, `tighten_objective_bounds`, `warm_start` and cite the nonexistent `huchette_cuts`.
- [ ] **P1** [api.md:183-184](../skills/fomtools/references/api.md#L183-L184): `ClaravelSolver` / `cuClaravelSolver` should be `ClarabelSolver` / `CuClarabelSolver`.
- [ ] **P2** omits `fixed_point`, `set_start_samples` and the simulator statistics via `vp["mean"]` etc. (`verifier.py:172-212`).

## skills/fomtools/references/codegen.md

- [ ] **P0** [codegen.md:20](../skills/fomtools/references/codegen.md#L20): `solve_method=None` fails validation; the default is `SolveMethod.AUTO`.
- [ ] **P0** [codegen.md:28](../skills/fomtools/references/codegen.md#L28): `float_type=` is silently ignored (pydantic drops unknown fields); the field is `dtype` (`codegen/config.py:172`).
- [ ] **P0** [codegen.md:132](../skills/fomtools/references/codegen.md#L132): `"LDLT"` / `"LU"` / `"CG"` fail validation; the enum values are lowercase (`config.py:63-66`).
- [ ] **P1** [codegen.md:64](../skills/fomtools/references/codegen.md#L64): "unrolled = body has more than one block" is wrong; `unrolled` means the body is the whole recorded run (`codegen/ir.py:275-280`).
- [ ] **P1** [codegen.md:156](../skills/fomtools/references/codegen.md#L156): `--example boxqp` does not exist (it is a test fixture, `tests/codegen/conftest.py:253`); `indef_kkt` is missing from the list.
- [ ] **P0** [codegen.md:203](../skills/fomtools/references/codegen.md#L203): `oracle.run(param_values, initial_state, max_iters)` does not exist; it is `ReferenceOracle.solve(param_values=None, x0=None)` (`codegen/oracle.py:88-92`).

## skills/fomtools/references/cvxpygen.md

- [ ] **P1** [cvxpygen.md:12](../skills/fomtools/references/cvxpygen.md#L12): the `cvxpygen` extra does not contain cvxpy.
- [ ] **P2** [cvxpygen.md:55-59](../skills/fomtools/references/cvxpygen.md#L55-L59): `relaxed_admm` is missing.
- [ ] **P1** [cvxpygen.md:96](../skills/fomtools/references/cvxpygen.md#L96): `cpg_set_solver_tol` exists only under `TerminationKind.INF_NORM_CHANGE`.
- [ ] **P1** [cvxpygen.md:103](../skills/fomtools/references/cvxpygen.md#L103): "must be stationary; schedules not yet wired" is stale; the scheduled and unrolled paths exist (`cvxpygen/builders.py:143, 402`; `api.py:62`).
- [ ] **P1** [cvxpygen.md:104](../skills/fomtools/references/cvxpygen.md#L104): "Embedded C only" is wrong; OpenMP and CUDA targets are supported (`api.py:99-106`).
- [ ] **P1** [cvxpygen.md:106](../skills/fomtools/references/cvxpygen.md#L106): "residuals are `||x_k - x_{k-1}||_inf`" is wrong; they are OSQP-style primal and dual residuals (`cvxpygen/glue.py:104-137`).
