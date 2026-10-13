# Tutorial notebook TODOs

Scope: the marimo notebooks under `docs/tutorials/`, their helper modules, the codegen
write-up pages, the gallery pages and `scripts/export_marimo.py`.

Baseline: line numbers are from commit `d235580` and drift; search by the quoted text if a
link misses. Claim an item by appending `(@yourname)`; tick it in the PR that fixes it.
Priorities: **P0** breaks the docs build, **P1** prose that contradicts the code or the
result, **P2** notation, wording, dead code.

Audit result: every fomtools call in the notebooks and helpers matches the current signatures
(Verifier kwargs, `add_param(samples=)`, `iterate_trajectory` / `.assign`, `step.*` kwargs,
`prox(IndBox)`, `set_objective_norm`, `VP[...]` stat keys, `codegen_rolled`,
`CodegenConfig` / `TerminationSpec`, `make_agent_driver`, `find_compiler`,
`fcpg.generate_solver`, solver names). The infinity-norm switch (commit `4da0756`) is
consistent in lasso, nnls, network and portfolio; control is consistently squared 2-norm.
`\ge` / `\le` inside `mo.md` cells are fine: `CLAUDE.md` bans them only in matplotlib labels,
and every plot label already uses `\geq`.

## P0: dependencies the docs build is missing

- [ ] **P0** [pepit_runner.py](../docs/tutorials/helper_functions/pepit_runner.py) imports PEPit, which is in no extra and not in `uv.lock`. `docs.yml` installs `.[docs, cvxpygen]`, so the lasso, nnls and control PEP cells fail to export in CI (they work locally only because PEPit is installed by hand). Fix: add `pepit` to the `docs` extra in `pyproject.toml` and re-lock.
- [ ] **P0** [network.py:5](../docs/tutorials/helper_functions/network.py#L5) imports `networkx`, which is only in the `dev` extra; the network notebook and the SDP notebook (imports this helper in its first cell) fail to export in docs CI. Fix: add `networkx` to the `docs` extra.
- [ ] **P1** [export_marimo.py:110](../scripts/export_marimo.py#L110): the staleness check looks only at the notebook and `_recorded_timings.json`, so a non-forced `make docs-export` misses edits to `helper_functions/*.py`. Add the helper modules to the dependency list.

## docs/tutorials/verification/lasso.py

- [ ] **P1** [lasso.py:112-115](../docs/tutorials/verification/lasso.py#L112-L115): the setup prose says p=20 / n=15 (and later 15 / 20), m=100 signals, columns drawn N(0, 1/p) and normalized. The code uses `p, n = 3, 4`, `num_star = 20`, N(0, 1) with no normalization (column 3 of `D` is all zero). Rewrite to match the code; do not reuse `m` for the signal count.
- [ ] **P2** [lasso.py:122](../docs/tutorials/verification/lasso.py#L122), :124: commented-out `# p, n, lambd = 15, 20, 1e-1` and `# num_star = 100`. Delete.
- [ ] **P1** [lasso.py:337-340](../docs/tutorials/verification/lasso.py#L337-L340): "the sampled radius used previously (R ~ 23.7 instead of 3.0)" refers to a removed cell (`estimate_lasso_pepit_radius` is gone). Print `_pepit_radius` from the code instead of hardcoding, and drop the comparison.

## docs/tutorials/verification/nnls.py

- [ ] **P1** [nnls.py:497-502](../docs/tutorials/verification/nnls.py#L497-L502): this `mo.md` repeats the "We run the simulation of momentum-accelerated ..." text, but the cell after it is the momentum *plot*. Reword.
- [ ] **P2** [nnls.py:146-160](../docs/tutorials/verification/nnls.py#L146-L160): the snippet is called "the shared implementation", but `pgd_step` also handles `IterateTrajectory`. Call it "a simplified version of".
- [ ] **P2** [nnls.py:237](../docs/tutorials/verification/nnls.py#L237): stray backticks ("descent``.").
- [ ] **P2** [nnls.py:425-426](../docs/tutorials/verification/nnls.py#L425-L426): "used to show up as a missing K=1 point" is history. Drop.
- [ ] **P2** [nnls.py:227](../docs/tutorials/verification/nnls.py#L227): "through following function" (missing "the").

## docs/tutorials/verification/control.py

- [ ] **P1** [control.py:84](../docs/tutorials/verification/control.py#L84) vs [:106](../docs/tutorials/verification/control.py#L106): the problem statement includes `s_min <= s_t <= s_max`, but the code says state constraints are not imposed. Remove the row or add the constraint.
- [ ] **P1** [control.py:203-208](../docs/tutorials/verification/control.py#L203-L208): "the simulator cannot project onto the resulting zero-width state box" is stale; symbolic `satlin` bounds are supported in the simulator since commit `0ef041a` (`tests/test_simulator.py:15`). Use the `IndBox` projection in simulator mode too, or reword the reason for the two code paths.
- [ ] **P2** [control.py:360-362](../docs/tutorials/verification/control.py#L360-L362), :590: `n_{xz}` (a code name) is used in math but never defined. Define once, e.g. `n_{xz} = n_s + (T-1) n_u`.
- [ ] **P2** [control.py:453-454](../docs/tutorials/verification/control.py#L453-L454): debug-style prints of the MIP arrays (no other notebook prints these). Remove or format like the summary print.
- [ ] **P2** [control.py:335](../docs/tutorials/verification/control.py#L335): the summary header does not say the values are squared (ord=2). "Summary statistics (squared, at K={K})".
- [ ] **P2** [control.py:483](../docs/tutorials/verification/control.py#L483): "the empirical radius above"; the radius is introduced below.
- [ ] **P2** [control.py:544](../docs/tutorials/verification/control.py#L544): `pepit_radius_calibrated` assigned but never used.
- [ ] **P1** [control.py:580](../docs/tutorials/verification/control.py#L580) vs [portfolio.py:346-348](../docs/tutorials/verification/portfolio.py#L346-L348): control still plots a PEPit curve calibrated to match the MIP at K=1; portfolio's prose says that construction "was not a bound" and removed it. Pick one stance (recommended: drop the calibrated curve) and apply it to both.
- [ ] **P2** [control.py:138](../docs/tutorials/verification/control.py#L138): tuning comment `# 0.1, 1, 10`.

## docs/tutorials/verification/network.py

- [ ] **P1** [network.py:536](../docs/tutorials/verification/network.py#L536): "the objective `\mu^T f`"; `f` is the variable of the feasibility-check code, the iterate is `z`. Write `\mu^T z`.
- [ ] **P2** [network.py:486](../docs/tutorials/verification/network.py#L486), :504: bare `print(all_residuals)` / `print(all_residuals_momentum)`.
- [ ] **P2** [network.py:201](../docs/tutorials/verification/network.py#L201), :333: momentum written `\tilde z^k = z^k + \beta_k (z^k - z^{k-1})` while the PDHG display uses the `k -> k+1` form. Write `\tilde z^{k+1} = z^{k+1} + \beta_{k+1} (z^{k+1} - z^k)` per the `CLAUDE.md` convention.
- [ ] **P2** [network.py:176](../docs/tutorials/verification/network.py#L176), :180: `\Pi_{\mathbf{R}_+}` has no dimension; use `\Pi_{\mathbf{R}^{n_s}_+}`.
- [ ] **P2** [network.py:448-450](../docs/tutorials/verification/network.py#L448-L450), :521-523: "solves in (well) under a second" and "does not finish" are hardcoded runtime claims. Soften or drop.
- [ ] **P2** [network.py:247](../docs/tutorials/verification/network.py#L247): "through following function".

## docs/tutorials/verification/portfolio.py

- [ ] **P1** [portfolio.py:340-351](../docs/tutorials/verification/portfolio.py#L340-L351): changelog-style prose ("Earlier versions ... used to fall below") that also names the private `_osqp_admm_residual` (public method: `PEPitRunner.osqp_admm_residuals`). Rewrite as a plain explanation of why there is no PEPit curve.
- [ ] **P2** [portfolio.py:269](../docs/tutorials/verification/portfolio.py#L269): mutable default `obbt_params={"output_flag": False}` (ruff B006). Default to `None` and fill in inside the function.
- [ ] **P2** [portfolio.py:195](../docs/tutorials/verification/portfolio.py#L195), :274: docstrings say `x^0 = 0` while the prose calls the allocation `z`. Use `z^0` (or "`x0` in code").

## docs/tutorials/verification/sdp.py

- [ ] **P1** [sdp.py:1069](../docs/tutorials/verification/sdp.py#L1069): same stale "simulator cannot project onto the zero-width state box" comment as control.py. Fix together.
- [ ] **P2** [sdp.py:1120-1126](../docs/tutorials/verification/sdp.py#L1120-L1126): the `collapse_expr="auto"` description leaves out that it also stops collapsing once assignments accumulate (ADMM / DR) on MIP. Add one sentence.
- [ ] **P1** [sdp.py:866-871](../docs/tutorials/verification/sdp.py#L866-L871), [:326-331](../docs/tutorials/verification/sdp.py#L326-L331): hardcoded result claims ("the first two horizons took several minutes", "the momentum PGD bound increases with K"). Re-check against a fresh run or soften.
- [ ] **P2** [sdp.py:250](../docs/tutorials/verification/sdp.py#L250), :407, :565, :741, :977: mutable dict defaults (B006). Use `None`.

## docs/tutorials/codegen/*.py

- [ ] **P1** [codegen/lasso.py:86](../docs/tutorials/codegen/lasso.py#L86), [control.py:66](../docs/tutorials/codegen/control.py#L66), [network.py:74](../docs/tutorials/codegen/network.py#L74), [portfolio.py:72](../docs/tutorials/codegen/portfolio.py#L72): "the NumPy reference solver shared with the X verification tutorial" is wrong; the verification notebooks do not use these solvers. Say "lives in the shared helper module".
- [ ] **P2** [codegen/control.py:85-151](../docs/tutorials/codegen/control.py#L85-L151): copies `A_dyn`, `B_dyn` and the condensed `P` inline. Use `make_control_dynamics()` and `build_control_qp(..., 2)` from the helper.
- [ ] **P1** [codegen/network.py:84](../docs/tutorials/codegen/network.py#L84): "lower-demand endpoint" is wrong; `x_lb = -7` is the larger demand. Say "at `x = x_lb`".
- [ ] **P2** [codegen/nnls.py:61-67](../docs/tutorials/codegen/nnls.py#L61-L67): title and intro are out of step with the other four ("NNLS: Verified Code Generation" instead of "...: Python vs Codegen", no timed-twice sentence). Align.
- [ ] **P1** [codegen/nnls.py:75-81](../docs/tutorials/codegen/nnls.py#L75-L81): "problem construction ... shared with the NNLS verification tutorial", but the box ([0, 0.25] vs [10, 11]) and the step size (1/L vs 1/(2L)) differ. Note the difference.
- [ ] **P2** [codegen/nnls.py:203-205](../docs/tutorials/codegen/nnls.py#L203-L205): empty trailing cell.
- `codegen/portfolio.py` is clean.

## docs/tutorials/codegen/*.md write-ups (pre-`bb57dc8` notation)

The verification notebooks were brought in line with the `CLAUDE.md` notation table in commit
`bb57dc8`; these write-ups were not. Port each from the matching verification notebook's
current prose.

- [ ] **P2** [codegen/lasso.md:21](../docs/tutorials/codegen/lasso.md#L21), :25, :34: iterate `s` instead of `z`; `\lVert` x2, `\mathbb{R}` x2, `\( \)` x3, `D^\top`; `\mathbf{soft}` used without its definition.
- [ ] **P2** [codegen/nnls.md:21-34](../docs/tutorials/codegen/nnls.md#L21-L34): `\lVert`, `\( \)` x2, `A^\top`, `\max\{\cdot, 0\}` where `\Pi_{\mathbf{R}^n_+}` is the convention.
- [ ] **P2** [codegen/control.md:22-46](../docs/tutorials/codegen/control.md#L22-L46): `^\top` x4, `\( \)` x7; `x_t` for the state (x is the parameter; the verification notebook uses `s_t`); horizon `H` instead of `T`; `w^k` where the verification notebook has `w^{k+1}`.
- [ ] **P2** [codegen/network.md:20-47](../docs/tutorials/codegen/network.md#L20-L47): `^\top` x3, `\( \)` x10; cost `c` and capacity `u` (verification uses `\mu` and `c`); `\max\{., 0\}` instead of `\Pi`.
- [ ] **P2** [codegen/portfolio.md:20-50](../docs/tutorials/codegen/portfolio.md#L20-L50): `^\top` x6, `\( \)` x6; `x` / `r` for allocation and returns (verification uses `z` / `\mu`); `w^k`.
- [ ] **P2** [codegen/speedup_results.md:5](../docs/tutorials/codegen/speedup_results.md#L5): column wording "after C optimization" vs the notebooks' "after optimization iteration"; "us" vs "µs". Unify. All five rows and the verification errors match `_recorded_timings.json`.
- [ ] **P2** [codegen/speedup_results.md:3](../docs/tutorials/codegen/speedup_results.md#L3): add the provenance that `_recorded_timings.json` carries (commit `afe8c9c`, 2026-07-05, Python median vs C mean, hardware unknown).

## docs/tutorials/cvxpygen_mpc.py and cvxpygen.md

- [ ] **P2** [cvxpygen_mpc.py:166](../docs/tutorials/cvxpygen_mpc.py#L166): "agree to within 1e-10" is hardcoded, and the docs build skips the solve so the number is never shown. Soften or drop.
- [ ] **P1** [cvxpygen.md:98](../docs/tutorials/cvxpygen.md#L98): `[generate_code](codegen.md)` now resolves to the tutorials gallery; it meant the guide. Link `../api/codegen.md#fomtools.codegen.generate_code` or `../user_guide/codegen.md`.
- The rest of `cvxpygen.md` matches the code (`generate_solver` kwargs, `fom_options` names, the three paths, OSQP-style residuals, auto-registration).

## Gallery pages

- [ ] **P1** [verification.md:22-24](../docs/tutorials/verification.md#L22-L24): says the SDP notebook bounds "all five examples above", but the control SDP solve is disabled by default (`sdp.py:1026`, `sdp_ctrl = None`). Mention the exception.
- [ ] **P2** [verification.md:12](../docs/tutorials/verification.md#L12): the Lasso entry omits FISTA; [:16-17](../docs/tutorials/verification.md#L16-L17): "operator-splitting method" for the network example should say PDHG.
- `tutorials/index.md` and `tutorials/codegen.md` are clean; the 18x to 222x and 95x to 910x ranges match the recorded data.

## Helpers (docs/tutorials/helper_functions/)

- [ ] **P2** [pepit_runner.py:68-81](../docs/tutorials/helper_functions/pepit_runner.py#L68-L81), :127-155, :246-275, :335-403: `admm_residuals`, `pdhg_residuals`, `pdhg_momentum_residuals` and their private versions are never called. Delete.
- [ ] **P2** [pepit_runner.py:46](../docs/tutorials/helper_functions/pepit_runner.py#L46): docstring says "FISTA-style" for both momentum variants. Use "accelerated (Nesterov-type or FISTA)".
- [ ] **P2** [nnls.py:18](../docs/tutorials/helper_functions/nnls.py#L18): "optionally with FISTA momentum", but `beta_k = (k-1)/(k+2)` is Nesterov-type momentum per `CLAUDE.md`.
- [ ] **P2** [codegen_benchmarks.py:14-78](../docs/tutorials/helper_functions/codegen_benchmarks.py#L14-L78): `run_lasso_codegen_benchmark` duplicates `run_codegen_benchmark`. Call the generic one with `example="lasso", params=[x_param], state=[s]` and delete the duplicate.
- `lasso.py`, `control.py`, `portfolio.py` helpers are clean; ruff reports no unused imports.
