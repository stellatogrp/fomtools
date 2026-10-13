# Docstring TODOs

Scope: docstrings and code comments under `fomtools/`. mkdocstrings renders these on the API
pages, so wrong text here appears on the site verbatim.

Baseline: line numbers are from commit `d235580` and drift; search by symbol if a link misses.
Claim an item by appending `(@yourname)`; tick it in the PR that fixes it.
Priorities: **P0** breaks a user or CI, **P1** a wrong statement a reader would act on,
**P2** consistency and polish.

Audit method: an AST scan compared every numpy-style `Parameters` entry under `fomtools/`
with the real signature. No entry describes a parameter that no longer exists. Everything
below is a wrong, missing or outdated description.

## fomtools/verifier.py

- [ ] **P1** [verifier.py:47](../fomtools/verifier.py#L47) `Verifier`: no class docstring. With `merge_init_into_class: false` and `ignore_init_summary: true` in `mkdocs.yml`, the API page renders the class with no description. Fix: add a class docstring (what it holds, the three backends, the declare / build / objective / solve flow) and keep the `__init__` Parameters.
- [ ] **P1** [verifier.py:76](../fomtools/verifier.py#L76) `Verifier.__init__`, `solver`: does not say the `"highs"` default is MIP-only. `Verifier(method="sdp")` with the default fails at solve time with `ValueError: Unknown SDP solver: highs`. Fix: say SDP needs `"clarabel"`, `"cvxpy"` or `"cuclarabel"`, and that the simulator ignores `solver`.
- [ ] **P1** [verifier.py:183](../fomtools/verifier.py#L183) `__getitem__`: return type says `jnp.ndarray` (numpy or cupy); the example `vp.solve(K=100)` raises `TypeError`; does not say the statistics keys are simulator-only; typo "avaiable".
- [ ] **P1** [verifier.py:218](../fomtools/verifier.py#L218) `_get_canonicalizer`: hint `solver: dict` (is `str`); the `method` entry says "identifier for the solver"; the solver list shows only `gurobi`/`grb` (add `highs`, `clarabel`, `cvxpy`, `cuclarabel`); 8 of the 10 cut flags are undocumented; the docstring is indented one space too far.
- [ ] **P2** [verifier.py:327](../fomtools/verifier.py#L327) `_get_step_decomposer`: docstring and the `ValueError` at :349 list only `mip`/`sdp`; `"simulator"` is handled too.
- [ ] **P2** [verifier.py:375](../fomtools/verifier.py#L375) `_create_var` and [verifier.py:437](../fomtools/verifier.py#L437) `add_iterate`: Returns says "dimension `shape`"; the parameter is `ndim`.
- [ ] **P1** [verifier.py:413](../fomtools/verifier.py#L413) `add_iterate`, [verifier.py:449](../fomtools/verifier.py#L449) `add_param`: `**kwargs` undocumented (forwarded to `canonicalizer.add_var`, e.g. simulator `samples=`, `init=`); `lb`/`ub` typed `float` but arrays are accepted.
- [ ] **P2** [verifier.py:554](../fomtools/verifier.py#L554) `add_constraint`: says the constraint goes to both the canonicalizer and OBBT; only linear, binary-free constraints reach OBBT.
- [ ] **P1** [verifier.py:865](../fomtools/verifier.py#L865) `linear_program`, [verifier.py:905](../fomtools/verifier.py#L905) `standard_form_linear_program`: no Parameters section.
- [ ] **P1** [verifier.py:942](../fomtools/verifier.py#L942) `step` property: does not list the names or kwargs. Fix: `relu(proj_ranges=)`, `satlin(l=, u=)`, `soft_threshold(lambd=)`, `implicit_linear(output_mat=)` or `(weight=, bias=)`, all accepting `relax_binary_vars=False`.
- [ ] **P2** [verifier.py:960](../fomtools/verifier.py#L960) `run_step`: `**kwargs` undocumented; Returns has no type.
- [ ] **P1** [verifier.py:1048](../fomtools/verifier.py#L1048) `theory_bound`: the `k` entry is garbled and refers to a nonexistent `bound_expr`; MIP-only (SDP raises `NotImplementedError`) but does not say so. Fix with the code bug in [code-bugs.md](code-bugs.md).
- [ ] **P1** [verifier.py:1117](../fomtools/verifier.py#L1117) `set_two_norm_objective`, [verifier.py:1171](../fomtools/verifier.py#L1171) `set_objective_norm`: every backend maximizes the *squared* 2-norm; only the second says so. Neither says the MIP path needs an MIQP-capable solver, so the default HiGHS fails.
- [ ] **P1** [verifier.py:1233](../fomtools/verifier.py#L1233) `solve`, `**kwargs`: examples are Gurobi-only (add HiGHS `time_limit=`, `mip_rel_gap=`); on the simulator the kwargs are `delta`/`epsilon`/`device` and the return is an `(N, n_residuals)` array, not a float.
- [ ] **P1** [verifier.py:1303](../fomtools/verifier.py#L1303) `post_process`: says bounds are relaxed by the residual sum; the code never uses the residuals. See [code-bugs.md](code-bugs.md).
- [ ] **P1** [verifier.py:1375](../fomtools/verifier.py#L1375) `codegen`, [verifier.py:1441](../fomtools/verifier.py#L1441) `codegen_rolled`: no Parameters section. Document `out_dir`, `iterates`, `config`, `driver`, `state`, `params`, `name`, `run_phase_b`, `profile`, `roll`, `param_values`, plus `next_state` for `codegen_rolled`.
- [ ] **P2** [verifier.py:1498](../fomtools/verifier.py#L1498) `get_solution_value`: MIP-only (SDP raises `NotImplementedError`; the simulator has no such method). Say so.

## fomtools/linearexpression.py

- [ ] **P1** [linearexpression.py:18-39](../fomtools/linearexpression.py#L18-L39) `LinearExpression` class docstring: "linear combination of function values, inner products (between points/gradients)" is PEPit wording. Fix: an affine expression `sum_leaf A_leaf x_leaf + const` over leaf variables. `counter` is a unique id, not a leaf count. List `const`, `lb`, `ub`, `branch_priority`.
- [ ] **P2** [linearexpression.py:21](../fomtools/linearexpression.py#L21), :226, :293, :320, :358: `Attributes:` / `Raises:` headers with a trailing colon are not numpy sections and render as plain text. Use `Attributes` + dashes.
- [ ] **P2** [linearexpression.py:66](../fomtools/linearexpression.py#L66) `__init__`: "length `shape`" should be `ndim`.
- [ ] **P2** [linearexpression.py:125-145](../fomtools/linearexpression.py#L125-L145) `from_int`: "integer `shape`" / "(shape, shape)" should be `ndim`; `lb`/`ub` undocumented and ignored when `is_binary`.
- [ ] **P2** [linearexpression.py:494](../fomtools/linearexpression.py#L494) `__rmatmul__`: Raises refers to `other`; the parameter is `matrix`.
- [ ] **P2** [linearexpression.py:613](../fomtools/linearexpression.py#L613) `update_bounds`: says "update"; it intersects (only tightens).
- [ ] **P1** [linearexpression.py:696](../fomtools/linearexpression.py#L696) `eval`: public on the rendered class, no docstring, a `# TODO`, raises `NotImplementedError`. Document as unsupported or make private.

## fomtools/constraint.py

- [ ] **P1** [constraint.py:42](../fomtools/constraint.py#L42), :54, :147, :192 `Constraint`, `EqualityConstraint`, `InequalityConstraint`: operators listed as `'=='`, `'<='`, `'>='`, but the stored values are `"="` and `"<="` (`>=` is normalized by negation at :116). `rhs` is documented as an expression but is always 0; `lhs` holds `lhs - rhs`. Describe the normalized form.

## fomtools/steps/

- [ ] **P2** [step.py:8-10](../fomtools/steps/step.py#L8-L10) `Step`: "thresholding and activation steps ... element-wise" excludes `Affine`, `QuadraticProgram`, `LinearProgram`. [step.py:74](../fomtools/steps/step.py#L74) `update_bounds` has no docstring.
- [ ] **P1** [relu.py:49](../fomtools/steps/relu.py#L49) `Relu.__init__`: `proj_ranges` is documented nowhere.
- [ ] **P1** [saturated_linear.py:9-20](../fomtools/steps/saturated_linear.py#L9-L20) `SaturatedLinear`: formula writes `S_[x,c]` with `y = (v, x)`, so `x` is overloaded (use `S_[l,u]`); "x, c in R" is wrong (`l`/`u` are arrays or a `LinearExpression`); the `l`/`u` text ("whose lower bound acts as ...") is wrong, both `.lb` and `.ub` of each are used.
- [ ] **P2** [soft_threshold.py:20](../fomtools/steps/soft_threshold.py#L20) vs :37: "lambda > 0" contradicts "must be non-negative".
- [ ] **P2** `affine.py`, `soft_threshold.py`, `saturated_linear.py`: constructor arguments appear only under Attributes; add Parameters to each `__init__` (or the class, since `merge_init_into_class` is off).

## fomtools/proxable_functions/

- [ ] **P1** [norm_l1.py:117](../fomtools/proxable_functions/norm_l1.py#L117) `NormL1.grad`: formula omits `lambda_i`. Fix with the code in [code-bugs.md](code-bugs.md).
- The other 23 modules are clean; every "Verification lowering" note matches an existing `prox_*` in the step decomposer.

## fomtools/sets/

- [ ] **P2** [set.py:88](../fomtools/sets/set.py#L88) `Set.contains`: refers to `PROX_ATOL`, which exists only in `tests/conftest.py`.
- [ ] **P2** [ball.py:20](../fomtools/sets/ball.py#L20) `L1Ball`, `dual_ub`: "needed by the backend"; it is optional and derived when `None`.
- [ ] **P2** [ball.py:12](../fomtools/sets/ball.py#L12), [polyhedron.py:13](../fomtools/sets/polyhedron.py#L13): the numerical projection "solves a KKT system"; it uses cvxpy (KKT is only the verification lowering).

## fomtools/stepdecomposers/

- [ ] **P2** [step_decomposer.py:72-85](../fomtools/stepdecomposers/step_decomposer.py#L72-L85) `StepDecomposer`: the example "(e.g. TotalVariation1D on MIP/SDP)" and the Simulator snippet are out of date; TV1D is in the base registry.
- [ ] **P1** [step_decomposer.py:484](../fomtools/stepdecomposers/step_decomposer.py#L484) `prox_indaffine`, [:1084](../fomtools/stepdecomposers/step_decomposer.py#L1084) `prox_norminf`, [:1169](../fomtools/stepdecomposers/step_decomposer.py#L1169) `prox_quadratic`: say the Simulator "defers"; it now implements all three (Affine step / batched activation).
- [ ] **P2** [step_decomposer.py:377](../fomtools/stepdecomposers/step_decomposer.py#L377) `_to_satlin_bound`: refers to `mip.py:603` (warm-start code). Name `MIP.canonicalize_saturatedlinear` instead of a line number.
- [ ] **P2** [step_decomposer.py:775](../fomtools/stepdecomposers/step_decomposer.py#L775) `prox_indsimplex`: documents the bound `lambda <= max x - min x`; the code uses the per-component `hi - arg_lb`.
- [ ] **P2** [sdp_step_decomposer.py](../fomtools/stepdecomposers/sdp_step_decomposer.py): "only natively supports ReLU" (Affine is native too). `mip_step_decomposer.py` does not mention the QP/LP decomposers.

## fomtools/canonicalizers/

- [ ] **P1** [canonicalizer.py:28](../fomtools/canonicalizers/canonicalizer.py#L28), :80, [mip.py:33](../fomtools/canonicalizers/mip.py#L33), [obbt_handler.py:29](../fomtools/canonicalizers/obbt_handler.py#L29): examples use `gurobi.Gurobi()`; the class is `GurobiSolver` and HiGHS is the default.
- [ ] **P1** [canonicalizer.py:44](../fomtools/canonicalizers/canonicalizer.py#L44) `Canonicalizer.__init__`: says `default "gurobi"`; the signature default is `"highs"`. The supported list omits the SDP names and `None`.
- [ ] **P2** [canonicalizer.py:15](../fomtools/canonicalizers/canonicalizer.py#L15): "interface for MIP and SDP"; it also covers `Simulator` and `OBBTHandler`.
- [ ] **P1** [mip.py:51](../fomtools/canonicalizers/mip.py#L51) `MIP.__init__`: no docstring (and see the default-solver item in [code-bugs.md](code-bugs.md)).
- [ ] **P1** [mip.py:767](../fomtools/canonicalizers/mip.py#L767), :878, :1049 `canonicalize_relu` / `canonicalize_saturatedlinear` / `canonicalize_softthreshold`: say they construct the convex hull (the `(I, o)` families). The code emits a big-M MILP plus a simple envelope; the hull inequalities are only separated lazily by `solve(convex_hull_cuts=True)`.
- [ ] **P2** [mip.py:1070-1071](../fomtools/canonicalizers/mip.py#L1070-L1071), :1108 `canonicalize_softthreshold`: `Phi` is copied from saturated-linear (`[b, c]`, `S_[b,c]`); a denominator reads `u^0_o - u^0_o`.
- [ ] **P2** [mip.py:1180](../fomtools/canonicalizers/mip.py#L1180), :1198: comments cite "line 918/927", which no longer match. Replace with symbol names.
- [ ] **P1** [sdp.py:23](../fomtools/canonicalizers/sdp.py#L23), :49-59 `SDP`: "solves via Clarabel or CVXPY" omits cuClarabel; Attributes document 2 of the 10 cut flags.
- [ ] **P2** [sdp.py:811](../fomtools/canonicalizers/sdp.py#L811) `canonicalize_affine`: refers to `mip.py:404`; the target is `MIP.canonicalize_affine` (~:747). Use the symbol.
- [ ] **P2** [sdp.py:1940-1943](../fomtools/canonicalizers/sdp.py#L1940-L1943) `solve_model`: `steps` and `convex_hull_cuts` "accepted for API compatibility", but no caller passes them. Remove the parameters or the note.
- [ ] **P2** [sdp.py:347](../fomtools/canonicalizers/sdp.py#L347), [solver.py:82](../fomtools/solvers/solver.py#L82), [gurobi.py:149](../fomtools/solvers/gurobi.py#L149) `add_param_stack_var`: dead API; `Verifier.add_param_stack` uses `stack()`. Remove or mark deprecated.
- [ ] **P2** [simulator.py:435](../fomtools/canonicalizers/simulator.py#L435) `Simulator`: "using CuPy"; it uses CuPy when available and falls back to NumPy.

## fomtools/solvers/

- [ ] **P1** [highs.py:13](../fomtools/solvers/highs.py#L13) `HiGHSSolver`: no class docstring, and the class is rendered on `docs/api/solvers.md`. [highs.py:196-200](../fomtools/solvers/highs.py#L196-L200) `update_bounds`: `lb`/`ub` are indented under `expression` and do not parse.
- [ ] **P2** [gurobi.py:25](../fomtools/solvers/gurobi.py#L25) `GurobiSolver`: a placeholder `...` is left in Attributes. [gurobi.py:119](../fomtools/solvers/gurobi.py#L119), :136 `add_var`: says it wraps `addVar` (uses `addMVar`); "Returns None" against a `-> gp.Var` annotation.
- [ ] **P2** [sdp_solver.py:12-27](../fomtools/solvers/sdp_solver.py#L12-L27) `SDPSolver`: "skeleton / placeholder ... separate pull request" is stale; `ClarabelSolver` and `CuClarabelSolver` are real subclasses.
- [ ] **P1** [factory.py:50](../fomtools/solvers/factory.py#L50) `make_solver`: advertises `"cvxpy_sdp"`, which the SDP canonicalizer rejects. See [code-bugs.md](code-bugs.md).
- [ ] **P2** [mip_solver.py:22](../fomtools/solvers/mip_solver.py#L22) `MIPSolver`: "adding convex hull cuts"; it separates and returns them, the canonicalizer adds them.
- [ ] **P2** [cuclarabel.py:42](../fomtools/solvers/cuclarabel.py#L42): says it accepts solver params; any non-empty params raise.

## fomtools/utils/ and fomtools/nn/

- [ ] **P2** [dict_utils.py:7](../fomtools/utils/dict_utils.py#L7), :37 `merge_dict`, `prune_dict` (exported at top level): keys typed `Variable` (no longer a class; it is `LinearExpression`); the text says `d1`/`d2` while the parameters are `dict1`/`dict2`. The other `utils/` modules are clean.
- [ ] **P1** [layers.py:11](../fomtools/nn/layers.py#L11), :22 `LinearLayer`, `ReLULayer`: no docstrings (exported by `fomtools.nn`). [layers.py:176](../fomtools/nn/layers.py#L176) `is_relu`: Parameters mis-indented; Returns underline has the wrong length.

## fomtools/codegen/

- [ ] **P1** [ir.py:3-21](../fomtools/codegen/ir.py#L3-L21) module docstring: "one iteration of a stationary FOM" is out of date (`FomIR` also holds unrolled runs and schedules); says the affine inverse is "folded into the linear-op coefficients (see `Affine.lhs_mat_inverse`)". That attribute does not exist (it is `output_mat_inverse`) and the IR uses `IterationStep.solve` plus a `Factorization`, never an inverse.
- [ ] **P2** [ir.py:244](../fomtools/codegen/ir.py#L244) `FomIR`: the `factorizations` field (:289) is missing from Attributes.
- [ ] **P1** [oracle.py:7-9](../fomtools/codegen/oracle.py#L7-L9) (`run_faithful` "rather than reusing its folded inverse"), [extract.py:134](../fomtools/codegen/extract.py#L134) ("the precomputed inverse"), [examples.py:312](../fomtools/codegen/examples.py#L312) `mpc_osqp_example` ("its inverse is folded into the generated coefficients"): same stale wording; solves are factorized (LDL^T / LU / CG).
- [ ] **P2** [generate.py:10-15](../fomtools/codegen/generate.py#L10-L15) module docstring: ties the baseline timing to `run_phase_b`; timing depends on `profile=True` plus an available compiler.
- [ ] **P1** [generate.py:249](../fomtools/codegen/generate.py#L249) `generate_code`: says args are forwarded to `extract_ir`; on the `iterates=` path they go to `extract_unrolled_ir`. No Returns section.
- [ ] **P1** [oracle.py:286](../fomtools/codegen/oracle.py#L286), :364 `check_stationary`, `check_unrolled`: no Parameters section; both are exported.
- [ ] **P2** [extract_unrolled.py:192](../fomtools/codegen/extract_unrolled.py#L192) `extract_unrolled_ir`: `name` undocumented. [factorize.py:220](../fomtools/codegen/factorize.py#L220) `factorize`: `symmetry_tol` undocumented.
- [ ] **P1** [config.py:134](../fomtools/codegen/config.py#L134) `CodegenConfig`: `agent_timeout_s`, `agent_optimize_max_turns`, `agent_optimize_timeout_s` missing from Attributes.
- [ ] **P2** [targets/base.py:10-11](../fomtools/codegen/targets/base.py#L10-L11), :33-36 and [emit.py:3-6](../fomtools/codegen/emit.py#L3-L6): say the agent-written file is always `src/fom_solver.c`; on the CUDA target it is `.cu`.
- Clean: `schedule`, `extract_scheduled`, `dedup`, `spec`, `wrapper`, `harness`, `postprocess`, `cli`, `codegen/__init__`.

## fomtools/codegen/agent/

- [ ] **P1** [driver.py:3-6](../fomtools/codegen/agent/driver.py#L3-L6) module docstring: "the real driver shells out to the Claude Agent SDK"; backends are now claude, codex, pi and CLI specs. [driver.py:184](../fomtools/codegen/agent/driver.py#L184) `run_verify_loop`: garbled sentence ("it calls runs one ...").
- [ ] **P2** [agent/__init__.py:13-14](../fomtools/codegen/agent/__init__.py#L13-L14), :54-55 and [cli_specs.py:16-18](../fomtools/codegen/agent/cli_specs.py#L16-L18): describe Pi as a "JSON-RPC subprocess" / "RPC driver"; `PiDevAgentDriver` runs `pi --print`.
- [ ] **P2** [pidev_sdk.py:13](../fomtools/codegen/agent/pidev_sdk.py#L13): "unlike the single-key Codex/Gemini backends"; there is no Gemini backend.
- [ ] **P2** [sdk_base.py:92-94](../fomtools/codegen/agent/sdk_base.py#L92-L94) `SdkAgentDriver` Attributes, `cli_timeout`: says it is the delay before falling back to the CLI; it is the per-attempt kill budget (the `__init__` docstring and the code agree). No module docstring. [:231](../fomtools/codegen/agent/sdk_base.py#L231) `_invoke_agent`: the fallback also triggers on `NotImplementedError`. [:266](../fomtools/codegen/agent/sdk_base.py#L266) `_invoke_cli` and [cli_driver.py:254](../fomtools/codegen/agent/cli_driver.py#L254): `timeout` undocumented.
- Clean: `stub`, `claude_sdk`, `codex_sdk`, `prompts`.

## fomtools/cvxpygen/

- [ ] **P2** [api.py:52](../fomtools/cvxpygen/api.py#L52) `generate_solver`: `bound_margin` missing from Parameters.
- [ ] **P2** [builders.py:304-305](../fomtools/cvxpygen/builders.py#L304-L305) `_admm_step`: refers to an "anchored builder ... Halpern average" that does not exist.
- [ ] **P1** [cvxpygen/__init__.py:18-19](../fomtools/cvxpygen/__init__.py#L18-L19): the example shows a manual `register_solve`; `generate_solver(..., wrapper=True)` registers automatically. [:29](../fomtools/cvxpygen/__init__.py#L29) lists cvxpy as optional; it is a core dependency.
- Clean: `glue`, `interface`, `qpdata`.

## Clean modules

`iteratetrajectory.py`, `canonicalizers/__init__.py`, `sets/__init__.py`, `solvers/solver.py`
(apart from the dead `add_param_stack_var`), `obbt_handler.py` (apart from the example),
`simulator_step_decomposer.py`.
