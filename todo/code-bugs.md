# Code bugs surfaced by the documentation audit

Scope: behaviour that contradicts its own docstring or the user-facing docs. Each fix should
change the code and the matching docstring in the same PR, and add a regression test.

Baseline: line numbers are from commit `d235580` and drift; search by symbol if a link misses.
Claim an item by appending `(@yourname)`; tick it in the PR that fixes it.
Priorities: **P0** wrong results or a broken user path, **P1** silent misbehaviour,
**P2** quality.

- [ ] **P0** [verifier.py:1074](../fomtools/verifier.py#L1074) `Verifier.theory_bound`:
  `_lb, _ub = lb - C, lb + C` builds the upper bound from the *lower* bound of `ref_expr`.
  The OBBT mirror two statements later uses `ub + C`, which is the intended form.
  Fix: `_ub = ub + C`. Test: a leaf target with asymmetric reference bounds, check the
  tightened `ub` equals `ref.ub + C`. Rewrite the garbled `k` entry and drop the nonexistent
  `bound_expr` from the docstring at the same time.

- [ ] **P1** [verifier.py:1328-1332](../fomtools/verifier.py#L1328-L1332) `Verifier.post_process`:
  `np.maximum(ref.lb - delta_sum, ref.lb)` is always `ref.lb` (and the `ub` line always
  `ref.ub`) for `delta_sum >= 0`, so the residuals never affect anything and the return value
  is always 0. Only `tests/test_partial_relu.py` and `skills/fomtools/SKILL.md:43` call it.
  Fix: decide the intended semantics (most likely: set the target bounds to
  `[ref.lb - delta_sum, ref.ub + delta_sum]` intersected with the target's current bounds) or
  delete the method and its two call sites. Update the docstring either way.

- [ ] **P1** [norm_l1.py:117](../fomtools/proxable_functions/norm_l1.py#L117) `NormL1.grad`:
  returns `sign(x)` for `f(x) = sum_i lambda_i |x_i|`; the subgradient is
  `lambda_i * sign(x_i)`. The docstring formula repeats the bug. Fix code and formula; test
  against a finite difference away from zero with a non-unit `lambd`.

- [ ] **P1** [factory.py:25](../fomtools/solvers/factory.py#L25) and
  [sdp.py:1994-2001](../fomtools/canonicalizers/sdp.py#L1994-L2001): `make_solver` and its
  docstring (`factory.py:50`, `:74`) advertise `"cvxpy_sdp"`, but the SDP canonicalizer's
  `match` accepts only `clarabel`, `cvxpy`, `cuclarabel`, so
  `Verifier(method="sdp", solver="cvxpy_sdp").solve()` raises `Unknown SDP solver`.
  Fix: add the alias to the `match` or remove it from `_SDP_SOLVERS` and both docstrings.

- [ ] **P1** [clarabel.py:194](../fomtools/solvers/clarabel.py#L194) `ClarabelSolver.solve`:
  returns `-sol.obj_val` without checking `sol.status`, so an infeasible, inaccurate or
  time-limited solve returns a number that looks optimal. Fix: check the status and raise or
  warn.

- [ ] **P2** no MIP solver exposes the dual bound separately from the incumbent (no
  `ObjBound` anywhere under `fomtools/solvers/`), so a time-limited solve returns the incumbent
  as "the worst case" with no indication of the gap. Fix: expose `obj_bound` and `mip_gap` on
  `GurobiSolver` and `HiGHSSolver` and surface them in `Verifier.solve` stats.

- [ ] **P2** [mip.py:51](../fomtools/canonicalizers/mip.py#L51) `MIP.__init__`: default
  `solver="gurobi"` while `Verifier` and `Canonicalizer.__init__` default to `"highs"`.
  Harmless today because `Verifier` always passes `solver`, but anyone constructing `MIP`
  directly gets Gurobi. Fix: default to `"highs"`.

- [ ] **P2** [verifier.py:349](../fomtools/verifier.py#L349) `_get_step_decomposer`: the
  `ValueError` message says "Method must be 'mip' or 'sdp'" although `"simulator"` is
  accepted. Fix the message together with the docstring item in [docstrings.md](docstrings.md).
