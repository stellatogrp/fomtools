# Uncondensed formulation

Nothing is eliminated.  The states stay as variables and the dynamics stay as
constraints,

```
minimize    (1/2) zeta' P zeta + q(r)' zeta
subject to  l(x) <= M zeta <= u(x),
```

over `zeta = (xi_1, ..., xi_100, z_1, ..., z_10)`: 1240 variables and 1240
constraint rows at the same one second lookahead the condensed study uses, in
three groups — the initial state, the dynamics, and the input box.

Everything is banded.  `P` is diagonal, each dynamics row touches two
consecutive states and one input move, and the system matrix
`P + sigma I + M' R M` holds 15188 entries out of 1.5 million, 0.99% filled.
Its sparse factorization holds 26739, against the 769000 a dense factor of that
size would hold.  Every solve applies that factorization; nothing is refactorized
inside the loop.

Two consequences distinguish this formulation from the condensed one, beyond
size.

**The measured state enters through the bounds.**  `xi_1 = x` is an equality,
and an equality is a bound whose two sides coincide, so the first twelve rows of
`l` and `u` move every sampling instant while the rest stay fixed.  The
condensed formulation puts the measured state in the linear cost instead.  This
matters for code generation and not at all for the simulation; see the note at
the end.

**The equality rows need their own penalty.**  `rho_equality` defaults to 1000,
which is OSQP's own convention for equality constraints, and it is not optional
here.  At the same penalty as the inequality rows the iterate buys objective by
violating the dynamics:

| equality penalty | objective gap at 5000 iterations | worst dynamics violation |
|---:|---:|---:|
| 0.1 (same as the box) | -0.56 | 1.1e-1 |
| 1 | -0.33 | 2.5e-2 |
| 10 | -0.11 | 3.5e-3 |
| 100 (the default) | -6.0e-6 | 3.1e-7 |

A negative gap means the iterate scores *better* than the true optimum, which it
can only do by being infeasible.  At the low settings the controller is
optimizing over motions the quadcopter cannot perform, and it is still doing so
after five thousand iterations.  The condensed formulation has no equality rows
for any of this to apply to: its dynamics are substituted away and every row of
its constraint matrix is a thrust bound.

## Running it

```
python -m uncondensed.check_against_fomtools    # the controller is the recorded algorithm
python -m uncondensed.run_budget_sweep          # flights against the iteration count
FOMTOOLS_CODEGEN_AGENT=1 python -m uncondensed.generate            # emit both C solvers
python -m uncondensed.calibrate                 # what one iteration costs, per implementation
python -m uncondensed.run_implementations       # flights against the compute budget
```

The iteration-count sweep writes `sweep.csv`, `comparison.pdf`, `error.pdf` and
`flight.gif` to `out/`, with the same names and meanings as in `condensed/`; the
budget study writes `implementations*.{csv,pdf,gif}` alongside them.  The
generated solvers land in `out/codegen/{emitted,optimized}` and are reused on a
later run if they still pass `make verify`, so the agent phases are paid for
once.  Each of those directories carries a 173 MB `data.npz` of the IR's
coefficient blocks, which is one reason `out/` is not in the repository.

## Calibration

Eight second flights, 10 ms sampling, error in metres, steady excluding the
first two seconds.

| iterations | outcome | steady error | thrusts on the box |
|---:|---|---:|---:|
| 10 | flew | 2.05 m | 86% |
| 25 | flew | 2.04 m | 90% |
| 50 | flew | 2.27 m | 85% |
| 100 | flew | 1.53 m | 83% |
| 250 | flew | 0.30 m | 92% |
| 500 | flew | 0.12 m | 80% |
| 1000 | flew | 0.12 m | 66% |
| 2500 | flew | 0.12 m | 58% |

The converged error, 0.119 m, is the same as the condensed study reaches, which
is the strongest evidence available that the two programs describe one control
problem; `compare.py` checks it directly on the optimal first input.

The approach to it is different, though.  This controller does not crash until
the budget falls below about ten iterations, where the condensed one is already
gone at a hundred.  Starved, it degrades into a sluggish controller rather than
a wrong one: the large penalty on the equality rows keeps the predicted
trajectory roughly consistent with the dynamics even when the solve is nowhere
near converged, so the first input stays sane.  That robustness does not make it
the better choice at a fixed time budget — see the parent README — but it is a
real difference and not an artifact.

A NumPy iteration costs about 37 us here, six times the condensed one for
thirty-one times the variables.

## Code generation

The measured state is a genuine parameter of the generated solver, not a frozen
constant, so what is emitted is a solver that flies.  Getting there needs one
rewrite, because the parameter enters through the bounds and code generation
stores saturated-linear bounds as names of constant arrays: the intermediate
representation has nowhere to put a bound that moves, and
`extract.py:_satlin_arrays` raises `NonStationaryFOMError` on one that is a
`LinearExpression`.  The identity

```
sat_[l, u](y) = l + sat_[0, u-l](y - l)
```

moves the parameter out of the bounds and into the expression the step is applied
to, which code generation accepts.  It is exact, and it costs almost nothing
here: the width `u - l` is constant everywhere, zero on all 1200 equality rows
and the thrust range on the 40 box rows, so only the twelve rows that pin the
initial state move at all.  `recording.py` holds that recording, and it is the
one `check_against_fomtools.py` checks, so the algorithm that is verified and the
algorithm that is generated are the same object.

What comes out: two body steps over a state of dimension 2480, two parameters
(the 1240-long cost and the 12-long measured state), 15 coefficient blocks
holding 28422 applied entries, and an LDL^T factorization holding 18679 — fewer than the
26739 of the SuperLU factor NumPy uses, because it is a symmetric factorization
under a fill-reducing ordering.  **65780 multiply-adds per iteration.**

### It is slower than NumPy, and the reason is specific

| implementation | us/iteration | spread | iterations in 10 ms | effective rate |
|---|---:|---:|---:|---:|
| NumPy | 37.39 | 2.3% | 267 | 1.8 G mul-add/s |
| emitted C | 71.92 | 0.2% | 139 | 0.9 G mul-add/s |
| optimized C | 64.32 | 0.3% | 155 | 1.0 G mul-add/s |

All three agree to 1e-10 at every iteration count checked, and flown for eight
seconds at a common iteration count their trajectories differ by 3.9e-11, so this
is one algorithm in three implementations and nothing else.

There is no interpreter overhead to recover.  The per-solve cost outside the loop
measures 1.1 us, so essentially all of NumPy's 37 us is arithmetic it performs in
compiled code: 21.8 us in SuperLU's triangular solve, 10.9 us in two CSR
matrix-vector products, about 2 us in everything else.  Generated C would have to
beat 1.8 G mul-add/s in scalar CSR loops merely to break even.  The condensed
program is the opposite regime and emission pays 2.1x there; see
`condensed/README.md`.

The optimization phase found real structure and still only bought 1.15x, because
of where the time sits.  Its final profile:

| region | us/solve | share |
|---|---:|---:|
| `t1_solve` | 1076.6 | 87.7% |
| `state_update` | 83.8 | 6.8% |
| `t1_op` | 51.6 | 4.2% |
| `t0_step` | 14.4 | 1.2% |

`t1_solve` is `fom_lin_solve_t1` in `fom_linsolve.c`, a deterministic file
restored before every verify and therefore off limits to the agent.  It started
the phase at 78% and ended it at 88% precisely because everything the agent could
reach got faster.  What it reached was worth reaching — `t1_state[:, 1240:]` is
bit-exactly `-0.5 * t1_t_t0` and `t1_state[:, :1240]` is exactly `1e-4 I`, which
collapses two 6982-nnz matrix-vector products into one; `t0_state`, `su_state`
and `su_t_t0` are exact identity or zero blocks and drop out entirely; the
parameter terms are loop invariant and hoist into `fom_update_params`.  That
halved the 22% it was allowed to touch, and it reported correctly that two or
three percent is all that remains.

So the whole deficit is one number: the emitted triangular solve runs at 53.8 us
where SuperLU runs the same factor in 21.8, because the emitter writes a scalar
CSC loop and SuperLU works on dense blocks within supernodes.  That 2.5x sits in
generated code the optimization phase is not permitted to change, and no amount
of agent effort moves it.  A blocked or supernodal emission in `fom_linsolve.c`
would close it, and would help every family whose iteration is dominated by a
sparse solve.

The condensed study reaches the same wall from the other side: 91% of its time is
inside the same routine, on a dense 820-nnz factor instead of a sparse 18679-nnz
one, and its optimization phase is capped at 1.08x for it.  Two problems
thirty-one times apart in size, one ceiling.  What limits the optimization phase
on this family is the emitted linear solve rather than the agent.

### Flights against the compute budget

Eight second flights.  The sampling period is 10 ms, which is the ceiling: a
controller cannot spend longer on one instant than the interval to the next.

| budget | NumPy | | emitted C | | optimized C | |
|---:|---:|---|---:|---|---:|---|
| 0.5 ms | 13 it | 1.79 m | 6 it | crashed | 7 it | crashed |
| 1 ms | 26 it | 2.06 m | 13 it | 1.79 m | 15 it | 1.81 m |
| 2 ms | 53 it | 2.23 m | 27 it | 2.08 m | 31 it | 2.17 m |
| 4 ms | 106 it | 1.44 m | 55 it | 2.19 m | 62 it | 2.02 m |
| 6 ms | 160 it | 0.98 m | 83 it | 1.75 m | 93 it | 1.64 m |
| 8 ms | 213 it | 0.52 m | 111 it | 1.39 m | 124 it | 1.27 m |
| 10 ms | 267 it | **0.24 m** | 139 it | **1.14 m** | 155 it | **1.02 m** |

The picture is the intended one with the arms in the opposite order: at a full
sampling period the NumPy controller holds the figure eight to 24 cm while both
generated solvers wander a metre off it, in the order their iteration costs
predict.  Below about 2 ms every implementation is too starved for the ordering
to survive the noise, and both generated solvers crash at six or seven
iterations.

## Checks

`check_against_fomtools.py` runs three executions of the recording — the NumPy
controller, `run_faithful` on the recorded steps, and `ReferenceOracle` on the
extracted representation — and they agree to about 3e-11.  It also confirms the
converged iterate matches OSQP to 1e-12 relative on the objective and satisfies
the dynamics to 2e-15.  Extraction is run with the recorded iterates supplied, so
`check_stationary` confirms that repeating one iteration reproduces the recorded
three; without that, a coefficient varying with the iteration index would be
frozen silently.

`calibrate.py` compares the generated solver against the NumPy controller at 1,
5, 20, 100 and 500 iterations.  Those counts differ from the 20 the solver was
generated with, so the check doubles as proof that the emitted loop reads
`settings.max_iter` at run time rather than the `FOM_MAX_ITER` baked into its
header — which is what lets one generated solver serve the whole budget sweep.
`run_implementations.py` flies every arm at a common iteration count before
sweeping anything, for the same reason.
