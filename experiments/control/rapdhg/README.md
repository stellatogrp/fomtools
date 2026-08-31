# Restarted accelerated PDHG for the quadcopter controller

The `osqp` study next door flies the same quadcopter under an ADMM controller
and compares three implementations of it: NumPy, the C solver as first emitted
by code generation, and the same solver after the optimization phase. It works,
but the headline it was built to produce did not appear. Both optimization
agents, on two problems thirty-one times apart in size, reported the same
ceiling: **91%** (condensed) and **88%** (uncondensed) of runtime inside
`fom_lin_solve_t1` in `fom_linsolve.c`. That file is regenerated before every
verification, so the agent cannot edit it, and the optimization phase bought
only 1.08x and 1.12x.

The bottleneck is the algorithm, not the generator. ADMM's iteration is a linear
solve, and a linear solve is exactly the part of a generated solver that is
already written for you.

So this study changes the algorithm. **rAPDHG** — restarted accelerated primal-dual
hybrid gradient, Algorithm 1 of Lu and Yang, *A Practical and Optimal First-Order
Method for Large-Scale Convex Quadratic Programming* (arXiv:2311.07710) — is
matrix free. Its iteration is matrix-vector products and vector arithmetic, and
it solves no linear system at all. The extracted intermediate representation for
this study reports `factorization: none`, which is the whole point: there is
nothing left in `fom_linsolve.c` for the runtime to hide in.

## What it bought

| | multiply-adds per iteration | NumPy | emitted C | optimized C | emission | optimization |
|---|---:|---:|---:|---:|---:|---:|
| `osqp/condensed` (ADMM) | 2000 | 6.06 us | 2.88 us | 2.67 us | 2.1x | **1.08x** |
| `rapdhg/condensed` | 4240 | 13.26 us | 2.87 us | 0.144 us | 4.6x | **19.9x** |

The optimization phase went from 1.08x to 19.9x on the same plant, the same
generator, and the same agent budget. The only thing that changed is whether
the iteration contains a linear solve. (The rAPDHG program also carries a tilt
bound the ADMM one does not, which is why its iteration is larger; the earlier,
unbounded run of this study gave 4.2x and 17.9x, the same picture.)

That is the answer to the question the `osqp` study left open. Its ceiling was
not the agent and not the generator; it was that 91% of the runtime sat in a
file the agent is not allowed to edit. Remove the linear solve and the whole
iteration moves into `src/fom_solver.c`, where it can be worked on.

The gain is not free, and the honest accounting is in `condensed/README.md`:
rAPDHG needs more iterations than ADMM for the same accuracy, a good part of
the 19.9x is dispatch overhead that phase A carries and phase B removes, and
this run's agent reassociated the arithmetic where the earlier one kept it bit
exact.

## The formulation

The paper poses `minimize (1/2) x'Qx + c'x subject to Ax <= b`, a one-sided
inequality with a nonnegative dual. The condensed program carries the two-sided
thrust box `lo <= z <= hi` and, in this study, a bound of 30 degrees on the two
tilt angles at the end of every input block, `-e <= F x + C z <= e`. Both are
stacked one-sided:

    G = [ I ; -I ; C ; -C ],    h(x) = [ hi ; -lo ; e - F x ; e + F x ]

Note the sign: the lower bounds are the ones that flip, and `G z <= h` is
`z <= hi` together with `-z <= -lo` and the two tilt rows. The right-hand side
moves with the measured state, so the generated solver takes two parameters per
sampling instant, the linear cost and `h`. Clarabel solves the stacked form
directly, so `check_against_fomtools.py` checks the reformulation and the
iteration at once.

The plant, the cost, the reference, the move blocking, and the sampling period
are shared with the `osqp` study through `../common/condensed_qp.py`; the tilt
bound is an opt-in of that module which the `osqp` study does not take, so it
still generates from the published unbounded program. `condensed/README.md`
says why the bound is there and what it costs a first-order method.

## Layout

```
rapdhg/
├── condensed/     the condensed formulation; see its README for the results
│   ├── mpc.py                     the program in one-sided inequality form
│   ├── rapdhg.py                  the iteration, both as the paper writes it
│   │                              and as the flat recurrence that is recorded
│   ├── recording.py               that recurrence on a Verifier, plus the tables
│   ├── check_tilt.py              which reference the tilt-bounded controller can fly
│   ├── tuning.py                  choosing K and the primal weight, by flying them
│   ├── extend_tables.py           lengthening the schedule tables without an agent
│   ├── run_figures.py             the three arms on a non-repeating trajectory
│   ├── check_render.py            the drawn quadcopter points where it accelerates
│   ├── study.py                   the adapter that makes it a common.study.Study
│   ├── controller.py              the adapter the shared simulator flies
│   ├── check_against_fomtools.py  the controller is the recorded algorithm
│   ├── generate.py                both agent phases
│   ├── calibrate.py               what one iteration costs, per implementation
│   └── run_implementations.py     the three arms at equal wall-clock budgets
└── README.md
```

Everything runs as a module from this directory (`experiments/control/rapdhg`).
The plant, the simulator, the plots, and the code-generation pipeline live one
level up in `../common`, shared with the `osqp` study.

```
python -m condensed.check_tilt                # which reference the bounded controller can fly
python -m condensed.tuning                    # choose the restart period and primal weight
python -m condensed.check_against_fomtools    # the controller is the recorded algorithm
python -m condensed.generate --max-iters 512  # both C solvers (needs FOMTOOLS_CODEGEN_AGENT=1)
python -m condensed.extend_tables             # lengthen the schedule tables, no agent
python -m condensed.run_implementations       # the three arms at equal wall-clock budgets
python -m condensed.run_figures               # the same three on a non-repeating path
```

Two reference trajectories are used, both at a 4.0 s period. The **figure
eight** closes after one period, so every lap is identical; that makes a
steady-state error meaningful, and it also lets the closed loop resonate with
the lap. The **quasi-periodic** path is the same Lissajous construction with the
axes set to an irrational frequency ratio (the golden ratio), so it never repeats
and slowly fills a box instead of retracing one curve. The period is set by the
tilt bound: at the original 2.25 s the paths ask for 40 to 75 degrees of tilt,
which a 30-degree vehicle cannot deliver and a first-order solver cannot even
be made to refuse within a controller's budget; at 4.0 s the demand peaks at
the limit. `condensed/README.md` gives both paths in closed form and the
measurements behind the period.

## Why there is no `uncondensed/`

The `osqp` study has one, and it is where a real controller would start: the
full-space formulation keeps the states as variables and the dynamics as
constraints, which is far smaller in memory than the condensed Hessian. rAPDHG
cannot solve it in any budget a controller has. This is not a bug and not a
tuning failure, and the reason is worth recording because it generalizes.

**The formulation is right.** Clarabel on the stacked `[M; -M] z <= [hi; -lo]`
form and ADMM on the two-sided form agree on the optimal first input to
**6.3e-12**.

**The algorithm is right.** Warm-started exactly at the Clarabel optimum, rAPDHG
stays there: after 1024 iterations `||x - x*|| = 8.2e-10` and
`||y - y*|| = 7.1e-10`. It is a genuine fixed point, so nothing is wrong with
the iteration, the step sizes, or the projection.

**Only the dual is stuck**, which splitting the warm start apart shows:

| started at | after 32768 iterations |
|---|---|
| cold `(0, 0)` | `\|\|y - y*\|\|`: 6115 → 5674, seven percent of the way |
| primal exact `(x*, 0)` | no better than cold |
| **dual exact `(0, y*)`** | `\|\|x - x*\|\|`: 59 → 2.85, tracking error 0.52 → 0.070 |

Given the right multipliers the primal converges normally. The dual never
arrives.

**The reason is geometric.** The optimal dual is enormous — `||y*|| = 6115`, with
`max |nu| = 745` on the dynamics rows. Those are the costates of a hundred-step
chain: the backward recursion `nu_t = A' nu_{t+1} + Q x_t` accumulates an O(10)
stage gradient over a hundred steps. Meanwhile Theorem 1 caps the dual step at
`tau_k <= 1 / (2 ||A||) = 0.17`, whatever the restart period. So

| | dual distance | max dual step | iterations, at best |
|---|---|---|---|
| uncondensed | 6115 | 0.1725 | **35,400** |
| condensed | 59 | 0.3536 | **167** |

A ratio of 212, and a straight-line lower bound that no restart schedule can
beat. Confirming the mechanism, `cond(M)` grows with the horizon — 7.5, 16, 40,
115, **641** at horizons 5, 10, 20, 40, 100 — and short horizons do converge
(horizon 10 reaches a centimetre in 18432 iterations).

**The fix works and is unaffordable.** A fixed primal weight is the right lever,
and `w >= 1` stays inside the paper's guarantee. At the optimal
`w = ||y*|| / ||x*|| = 103` the uncondensed problem does converge: 8.8e-2 at
32768 iterations, 7.9e-3 at 131072. At 40 microseconds per NumPy iteration that
is 1.3 seconds per solve, against a 10 ms sampling period.

Diagonal preconditioning does not rescue it either. Ruiz plus Pock-Chambolle,
the paper's own preprocessing, drives `||Q||/||A||` to 1.59 — essentially ideal —
and convergence is no better, which is what identifies the obstruction as the
sharpness constant of the KKT system rather than any scaling.

The point generalizes past this problem. Being matrix free means information
walks the dynamics chain one matrix-vector product at a time, and the
multipliers that have to be built along it are two orders of magnitude larger
than the primal variables. That is the work the linear solve was doing.
