# Quadcopter control experiments

One control problem, two algorithms, and a question about what code generation
is worth.

A quadcopter tracks a figure eight under model predictive control, starting on
the path. Every 10 ms it solves a quadratic program and applies the first input
move, and it gets a fixed slice of that period to do it in. How many iterations fit in the slice is
not a property of the algorithm — it is a property of the code. So each study
holds the algorithm, the problem, the reference, and the cold start fixed, and
varies only what runs the iteration: NumPy, the C solver as first emitted by
code generation, and the same solver after the optimization phase.

```
control/
├── common/     the plant, the simulator, the plots, the code-generation pipeline
├── osqp/       ADMM, in a condensed and an uncondensed formulation
└── rapdhg/     restarted accelerated PDHG, matrix free
```

Each study runs as a module from its own directory, so `common` is reached by a
short path shim in each study package rather than an environment variable:

```
cd osqp   && python -m condensed.run_implementations
cd rapdhg && python -m condensed.run_implementations
```

Both studies build their program from `common/condensed_qp.py`: the same plant,
cost, reference, and move blocking. The `rapdhg` study additionally bounds the
two tilt angles at 30 degrees, the state constraint the OSQP example carries;
`rapdhg/condensed/README.md` says why, and what a bound like that costs a
first-order method.

## What the two studies say together

| study | iteration is | emission | optimization |
|---|---|---:|---:|
| `osqp/condensed` | a linear solve | 2.1x | **1.08x** |
| `rapdhg/condensed` | matrix-vector products | 4.6x | **19.9x** |

The `osqp` study found a ceiling and identified it: 91% of the emitted runtime
sat inside `fom_lin_solve_t1` in `fom_linsolve.c`, one of the deterministic files
the generator writes and the optimization agent may not edit. Nothing the agent
could do would move it.

The `rapdhg` study is the control for that explanation. It is the same plant
and cost, the same generator, and the same agent budget, changing the algorithm
to one whose iteration contains no linear system. Its `src/fom_linsolve.c` is
one line long and declares no functions, every multiply-add lands in the file
the agent owns, and the optimization phase buys 19.9x.

So the ceiling was the algorithm's shape, not the generator's. Neither number is
the whole story on its own — rAPDHG needs more iterations per solve than ADMM,
and each study's README gives the accounting.
