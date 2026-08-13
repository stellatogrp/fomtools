"""Fly the quadcopter at a range of iteration budgets and record what happens.

This is the calibration stage of the experiment.  It fixes the algorithm and the
problem and varies only the number of ADMM iterations the controller is allowed
per sampling instant, which is the quantity that the implementation will later
determine: an interpreted iteration, an emitted one, and an optimized one buy
different numbers of iterations inside the same wall-clock budget.

Outputs, all under ``out/``:

* ``sweep.csv``       one row per budget: tracking error, crash, saturation;
* ``rollouts.npz``    the flown trajectories, for replotting without re-running;
* ``comparison.pdf``  side-by-side panels at the three illustrative budgets;
* ``flight.gif``      the same comparison, animated;
* ``error.pdf``       tracking error against time.

Run it from this directory:

    python run_budget_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import plots
import simulate
from admm import solve as admm_solve
from mpc import build_mpc
from reference import figure_eight

#: Budgets swept for the table.
BUDGET_GRID = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

#: Budgets drawn in the comparison figure and the animation.  These three show
#: the whole range of outcomes: lost, holding on badly, and flying the curve.
FIGURE_BUDGETS = (100, 250, 1000)

OUT = Path(__file__).parent / "out"


def time_per_iteration(problem, q, iterations: int = 2000, repeats: int = 5) -> float:
    """Median wall-clock cost of one ADMM iteration, in microseconds.

    Stage three of the experiment converts a time budget into an iteration
    budget with this number, so it is recorded here alongside the flights.
    """
    admm_solve(problem, q, 200)  # warm the caches and the BLAS threads
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        admm_solve(problem, q, iterations)
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0, help="flight length in seconds")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(BUDGET_GRID),
        help="iteration budgets to sweep",
    )
    parser.add_argument(
        "--figure-budgets",
        type=int,
        nargs="+",
        default=list(FIGURE_BUDGETS),
        help="budgets to draw in the comparison figure and animation",
    )
    parser.add_argument("--no-gif", action="store_true", help="skip the animation")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    problem = build_mpc()
    ref = figure_eight()

    print(
        f"condensed program: {problem.n} variables "
        f"({problem.num_blocks} input moves over a {problem.lookahead:.2f} s horizon, "
        f"{problem.horizon} steps of {1000 * problem.dt:.0f} ms, held {problem.block} at a time)"
    )
    print(f"ADMM parameters:   rho = {problem.rho}, sigma = {problem.sigma}")

    q0 = problem.linear_cost(simulate.initial_state(ref), problem.reference_window(ref, 0.0))
    per_iter = time_per_iteration(problem, q0)
    print(
        f"NumPy iteration:   {per_iter:.2f} us  "
        f"({10_000 / per_iter:.0f} iterations in a 10 ms budget)\n"
    )

    budgets = sorted(set(args.budgets) | set(args.figure_budgets))
    rollouts = {}
    for k in budgets:
        roll = simulate.closed_loop(problem, ref, k, duration=args.duration)
        rollouts[k] = roll
        print(" ", roll.summary())

    _write_csv(OUT / "sweep.csv", rollouts, per_iter)
    _write_npz(OUT / "rollouts.npz", rollouts)

    chosen = [rollouts[k] for k in args.figure_budgets]
    print()
    print("wrote", plots.still_comparison(chosen, OUT / "comparison.pdf"))
    print("wrote", plots.error_vs_time(chosen, OUT / "error.pdf"))
    if not args.no_gif:
        print("wrote", plots.make_gif(chosen, OUT / "flight.gif", fps=args.fps))


def _write_csv(path: Path, rollouts: dict, per_iter: float) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iterations",
                "crashed",
                "flight_time_s",
                "steady_error_m",
                "mean_error_m",
                "max_error_m",
                "saturation_rate",
                "numpy_us_per_iteration",
            ]
        )
        for k, roll in sorted(rollouts.items()):
            writer.writerow(
                [
                    k,
                    int(roll.crashed),
                    f"{roll.steps * roll.dt:.3f}",
                    f"{roll.steady_error():.6f}",
                    f"{roll.mean_error:.6f}",
                    f"{roll.max_error:.6f}",
                    f"{roll.saturation_rate:.4f}",
                    f"{per_iter:.4f}",
                ]
            )
    print("\nwrote", path)


def _write_npz(path: Path, rollouts: dict) -> None:
    payload = {}
    for k, roll in rollouts.items():
        payload[f"states_{k}"] = roll.states
        payload[f"inputs_{k}"] = roll.inputs
        payload[f"errors_{k}"] = roll.errors
        payload[f"reference_{k}"] = roll.reference
        payload[f"crashed_{k}"] = np.array(roll.crashed)
    np.savez_compressed(path, **payload)
    print("wrote", path)


if __name__ == "__main__":
    main()
