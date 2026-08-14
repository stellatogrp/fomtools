"""The budget sweep, shared by both formulations.

Fixes the algorithm and the problem and varies only the number of ADMM
iterations the controller is allowed per sampling instant.  That is the quantity
an implementation determines: an interpreted iteration, an emitted one, and an
agent-optimized one buy different numbers of iterations inside the same
wall-clock budget.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import plots
from common.reference import figure_eight
from common.simulate import Controller, Rollout, closed_loop, time_per_iteration

#: Budgets swept for the table.
BUDGET_GRID = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

#: Budgets drawn in the comparison figure and the animation.  These show the
#: whole range of outcomes: lost, holding on badly, and flying the curve.
FIGURE_BUDGETS = (100, 250, 1000)


def parser(description: str) -> argparse.ArgumentParser:
    """The command line both drivers share."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--duration", type=float, default=8.0, help="flight length in seconds")
    p.add_argument(
        "--budgets", type=int, nargs="+", default=list(BUDGET_GRID), help="budgets to sweep"
    )
    p.add_argument(
        "--figure-budgets",
        type=int,
        nargs="+",
        default=list(FIGURE_BUDGETS),
        help="budgets to draw in the comparison figure and animation",
    )
    p.add_argument("--no-gif", action="store_true", help="skip the animation")
    p.add_argument("--fps", type=int, default=20)
    return p


def run(controller: Controller, out: Path, args) -> dict[int, Rollout]:
    """Sweep the budgets, write the table and the figures, return the flights."""
    out.mkdir(parents=True, exist_ok=True)
    ref = figure_eight()

    print(controller.describe())
    per_iter = time_per_iteration(controller, ref)
    print(
        f"NumPy iteration:   {per_iter:.2f} us  "
        f"({10_000 / per_iter:.0f} iterations in a 10 ms budget)\n"
    )

    budgets = sorted(set(args.budgets) | set(args.figure_budgets))
    rollouts = {}
    for k in budgets:
        roll = closed_loop(controller, ref, k, duration=args.duration)
        rollouts[k] = roll
        print(" ", roll.summary())

    write_csv(out / "sweep.csv", rollouts, per_iter)
    write_npz(out / "rollouts.npz", rollouts)

    chosen = [rollouts[k] for k in args.figure_budgets]
    print()
    print("wrote", plots.still_comparison(chosen, out / "comparison.pdf"))
    print("wrote", plots.error_vs_time(chosen, out / "error.pdf"))
    if not args.no_gif:
        print("wrote", plots.make_gif(chosen, out / "flight.gif", fps=args.fps))
    return rollouts


def write_csv(path: Path, rollouts: dict[int, Rollout], per_iter: float) -> None:
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


def write_npz(path: Path, rollouts: dict[int, Rollout]) -> None:
    payload = {}
    for k, roll in rollouts.items():
        payload[f"states_{k}"] = roll.states
        payload[f"inputs_{k}"] = roll.inputs
        payload[f"errors_{k}"] = roll.errors
        payload[f"reference_{k}"] = roll.reference
        payload[f"crashed_{k}"] = np.array(roll.crashed)
    np.savez_compressed(path, **payload)
    print("wrote", path)
