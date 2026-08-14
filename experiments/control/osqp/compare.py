"""Compare the two formulations at equal wall-clock budgets.

Sweeping the iteration count, as each study does on its own, answers "how many
iterations does this formulation need".  It does not answer the question a
controller actually faces, which is how well it flies given a fixed slice of
time.  The two formulations do not cost the same per iteration, so that question
needs both numbers: a budget is converted into an iteration count separately for
each, and only then are the flights compared.

    python -m compare

Writes to ``out/``:

* ``comparison.csv``     one row per budget per formulation;
* ``budget.pdf``         tracking error against the time budget, both curves;
* ``flights.pdf``        side-by-side flights at one budget.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Same shim as ``condensed/__init__.py``; this module is not inside a package, so
# nothing else puts the shared ``common`` package on the path for it.
_shared = str(Path(__file__).resolve().parents[1])
if _shared not in sys.path:
    sys.path.append(_shared)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import plots  # noqa: E402
from common.reference import figure_eight  # noqa: E402
from common.simulate import closed_loop, initial_state, time_per_iteration  # noqa: E402
from condensed import admm as condensed_admm  # noqa: E402
from condensed.controller import CondensedController  # noqa: E402
from condensed.mpc import build_mpc as build_condensed  # noqa: E402
from uncondensed import admm as uncondensed_admm  # noqa: E402
from uncondensed.controller import UncondensedController  # noqa: E402
from uncondensed.mpc import build_mpc as build_uncondensed  # noqa: E402

OUT = Path(__file__).parent / "out"

#: Compute budgets in microseconds.  The sampling period is 10 ms; a real
#: controller shares it with estimation, communication, and logging, so the
#: solver's slice of it is a fraction and the interesting range is wide.
BUDGETS_US = (200.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0)


def same_control_problem(ref) -> None:
    """Check that the two programs have the same minimizing first input.

    If this fails nothing downstream means anything: the flights would differ
    because the controllers differ, not because the algorithm ran out of time.
    """
    cond = build_condensed()
    unc = build_uncondensed()
    state = initial_state(ref)

    q_c = cond.linear_cost(state, cond.reference_window(ref, 0.0))
    u_c = cond.first_input(condensed_admm.solve_exact(cond, q_c))

    q_u = unc.linear_cost(unc.reference_window(ref, 0.0))
    lo, hi = unc.bounds(state)
    u_u = unc.first_input(uncondensed_admm.solve_exact(unc, q_u, lo, hi))

    gap = float(np.max(np.abs(u_c - u_u)))
    print(f"optimal first input agrees to {gap:.3e}")
    print(f"  condensed   {np.array2string(u_c, precision=5)}")
    print(f"  uncondensed {np.array2string(u_u, precision=5)}")
    if gap > 1e-4:
        raise SystemExit("FAILED: the two formulations do not describe the same control problem")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS_US))
    parser.add_argument(
        "--figure-budget", type=float, default=2000.0, help="budget drawn in flights.pdf"
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ref = figure_eight()

    same_control_problem(ref)

    controllers = {
        "condensed": CondensedController(build_condensed(), ref),
        "uncondensed": UncondensedController(build_uncondensed(), ref),
    }

    print()
    per_iter = {}
    for name, controller in controllers.items():
        print(controller.describe())
        per_iter[name] = time_per_iteration(controller, ref)
        print(f"iteration cost:      {per_iter[name]:.2f} us\n")

    budgets = sorted(set(args.budgets) | {args.figure_budget})
    rows = []
    flights = {}
    for name, controller in controllers.items():
        for budget in budgets:
            k = max(1, int(budget / per_iter[name]))
            roll = closed_loop(controller, ref, k, duration=args.duration)
            rows.append(
                {
                    "formulation": name,
                    "budget_us": budget,
                    "us_per_iteration": per_iter[name],
                    "iterations": k,
                    "crashed": int(roll.crashed),
                    "steady_error_m": roll.steady_error(),
                    "mean_error_m": roll.mean_error,
                }
            )
            flights[(name, budget)] = roll
            status = "CRASHED" if roll.crashed else f"{roll.steady_error():.3f} m"
            print(f"  {name:<12} {budget / 1000:5.1f} ms -> {k:5d} iterations   {status}")

    _write_csv(OUT / "comparison.csv", rows)
    _plot_budget(OUT / "budget.pdf", rows)

    chosen = [flights[(n, args.figure_budget)] for n in controllers]
    labels = [
        f"{n} ({flights[(n, args.figure_budget)].iterations} iterations)" for n in controllers
    ]
    print("\nwrote", plots.still_comparison(chosen, OUT / "flights.pdf", labels=labels))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("\nwrote", path)


def _plot_budget(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    colors = {"condensed": "#1f77b4", "uncondensed": "#d95f02"}
    for name, color in colors.items():
        subset = [r for r in rows if r["formulation"] == name]
        budgets = [r["budget_us"] / 1000 for r in subset]
        errors = [np.nan if r["crashed"] else r["steady_error_m"] for r in subset]
        ax.loglog(budgets, errors, "o-", color=color, linewidth=1.6, markersize=4, label=name)
        for r, b in zip(subset, budgets):
            if r["crashed"]:
                ax.plot(b, 3.0, "x", color=color, markersize=9, markeredgewidth=2)
    ax.set_xlabel("compute budget per sampling instant (ms)")
    ax.set_ylabel("steady tracking error (m)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
