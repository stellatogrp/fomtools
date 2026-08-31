"""Which reference the tilt-bounded controller can fly, before anything is generated.

Adding the tilt bound changes the program in a way a first-order method feels:
the multipliers on the tilt rows have to be built by dual steps whose size
Theorem 1 caps at ``1 / (2 ||G||)``, and the bound raises ``||G||`` from
``sqrt(2)`` to about nine.  On a path the bounded vehicle cannot follow those
multipliers are enormous, and no iteration budget a controller has will build
them: the flight then neither tracks nor respects the bound.  On a path it can
nearly follow the bound only clips the peaks, and the method converges as it
did before.

So this flies the NumPy arm on both references at several periods, at the
iteration counts the budgets buy and at one generous count, and prints what a
Clarabel solve of the same program does alongside, which is what the flight
would be with an exact solver.  The period the study generates at is the most
demanding one on which the bounded controller still converges: median relative
KKT error below ``KKT_TOL`` at ``PICK_ITERATIONS`` iterations, and the exact
flight's tilt within a degree of the limit.  The count is a generous one on
purpose.  Whenever the bound is active the multipliers on its rows are large
(thousands, against a dual step capped near 0.1), so respecting it takes a few
thousand iterations however K and w are set; what the pre-check has to rule
out is the path on which no count suffices, which is the 2.25 s one.

    python -m condensed.check_tilt                 # about five minutes
    python -m condensed.check_tilt --periods 3.0 3.5

Writes ``out/tilt_precheck.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from common import reference
from common.quadcopter import N_INPUT, TILT_LIMIT, U_MAX, U_MIN
from common.simulate import Command, closed_loop
from condensed import rapdhg
from condensed.controller import RapdhgController
from condensed.mpc import build_mpc
from condensed.study import OUT

PERIODS = (3.0, 3.5, 4.0)
ITERATIONS = (6, 26, 469, 2048)
PICK_ITERATIONS = 2048
KKT_TOL = 1e-3
LIMIT_DEG = float(np.degrees(TILT_LIMIT))

REFERENCES = {
    "quasi": lambda period: reference.quasi_periodic(period=period),
    "figure8": lambda period: reference.figure_eight(period=period),
}


class ExactController:
    """Clarabel on the same program at every instant: the flight an exact solver gives."""

    def __init__(self, problem, ref):
        self.problem, self.ref = problem, ref
        self.dt, self.A, self.B, self.n = problem.dt, problem.A, problem.B, problem.n

    def describe(self) -> str:
        return "Clarabel, exact"

    def command(self, state, t, iterations) -> Command:
        p = self.problem
        q = p.linear_cost(state, p.reference_window(self.ref, t))
        z = rapdhg.solve_exact(p, q, h=p.rhs(state))
        return Command(u=np.clip(z[:N_INPUT], U_MIN, U_MAX), residual=0.0, saturated=False)


def tilt_deg(roll) -> np.ndarray:
    return np.degrees(np.abs(roll.states[:, 0:2]).max(axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=float, nargs="+", default=list(PERIODS))
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    problem = build_mpc()
    unbounded = build_mpc(tilt_limit=None)
    print(
        f"bounded program: m = {problem.m}, ||G|| = {problem.norm_g:.3f}; limit {LIMIT_DEG:.0f} deg\n"
    )
    header = f"{'reference':9s}{'period':>7s}  {'arm':<16s}{'steady m':>9s}{'max tilt':>9s}{'>limit':>7s}{'med KKT':>9s}"
    print(header)
    rows = []
    verdict: dict[tuple[str, float], bool] = {}
    for period in args.periods:
        for name, make in REFERENCES.items():
            ref = make(period)
            arms = [
                ("exact, unbounded", ExactController(unbounded, ref), 1),
                ("exact, bounded", ExactController(problem, ref), 1),
            ]
            controller = RapdhgController(problem, ref, max_iterations=max(ITERATIONS))
            arms += [(f"NumPy {it} it", controller, it) for it in ITERATIONS]
            ok_kkt = ok_exact = None
            for label, ctrl, its in arms:
                t0 = time.time()
                roll = closed_loop(ctrl, ref, its, duration=args.duration)
                tilt = tilt_deg(roll)
                kkt = float(np.median(roll.residuals)) if "NumPy" in label else float("nan")
                over = float(np.mean(tilt > LIMIT_DEG + 0.5))
                rows.append(
                    {
                        "reference": name,
                        "period_s": period,
                        "arm": label,
                        "iterations": its,
                        "crashed": int(roll.crashed),
                        "steady_error_m": round(roll.steady_error(), 4),
                        "max_tilt_deg": round(float(tilt.max()), 2),
                        "over_limit_frac": round(over, 3),
                        "median_relkkt": kkt,
                    }
                )
                flag = " CRASHED" if roll.crashed else ""
                print(
                    f"{name:9s}{period:7.2f}  {label:<16s}{roll.steady_error():9.3f}{tilt.max():9.1f}"
                    f"{over:7.2f}{kkt:9.1e}{flag}   ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                if label == "exact, bounded":
                    ok_exact = tilt.max() <= LIMIT_DEG + 1.0 and not roll.crashed
                if its == PICK_ITERATIONS and "NumPy" in label:
                    ok_kkt = kkt < KKT_TOL and not roll.crashed
            verdict[(name, period)] = bool(ok_kkt and ok_exact)
            print()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tilt_precheck.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)

    passing = sorted(
        {p for (_, p), ok in verdict.items() if ok and all(verdict[(r, p)] for r in REFERENCES)}
    )
    print(
        "\nperiods on which the bounded controller converges on both references:", passing or "none"
    )
    if passing:
        print(f"recommended REFERENCE_PERIOD = {passing[0]} (the most demanding that passes)")


if __name__ == "__main__":
    main()
