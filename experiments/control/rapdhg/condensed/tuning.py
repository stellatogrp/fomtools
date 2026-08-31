"""Choose the restart period and the primal weight, by flying them.

rAPDHG has two knobs this study has to fix before it can generate anything, and
neither can be read off the theory.

The restart period K is bounded below in Theorem 1 by a maximum of five terms,
four of which involve the sharpness constant of the KKT system and are not
computable.  The fifth, ``||Q|| / ||A||``, is 18250 here, and the paper says
plainly that this term "comes from the change of norm, and can be diminished by
rescaling the constraints and/or the objective" -- so it is a gauge artifact,
not a requirement.  The primal weight is likewise a practical device: PDQP.jl
initializes it at ``||c||/||b||`` and then adapts it, which a fixed table cannot
do.

So both are swept.  What they are swept *against* matters: not the accuracy of
one solve, but how well the closed loop flies at a fixed iteration count.  Those
disagree.  A large K converges faster asymptotically and is what an open-loop
accuracy sweep picks; but a controller always stops early, and truncated at a
few hundred iterations a large-K run can be worse than a small-K one and can
fail to improve monotonically as its budget grows.  A controller that tracks
*worse* when given more time is unusable however good its asymptotics, so
monotonicity is scored here alongside the error.

    python -m condensed.tuning

The sweep averages over several reference trajectories.  The spread between the
better configurations is a few percent, which one trajectory decides by chance,
so choosing on the figure eight alone would be fitting the choice to it.

Writes to ``out/``:

* ``tuning.csv``   one row per (K, w, trajectory, iteration count);
* ``tuning.pdf``   tracking error against iteration count, one curve per (K, w).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.reference import bspline_waypoints, figure_eight, quasi_periodic  # noqa: E402
from common.simulate import closed_loop  # noqa: E402
from condensed.controller import RapdhgController  # noqa: E402
from condensed.mpc import build_mpc  # noqa: E402
from condensed.study import REFERENCE_PERIOD  # noqa: E402

OUT = Path(__file__).parent / "out"

#: Restart periods swept.
RESTARTS = (8, 16, 32, 64, 128, 256)

#: Primal weights swept.  Values below one are excluded: the slack Lemma 1's
#: condition (3) keeps under this reparameterization is ``(w - 1) ||Q|| / beta_k``,
#: so ``w >= 1`` is what stays inside the paper's guarantee.
WEIGHTS = (1.0, 2.0, 4.0)

#: Iteration counts flown.  Weighted towards the low end, which is where a
#: millisecond of compute actually lands.
COUNTS = (2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)

#: A crashed flight is plotted and scored at this error, above any real one.
CRASH_ERROR = 3.0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=8.0, help="flight length in seconds")
    p.add_argument("--restarts", type=int, nargs="+", default=list(RESTARTS))
    p.add_argument("--weights", type=float, nargs="+", default=list(WEIGHTS))
    p.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    p.add_argument(
        "--references",
        type=int,
        default=3,
        help="trajectories to average over; the figure eight and the quasi-periodic path "
        "at the study's period, plus random B-splines",
    )
    return p


#: A regression is only counted while the curve is still meaningfully above the
#: floor that configuration reaches.  Past it the flight is limited by the
#: horizon and the actuator rather than by the solve, and the remaining wobble is
#: closed-loop noise; counting it would rank a configuration that bottoms out
#: early below one that never gets there.
FLOOR_FACTOR = 1.25


def regressions(errors: list[float]) -> tuple[int, float]:
    """How badly the error gets worse as the budget grows.

    Returns the number of regressions above the floor and the worst ratio.  A
    controller that tracks worse when given more time is unusable, but a 6%
    wobble at the floor and a fourfold blowup are not the same failure, so the
    magnitude is carried alongside the count.
    """
    floor = FLOOR_FACTOR * min(errors)
    real = [b / a for a, b in zip(errors, errors[1:]) if b > a * 1.02 and b > floor]
    return len(real), max(real, default=1.0)


def typical(errors: list[float]) -> float:
    """Geometric mean error, the summary the choice is actually made on.

    A budget buys an iteration count, and every count in the sweep is one a
    budget could land on, so the average over the range is what a controller
    experiences.  Geometric because the errors span a decade.

    A flight shorter than the settling window has no steady error to report and
    comes back as a NaN; that is a mistake in how the sweep was called, not a
    result, so it is refused rather than propagated into a silent ranking.
    """
    values = np.asarray(errors, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            f"tracking errors must be finite and positive to average, got {errors}; "
            "a flight shorter than common.simulate.SETTLE has no steady error"
        )
    return float(np.exp(np.mean(np.log(values))))


#: Reference paths are rejected below this altitude, in metres.  A B-spline
#: through random waypoints can dip well under the lowest of them, and a path
#: that touches the ground is unflyable by any controller -- seed 0 at the
#: default altitude reaches -0.00 m and crashes at every iteration count,
#: including ten thousand.  Scoring that as a solver failure would be wrong.
MIN_ALTITUDE = 0.35


def references(count: int, duration: float) -> list[tuple[str, object]]:
    """The trajectories the sweep averages over.

    Choosing on one trajectory overfits it.  The spread between the better
    configurations is only a few percent, which is well inside what a single
    reference decides by chance, so the figure eight is joined by random
    B-spline paths -- the generator ``common.reference`` provides for exactly
    this.  The figure eight and the quasi-periodic path come first, at the
    period the study reports on, because they are what the reported flights
    use and the figure eight is the most demanding of them; the splines are
    there to stop one path deciding the answer, not to set the difficulty.

    Seeds whose path leaves the flyable region are skipped, so the sweep never
    scores a configuration on a reference no controller could follow.
    """
    refs: list[tuple[str, object]] = [
        ("figure8", figure_eight(period=REFERENCE_PERIOD)),
        ("quasi", quasi_periodic(period=REFERENCE_PERIOD)),
    ]
    times = np.linspace(0.0, duration, 400)
    seed = 0
    while len(refs) < count:
        candidate = bspline_waypoints(duration, altitude=1.5, seed=seed)
        altitude = min(candidate(float(t))[0][2] for t in times)
        if altitude >= MIN_ALTITUDE:
            refs.append((f"spline{seed}", candidate))
        else:
            print(f"  skipping spline seed {seed}: path descends to {altitude:.2f} m")
        seed += 1
        if seed > 100:
            raise RuntimeError("could not find enough flyable B-spline references")
    return refs


def run(args) -> list[dict]:
    refs = references(args.references, args.duration)
    rows: list[dict] = []
    header = f"{'K':>5} {'w':>5}  " + "".join(f"{k:>7}" for k in args.counts)
    print(f"averaging over {len(refs)} trajectories: {', '.join(n for n, _ in refs)}\n")
    print(header + f"  {'typical':>8} {'regr':>5} {'worst':>7}")
    for restart in args.restarts:
        for weight in args.weights:
            problem = build_mpc(restart=restart, primal_weight=weight)
            per_count: list[list[float]] = []
            for count in args.counts:
                here = []
                for ref_name, ref in refs:
                    controller = RapdhgController(problem, ref, max_iterations=max(args.counts))
                    roll = closed_loop(controller, ref, count, duration=args.duration)
                    error = CRASH_ERROR if roll.crashed else roll.steady_error()
                    here.append(error)
                    rows.append(
                        {
                            "restart": restart,
                            "primal_weight": weight,
                            "reference": ref_name,
                            "iterations": count,
                            "crashed": int(roll.crashed),
                            "steady_error_m": round(error, 6),
                            "mean_error_m": round(float(roll.mean_error), 6),
                            "saturation_rate": round(float(roll.saturation_rate), 6),
                        }
                    )
                per_count.append(here)

            # Geometric mean across trajectories, so one crash does not decide a
            # configuration on its own but is not averaged away either.
            errors = [typical(here) for here in per_count]
            cells = [
                "      X" if any(e >= CRASH_ERROR for e in here) else f"{mean:7.3f}"
                for here, mean in zip(per_count, errors)
            ]
            count_regr, worst = regressions(errors)
            print(
                f"{restart:>5} {weight:>5.1f}  "
                + "".join(cells)
                + f"  {typical(errors):>8.3f} {count_regr:>5} {worst:>6.2f}x"
            )
    return rows


#: A configuration is rejected outright if its error ever grows by more than this
#: as the budget grows.  Below it the wobble is tolerable and the choice is made
#: on accuracy; above it the controller is unpredictable in the budget it is
#: about to be given.
MAX_REGRESSION = 2.0


def best(rows: list[dict]) -> tuple[int, float]:
    """The most accurate (K, w) among those that stay predictable.

    Ranking on monotonicity alone picks a restart period so short that the
    method barely accelerates; ranking on accuracy alone picks one whose flight
    crashes at some budgets and not at larger ones.  So predictability is a
    filter and accuracy is the objective.
    """
    by_count: dict[tuple[int, float, int], list[float]] = {}
    for row in rows:
        key = (row["restart"], row["primal_weight"], row["iterations"])
        by_count.setdefault(key, []).append(row["steady_error_m"])

    scored: dict[tuple[int, float], list[float]] = {}
    for (restart, weight, _count), errors in sorted(by_count.items()):
        scored.setdefault((restart, weight), []).append(typical(errors))

    safe = {k: v for k, v in scored.items() if regressions(v)[1] <= MAX_REGRESSION}
    if not safe:  # nothing was well behaved; fall back to the least bad
        safe = {
            k: v
            for k, v in scored.items()
            if regressions(v)[1] == min(regressions(u)[1] for u in scored.values())
        }
    return min(safe.items(), key=lambda kv: typical(kv[1]))[0]


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict], path: Path, chosen: tuple[int, float]) -> Path:
    # One point per (K, w, count), averaged over the trajectories flown there.
    by_count: dict[tuple[int, float, int], list[float]] = {}
    for row in rows:
        key = (row["restart"], row["primal_weight"], row["iterations"])
        by_count.setdefault(key, []).append(row["steady_error_m"])

    series: dict[tuple[int, float], list[tuple[int, float]]] = {}
    for (restart, weight, count), errors in by_count.items():
        series.setdefault((restart, weight), []).append((count, typical(errors)))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len({k for k, _ in series})))
    restarts = sorted({k for k, _ in series})
    styles = {1.0: "-", 2.0: "--", 4.0: ":"}
    for (restart, weight), points in sorted(series.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        is_chosen = (restart, weight) == chosen
        ax.plot(
            xs,
            ys,
            styles.get(weight, "-."),
            color=colors[restarts.index(restart)],
            linewidth=2.4 if is_chosen else 1.0,
            alpha=1.0 if is_chosen else 0.55,
            label=f"K={restart}, w={weight:g}" + (" (chosen)" if is_chosen else ""),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iterations per solve")
    ax.set_ylabel("steady tracking error (m)")
    ax.set_title("rAPDHG restart period and primal weight, flown")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    args = parser().parse_args()
    rows = run(args)
    chosen = best(rows)
    print(
        f"\nmost accurate among those never regressing by more than "
        f"{MAX_REGRESSION:g}x: K = {chosen[0]}, w = {chosen[1]:g}"
    )
    print(f"wrote {write_csv(rows, OUT / 'tuning.csv')}")
    print(f"wrote {plot(rows, OUT / 'tuning.pdf', chosen)}")


if __name__ == "__main__":
    main()
