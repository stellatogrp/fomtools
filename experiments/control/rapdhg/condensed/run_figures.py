"""Draw the three-arm flight figures on a reference of your choosing.

``run_implementations.py`` sweeps the budget grid and draws its figures on the
figure eight, which is the trajectory every number in the study is quoted
against.  That curve closes after one period, so every lap is identical.  This
draws the same three arms on a path that does not repeat, which is a fairer
picture of what the controller does and a less flattering one to look at.

The arms, their timings, and the conversion from budget to iteration count are
exactly the ones ``run_implementations`` uses, so the panels are comparable; only
the reference differs.  Output names carry the reference, so nothing here
overwrites the reported results.

    python -m condensed.run_figures                       # quasi-periodic, 63 us
    python -m condensed.run_figures --budget 100
    python -m condensed.run_figures --reference figure8

Writes ``figures_<reference>_flight.pdf``, ``figures_<reference>_error.pdf``, and
``figures_<reference>.gif`` to ``out/``.
"""

from __future__ import annotations

import argparse

from common import plots
from common import study as pipeline
from common.reference import bspline_waypoints, figure_eight, quasi_periodic
from common.simulate import closed_loop
from condensed.study import REFERENCE_PERIOD, build

#: The reference paths this can draw on.  ``quasi`` is the default because it is
#: the one the figure eight cannot show: an unrepeating path, so the tracking
#: error is not free to settle into a pattern tied to the lap.
REFERENCES = {
    "quasi": lambda duration, period: quasi_periodic(period=period),
    "figure8": lambda duration, period: figure_eight(period=period),
    "spline": lambda duration, period: bspline_waypoints(duration, altitude=1.5, seed=1),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", choices=sorted(REFERENCES), default="quasi")
    p.add_argument("--budget", type=float, default=40.0, help="compute budget in microseconds")
    p.add_argument(
        "--period",
        type=float,
        default=REFERENCE_PERIOD,
        help="period of the reference in seconds (ignored by the spline)",
    )
    p.add_argument("--duration", type=float, default=8.0, help="flight length in seconds")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="simulation steps per animation frame; default plays in real time at --fps",
    )
    p.add_argument("--dpi", type=int, default=100, help="resolution of the animation frames")
    p.add_argument("--no-gif", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    ref = REFERENCES[args.reference](args.duration, args.period)

    # The reference has to go in at build time: the generated arms take their
    # cost vector from the study's params_at, which closes over it.
    study = build(reference=ref)
    arms = pipeline.build_arms(study)

    print(
        f"drawing on the {args.reference!r} reference (period {args.period:g} s) "
        f"at a {args.budget:.0f} us budget\n"
    )
    print(
        f"{'implementation':<16}{'us/iteration':>13}{'iterations':>12}   outcome"
        f"{'':16}max tilt   over {pipeline.TILT_LIMIT_DEG:.0f} deg"
    )
    rollouts, labels, colors = [], [], []
    for arm in arms:
        count = arm.iterations_for(args.budget)
        roll = closed_loop(arm.controller, ref, count, duration=args.duration)
        rollouts.append(roll)
        labels.append(f"{arm.name} ({count} iterations)")
        colors.append(pipeline.COLORS[arm.name])
        outcome = "CRASHED" if roll.crashed else f"{roll.steady_error():.3f} m steady"
        print(
            f"  {arm.name:<14}{arm.us_per_iteration:>11.3f}{count:>12}   {outcome:<16}"
            f"{roll.max_tilt_deg:8.1f} deg  "
            f"{100 * roll.tilt_exceedance(pipeline.TILT_LIMIT_DEG):5.1f}%"
        )

    out = study.out
    out.mkdir(parents=True, exist_ok=True)
    stem = f"figures_{args.reference}"
    print()
    print(
        "wrote",
        plots.still_comparison(
            rollouts,
            out / f"{stem}_flight.pdf",
            labels,
            colors=colors,
            suptitle=plots.budget_title(args.budget),
        ),
    )
    print("wrote", plots.error_vs_time(rollouts, out / f"{stem}_error.pdf", labels, colors=colors))
    if not args.no_gif:
        print(
            "wrote",
            plots.make_gif(
                rollouts,
                out / f"{stem}.gif",
                labels,
                colors=colors,
                fps=args.fps,
                frame_stride=args.frame_stride,
                dpi=args.dpi,
                show_tilt=True,
            ),
        )


if __name__ == "__main__":
    main()
