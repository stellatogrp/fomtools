"""The code-generation pipeline, shared by the two formulations.

A study is everything that differs between them: the program, the recorded
iteration, how a measured state and a time become solve-time parameters, and how
a solver's answer becomes rotor thrusts.  Everything else — generating the two C
solvers, timing them, checking them against NumPy, flying them at equal compute
budgets, drawing the figures — is the same work and lives here.

That matters for what the experiment claims.  The two formulations are compared
on trajectories produced by identical code, so a difference between them is a
property of the formulation and not of two scripts that drifted apart.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import codegen_driver, plots  # noqa: E402
from common.csolver import CSolver  # noqa: E402
from common.quadcopter import TILT_LIMIT  # noqa: E402
from common.recording import ParameterBoxes, Recording, ScheduledRecording  # noqa: E402
from common.reference import Reference  # noqa: E402
from common.simulate import (  # noqa: E402
    Command,
    Controller,
    Rollout,
    closed_loop,
    us_per_iteration,
)

#: Iteration counts at which a generated solver is checked against NumPy.  They
#: differ from the count the solver is generated with, so the check doubles as
#: proof that the emitted loop reads ``settings.max_iter`` at run time rather
#: than the ``FOM_MAX_ITER`` baked into its header.  That is what lets one
#: generated solver serve a whole budget sweep.
CHECK_ITERATIONS = (1, 5, 20, 100, 500)

#: Agreement required between a generated solver and the NumPy controller.  Set
#: at the code generator's own equivalence tolerance; both run the same
#: arithmetic in a different order and in practice agree far better.
TOL = 1e-6

#: Compute budgets per sampling instant, in microseconds.  The sampling period is
#: 10 ms, which is the hard ceiling: a controller cannot spend longer on one
#: instant than the interval to the next.  A real one shares that period with
#: estimation, communication, and logging, so the grid runs down from it.
BUDGETS_US = (100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 10_000.0)

#: One colour per implementation, shared by the flight panels, the error curve,
#: and the budget plot.  Ordered by how well each tracks, so the figures read the
#: same way the table does.
#: The tilt bound the flights are reported against, in degrees.
TILT_LIMIT_DEG = float(np.degrees(TILT_LIMIT))

COLORS = {"NumPy": "#1b7837", "optimized C": "#e08214", "emitted C": "#c0392b"}


@dataclass
class Study:
    """One formulation, ready to be generated, timed, and flown."""

    name: str
    problem: object
    ref: Reference
    out: Path
    #: Either a stationary :class:`~common.recording.Recording`, or a
    #: :class:`~common.recording.ScheduledRecording` for a method whose
    #: coefficients vary with the iteration index.
    recording: Recording | ScheduledRecording
    boxes: ParameterBoxes
    numpy_controller: Controller

    #: Solve-time parameters at a measured state and a time, keyed by emitted name.
    params_at: Callable[[np.ndarray, float], dict]
    #: Rotor thrusts from a solver's state: returns (saturated, raw) input moves.
    first_input: Callable[[np.ndarray], tuple]
    #: The NumPy controller's state vector after a given number of iterations at
    #: the representative parameters, for the equivalence check.
    numpy_solution: Callable[[int], np.ndarray]

    #: Absolute agreement the generated solver must reach against the reference
    #: oracle during ``make verify``.  ``None`` keeps the generator's default.
    #: Raise it only when the iterates are large at the sampled parameters, and
    #: see :func:`common.codegen_driver.config_for` for why that is not the same
    #: as weakening the check.
    tol_verify: float | None = None

    ir: object = field(default=None, repr=False)

    @property
    def codegen_dir(self) -> Path:
        return self.out / "codegen"

    @property
    def emitted(self) -> Path:
        return self.codegen_dir / "emitted"

    @property
    def optimized(self) -> Path:
        return self.codegen_dir / "optimized"


class GeneratedController:
    """A generated C solver behind the :class:`Controller` protocol.

    Everything outside the solve is the study's own code — the same parameters,
    the same first input move, the same actuator saturation — so a difference
    between this and the NumPy controller is a difference between two
    implementations of one algorithm and nothing else.
    """

    def __init__(self, study: Study, solver: CSolver, label: str = "generated C"):
        self.study = study
        self.solver = solver
        self.label = label
        self.dt = study.problem.dt
        self.A = study.problem.A
        self.B = study.problem.B
        self.n = study.problem.n

        expected = solver.state_dim
        got = len(study.numpy_solution(1))
        if expected != got:
            raise ValueError(
                f"{label} was generated for a state of dimension {expected}, "
                f"but this program needs {got}"
            )

        # A scheduled solver reads its coefficients from tables sized at
        # generation time.  Past the end the emitted accessor clamps to the last
        # entry, which is memory safe but freezes the schedule, so the solver
        # would quietly stop running the method it was generated for.  A fixed
        # coefficient has no such horizon.
        self.max_iterations = (
            solver.default_max_iter if isinstance(study.recording, ScheduledRecording) else None
        )

    def describe(self) -> str:
        return (
            f"{self.label}: state dimension {self.solver.state_dim}, "
            f"parameters {self.solver.param_dims}\nloaded from {self.solver.directory}"
        )

    def command(self, state: np.ndarray, t: float, iterations: int) -> Command:
        if self.max_iterations is not None and iterations > self.max_iterations:
            raise ValueError(
                f"{self.label} was generated with schedule tables covering "
                f"{self.max_iterations} iterations but {iterations} were requested; past "
                f"that the emitted accessor clamps and the coefficients freeze. "
                f"Regenerate with a larger max_iters, or lower the budget."
            )
        solution = self.solver.solve(iterations=iterations, **self.study.params_at(state, t))
        u, raw = self.study.first_input(solution)
        return Command(
            u=u,
            # The emitted loop only forms a residual when it is generated with an
            # early-exit criterion; under a fixed iteration count it may leave
            # this alone, so it is displayed rather than relied on.
            residual=float(self.solver.residual()),
            saturated=bool(np.any(np.abs(raw - u) > 1e-9)),
        )


# -- description ----------------------------------------------------------------


def describe_ir(ir) -> str:
    """What was extracted, and how much arithmetic one iteration of it is."""
    # Count only what the emitted matvecs actually touch: a block reached by no
    # term is never applied, and the C emitter drops entries below 1e-12, which
    # is where the near-zero base blocks of a schedule fit end up.
    applied = {
        t.matrix for op in [s.linear_op for s in ir.body] + [ir.state_update] for t in op.terms
    }
    matrix_nnz = sum(
        int(np.count_nonzero(np.abs(ir.matrices[n].array) > 1e-12)) for n in sorted(applied)
    )
    factor_nnz = sum(f.nnz for f in ir.factorizations.values())
    methods = ", ".join(f"{n}: {f.method}" for n, f in ir.factorizations.items()) or "none"
    lines = [
        f"extracted iteration:  {len(ir.body)} steps, state dimension {ir.state_dim}",
        f"parameters:           {', '.join(f'{p.name} ({p.dim})' for p in ir.params)}",
        f"coefficient blocks:   {len(ir.matrices)}, {matrix_nnz} applied entries",
        f"factorization:        {methods}, {factor_nnz} stored entries",
        f"work per iteration:   {matrix_nnz + 2 * factor_nnz} multiply-adds",
    ]
    if ir.schedules:
        length = min(len(v) for v in ir.schedules.values())
        lines.insert(
            2,
            f"schedules:            {len(ir.schedules)} tables of {length} "
            f"({', '.join(sorted(ir.schedules))})",
        )
    return "\n".join(lines)


def check_recording(study: Study, horizon: int, tol: float = 1e-9) -> object:
    """Run the recording three ways and confirm they are one algorithm.

    1. the NumPy controller the closed loop flies with;
    2. ``run_faithful``, which evaluates the recorded step objects directly,
       solving the affine system rather than reusing any folded inverse;
    3. ``ReferenceOracle``, which runs the extracted representation, the object
       code generation lowers to C.

    Agreement between the first two says the hand-written controller is the
    recorded algorithm.  Agreement with the third says extraction preserved it,
    which is the precondition for generating a solver from this recording.
    Extraction runs with the recorded iterates supplied, so ``check_stationary``
    also confirms that repeating one iteration reproduces the recording.

    A scheduled recording has no single recorded tape to replay, so step 2 does
    not apply to it: its build function is probed at synthetic scalars, and the
    check that the fit is right is the held-out probe inside
    ``extract_scheduled_ir``, which raises rather than returning a wrong IR.
    Steps 1 and 3 still run, and the study's own check script is where the
    recorded recurrence is compared against the algorithm as the paper writes it.

    Returns the IR; raises :class:`SystemExit` if anything disagrees.
    """
    from fomtools.codegen import ReferenceOracle
    from fomtools.codegen.oracle import run_faithful

    rec, boxes = study.recording, study.boxes
    reference = study.numpy_solution(horizon)
    scheduled = isinstance(rec, ScheduledRecording)

    err_faithful = None
    if not scheduled:
        faithful = run_faithful(
            rec.verifier,
            rec.next_state,
            rec.param_values(boxes),
            n_iters=horizon,
            state=rec.state,
            params=rec.params,
        )
        err_faithful = float(np.max(np.abs(faithful[-1] - reference)))

    ir = codegen_driver.extract(
        rec,
        boxes,
        config=codegen_driver.config_for(horizon, study.tol_verify),
        name=f"mpc_{study.name}",
    )
    study.ir = ir
    oracle = ReferenceOracle(ir).solve(param_values=boxes.named_values())
    err_oracle = float(np.max(np.abs(oracle.solution - reference)))

    print()
    print(describe_ir(ir))
    print()
    if err_faithful is not None:
        print(f"  NumPy vs recorded steps      {err_faithful:.3e}")
    print(f"  NumPy vs extracted iteration {err_oracle:.3e}")

    checked = [("extracted iteration", err_oracle)]
    if err_faithful is not None:
        checked.insert(0, ("recorded steps", err_faithful))
    failures = [name for name, err in checked if not err < tol]
    if failures:
        raise SystemExit(f"FAILED: disagreement with {', '.join(failures)}")
    return ir


def check_equivalence(study: Study, solver: CSolver, label: str, verbose: bool = True) -> float:
    """Compare a generated solver against the NumPy controller at several counts."""
    values = study.boxes.named_values()
    worst = 0.0
    if verbose:
        print(f"\n{label}: iterations   max |C - NumPy|")
    for k in CHECK_ITERATIONS:
        err = float(np.max(np.abs(solver.solve(iterations=k, **values) - study.numpy_solution(k))))
        worst = max(worst, err)
        if verbose:
            print(f"  {k:<9d}  {err:.3e}")
    return worst


# -- the three entry points -----------------------------------------------------


def generate(study: Study, force: bool = False, max_iters: int | None = None) -> None:
    """Emit both C solvers and check them against the NumPy controller."""
    config = codegen_driver.config_for(
        max_iters or codegen_driver.DEFAULT_MAX_ITERS, study.tol_verify
    )
    ir, emitted = codegen_driver.translate(
        study.recording,
        study.boxes,
        study.codegen_dir,
        name=f"mpc_{study.name}",
        config=config,
        force=force,
    )
    study.ir = ir
    print()
    print(describe_ir(ir))

    result = codegen_driver.optimize(ir, emitted, study.codegen_dir, config=config, force=force)
    print()
    if result.kept:
        print(f"optimization kept: {result.path}")
    else:
        print(f"optimization NOT kept ({result.notes})")
        print("the optimized directory holds the translated solver, so both arms are the same")
    if result.emitted_us and result.optimized_us:
        print(
            f"emitted benchmark {result.emitted_us:.1f} us/solve, "
            f"optimized {result.optimized_us:.1f} us/solve "
            f"({result.emitted_us / result.optimized_us:.2f}x)"
        )

    values = study.boxes.named_values()
    print("\nimplementation   cost and agreement with the NumPy controller")
    worst = 0.0
    for label, directory in (("emitted C", emitted), ("optimized C", result.path)):
        solver = CSolver(directory)
        err = check_equivalence(study, solver, label, verbose=False)
        worst = max(worst, err)
        print(
            f"  {label:<12s} {solver.us_per_iteration(values):8.2f} us/iteration   agrees to {err:.3e}"
        )
    if worst > TOL:
        raise SystemExit(f"FAILED: a generated solver disagrees with NumPy by {worst:.3e}")
    print(f"\nOK: both generated solvers agree with the NumPy controller to {worst:.3e}.")


def calibrate(study: Study, force: bool = False, max_iters: int | None = None) -> None:
    """Translate only, then report what one iteration costs in each implementation."""
    config = codegen_driver.config_for(
        max_iters or codegen_driver.DEFAULT_MAX_ITERS, study.tol_verify
    )
    ir, emitted = codegen_driver.translate(
        study.recording,
        study.boxes,
        study.codegen_dir,
        name=f"mpc_{study.name}",
        config=config,
        force=force,
    )
    study.ir = ir
    print()
    print(describe_ir(ir))

    solver = CSolver(emitted)
    worst = check_equivalence(study, solver, "emitted C")

    values = study.boxes.named_values()
    numpy_us = us_per_iteration(study.numpy_controller, study.ref)
    c_us = solver.us_per_iteration(values)
    per_solve = solver.us_per_solve(values, iterations=100)

    print("\nimplementation      us/iteration    spread   iterations in 10 ms")
    for label, cost in (("NumPy", numpy_us), ("emitted C", c_us)):
        print(
            f"  {label:<17s} {cost.us:9.2f}   {cost.spread_pct:5.1f}%      {10_000 / cost.us:8.0f}"
        )
    print(f"\nemitted C, 100 iterations end to end: {per_solve:.1f} us")
    print(f"per-solve overhead outside the loop:  {per_solve - 100 * c_us.us:.1f} us")
    print(f"emission changes the iteration cost by {numpy_us.us / c_us.us:.2f}x")

    if worst > TOL:
        raise SystemExit(
            f"FAILED: the generated solver and the NumPy controller disagree by {worst:.3e}"
        )
    print(f"\nOK: the two implementations agree to {worst:.3e} at every iteration count checked.")


@dataclass
class Arm:
    """One implementation of the iteration, and what it costs to run."""

    name: str
    controller: object
    cost: object  # common.simulate.IterationCost

    @property
    def us_per_iteration(self) -> float:
        return self.cost.us

    def iterations_for(self, budget_us: float) -> int:
        return max(1, int(budget_us / self.us_per_iteration))


#: Slope endpoints for timing an arm, in iterations.  The default 20-to-200 is
#: too short once an iteration is fast: at 0.13 us the whole 200-iteration leg is
#: 26 us, comparable to the per-solve marshalling the slope exists to cancel, and
#: the measurement comes back with tens of percent of spread.  A longer leg fixes
#: it -- the same solver measures 0.19 us +/- 41% at 20-to-200 and 0.133 us
#: +/- 1.3% at 64-to-512.  This is not cosmetic: every arm's measured cost is
#: what converts a budget into an iteration count, so a noisy measurement
#: silently hands the fastest arm a smaller budget than it earned.
SLOPE_LOW = 64
SLOPE_HIGH = 512


def build_arms(study: Study, only: list[str] | None = None) -> list[Arm]:
    """The three implementations, each timed the same way: as a slope in the loop count."""
    values = study.boxes.named_values()
    high = SLOPE_HIGH
    if isinstance(study.recording, ScheduledRecording):
        # Never slope past the schedule tables: beyond them the coefficients
        # freeze, so the far endpoint would be timing a different computation.
        high = min(high, study.recording.max_iters)
    arms = [
        Arm(
            "NumPy",
            study.numpy_controller,
            us_per_iteration(study.numpy_controller, study.ref, low=SLOPE_LOW, high=high),
        )
    ]  # timed by the same two-point slope as the generated solvers
    for name, directory in (("emitted C", study.emitted), ("optimized C", study.optimized)):
        if not (directory / "src" / "fom_solver.c").exists():
            print(f"skipping {name}: nothing generated in {directory}")
            continue
        solver = CSolver(directory)
        arms.append(
            Arm(
                name,
                GeneratedController(study, solver, name),
                solver.us_per_iteration(
                    values, low=SLOPE_LOW, high=high, repeats=15, solves=40, trials=9
                ),
            )
        )
    if only:
        arms = [a for a in arms if a.name in only]
    return arms


def cross_check(arms: list[Arm], ref, iterations: int, duration: float) -> None:
    """Fly every arm at the *same* iteration count and confirm they agree.

    This is what licenses reading the figures as a statement about speed.  If two
    arms flown at one iteration count part company, they are not implementations
    of one algorithm and the comparison means nothing.
    """
    print(f"\ncross-check: every implementation at {iterations} iterations")
    flights = {a.name: closed_loop(a.controller, ref, iterations, duration=duration) for a in arms}
    base = flights[arms[0].name]
    for name, roll in flights.items():
        steps = min(roll.steps, base.steps)
        gap = float(np.max(np.abs(roll.states[: steps + 1] - base.states[: steps + 1])))
        print(f"  {name:<13s} max state difference from {arms[0].name}: {gap:.3e}")
        if gap > 1e-6:
            raise SystemExit(
                f"FAILED: {name} and {arms[0].name} fly differently at equal iteration counts"
            )


def parser() -> argparse.ArgumentParser:
    """The command line the budget study takes."""
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=8.0, help="flight length in seconds")
    p.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS_US))
    p.add_argument("--figure-budget", type=float, default=None, help="budget drawn in the figures")
    p.add_argument("--only", nargs="+", help="restrict to these implementations")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="simulation steps per animation frame; default plays in real time at --fps",
    )
    p.add_argument("--cross-check-iterations", type=int, default=200)
    return p


def run_implementations(study: Study, args) -> None:
    """Fly every implementation at every budget, and draw the comparison."""
    arms = build_arms(study, args.only)
    if not arms:
        raise SystemExit("no implementations to fly")

    print(f"\n{study.name}: {study.problem.n} variables\n")
    print("implementation   us/iteration    spread   iterations in 10 ms")
    for arm in arms:
        rate = 10_000 / arm.us_per_iteration
        print(
            f"  {arm.name:<14s} {arm.us_per_iteration:9.2f}   {arm.cost.spread_pct:5.1f}%"
            f"      {rate:8.0f}"
        )

    cross_check(arms, study.ref, args.cross_check_iterations, args.duration)

    figure_budget = args.figure_budget if args.figure_budget is not None else max(args.budgets)
    budgets = sorted(set(args.budgets) | {figure_budget})

    print()
    rows = []
    flights: dict[tuple[str, float], Rollout] = {}
    for arm in arms:
        for budget in budgets:
            k = arm.iterations_for(budget)
            roll = closed_loop(arm.controller, study.ref, k, duration=args.duration)
            flights[(arm.name, budget)] = roll
            rows.append(
                {
                    "formulation": study.name,
                    "implementation": arm.name,
                    "budget_us": budget,
                    "us_per_iteration": round(arm.us_per_iteration, 4),
                    "us_spread_pct": round(arm.cost.spread_pct, 2),
                    "iterations": k,
                    "crashed": int(roll.crashed),
                    "flight_time_s": round(roll.steps * roll.dt, 3),
                    "steady_error_m": round(roll.steady_error(), 6),
                    "mean_error_m": round(roll.mean_error, 6),
                    "max_error_m": round(roll.max_error, 6),
                    "saturation_rate": round(roll.saturation_rate, 4),
                    "max_tilt_deg": round(roll.max_tilt_deg, 2),
                    "tilt_over_limit_pct": round(100 * roll.tilt_exceedance(TILT_LIMIT_DEG), 2),
                }
            )
            status = "CRASHED" if roll.crashed else f"{roll.steady_error():7.3f} m"
            print(
                f"  {arm.name:<14s} {budget / 1000:5.2f} ms -> {k:6d} iterations   {status}"
                f"   max tilt {roll.max_tilt_deg:4.1f} deg"
                f"   over limit {100 * roll.tilt_exceedance(TILT_LIMIT_DEG):4.1f}%"
            )

    out = study.out
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "implementations.csv", rows)
    _plot_budget(out / "implementations_budget.pdf", rows)

    chosen = [flights[(a.name, figure_budget)] for a in arms]
    labels = [f"{a.name} ({flights[(a.name, figure_budget)].iterations} iterations)" for a in arms]
    colors = [COLORS[a.name] for a in arms]
    print(f"\nfigures at a {figure_budget / 1000:.2f} ms budget")
    print(
        "wrote",
        plots.still_comparison(
            chosen,
            out / "implementations_flight.pdf",
            labels=labels,
            colors=colors,
            suptitle=plots.budget_title(figure_budget),
        ),
    )
    print(
        "wrote",
        plots.error_vs_time(
            chosen, out / "implementations_error.pdf", labels=labels, colors=colors
        ),
    )
    if not args.no_gif:
        print(
            "wrote",
            plots.make_gif(
                chosen,
                out / "implementations.gif",
                labels=labels,
                colors=colors,
                fps=args.fps,
                frame_stride=args.frame_stride,
            ),
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote", path)


def _plot_budget(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    names = list(dict.fromkeys(r["implementation"] for r in rows))
    ceiling = max((r["steady_error_m"] for r in rows if not r["crashed"]), default=1.0)
    for name in names:
        subset = [r for r in rows if r["implementation"] == name]
        budgets = [r["budget_us"] / 1000 for r in subset]
        errors = [np.nan if r["crashed"] else r["steady_error_m"] for r in subset]
        color = COLORS.get(name)
        ax.loglog(budgets, errors, "o-", color=color, linewidth=1.6, markersize=4, label=name)
        for row, budget in zip(subset, budgets):
            if row["crashed"]:
                ax.plot(budget, 2 * ceiling, "x", color=color, markersize=9, markeredgewidth=2)
    ax.set_xlabel("compute budget per sampling instant (ms)")
    ax.set_ylabel("steady tracking error (m)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
