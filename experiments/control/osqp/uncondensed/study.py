"""The uncondensed formulation as a :class:`common.study.Study`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.quadcopter import N_INPUT
from common.recording import ParameterBoxes
from common.reference import figure_eight
from common.study import Study
from uncondensed import admm, recording
from uncondensed.controller import UncondensedController
from uncondensed.mpc import build_mpc

OUT = Path(__file__).parent / "out"


def load_boxes(problem, ref, force: bool = False) -> ParameterBoxes:
    """The parameter boxes, cached — they cost a well-converged eight-second flight."""
    path = OUT / "parameter_box.npz"
    if path.exists() and not force:
        try:
            return ParameterBoxes.load(path)
        except KeyError:
            pass  # written by an older version; recompute
    boxes = recording.parameter_boxes(problem, ref)
    print(f"wrote {boxes.save(path)}")
    return boxes


def build(force_boxes: bool = False) -> Study:
    """Assemble the uncondensed study."""
    problem = build_mpc()
    ref = figure_eight()
    boxes = load_boxes(problem, ref, force=force_boxes)
    rec = recording.record(problem, boxes)

    def params_at(state: np.ndarray, t: float) -> dict:
        return {
            "p0": problem.linear_cost(problem.reference_window(ref, t)),
            "p1": np.asarray(state, dtype=float),
        }

    def first_input(solution: np.ndarray):
        start = problem.input_offset
        raw = solution[start : start + N_INPUT]
        return problem.first_input(solution[: problem.n]), raw

    def numpy_solution(iterations: int) -> np.ndarray:
        lo, hi = problem.bounds(boxes.state)
        result = admm.solve(problem, boxes["p0"].mid, lo, hi, iterations)
        return np.concatenate([result.zeta, result.v])

    return Study(
        name="uncondensed",
        problem=problem,
        ref=ref,
        out=OUT,
        recording=rec,
        boxes=boxes,
        numpy_controller=UncondensedController(problem, ref),
        params_at=params_at,
        first_input=first_input,
        numpy_solution=numpy_solution,
    )
