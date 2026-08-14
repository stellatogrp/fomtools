"""The condensed formulation as a :class:`common.study.Study`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.quadcopter import N_INPUT, U_MAX, U_MIN
from common.recording import ParameterBoxes
from common.reference import figure_eight
from common.study import Study
from condensed import admm, recording
from condensed.controller import CondensedController
from condensed.mpc import build_mpc

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
    """Assemble the condensed study."""
    problem = build_mpc()
    ref = figure_eight()
    boxes = load_boxes(problem, ref, force=force_boxes)
    rec = recording.record(problem, boxes)

    def params_at(state: np.ndarray, t: float) -> dict:
        return {"p0": problem.linear_cost(state, problem.reference_window(ref, t))}

    def first_input(solution: np.ndarray):
        raw = solution[:N_INPUT]
        return np.clip(raw, U_MIN, U_MAX), raw

    def numpy_solution(iterations: int) -> np.ndarray:
        result = admm.solve(problem, boxes["p0"].mid, iterations)
        return np.concatenate([result.z, result.v])

    return Study(
        name="condensed",
        problem=problem,
        ref=ref,
        out=OUT,
        recording=rec,
        boxes=boxes,
        numpy_controller=CondensedController(problem, ref),
        params_at=params_at,
        first_input=first_input,
        numpy_solution=numpy_solution,
    )
